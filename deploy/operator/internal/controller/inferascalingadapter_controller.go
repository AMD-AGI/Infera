/*
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/

package controller

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

// InferaScalingAdapterReconciler keeps an adapter's status in step with the
// workload it scales.
//
// It deliberately does not write the workload. The InferaDeployment reconciler
// reads adapters when it builds children, so there is exactly one writer of any
// child `.Spec`. Two writers would not merely race -- that reconciler assigns
// the whole spec on every pass, so the loser is reverted within seconds, which
// is precisely the failure an HPA pointed at the child Deployment hits today.
//
// What this controller owns is the half `/scale` reads back: `status.replicas`
// from the live workload, and `status.selector`, without which HorizontalPod-
// Autoscaler refuses to scale the resource at all.
type InferaScalingAdapterReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=infera.amd.com,resources=inferascalingadapters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=infera.amd.com,resources=inferascalingadapters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=infera.amd.com,resources=inferascalingadapters/scale,verbs=get;update;patch

func (r *InferaScalingAdapterReconciler) Reconcile(
	ctx context.Context, req ctrl.Request,
) (ctrl.Result, error) {
	lg := log.FromContext(ctx)

	adapter := &inferav1alpha1.InferaScalingAdapter{}
	if err := r.Get(ctx, req.NamespacedName, adapter); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !adapter.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	st := inferav1alpha1.InferaScalingAdapterStatus{
		ObservedGeneration: adapter.Generation,
	}

	idep := &inferav1alpha1.InferaDeployment{}
	err := r.Get(ctx, types.NamespacedName{
		Name: adapter.Spec.DeploymentRef, Namespace: adapter.Namespace,
	}, idep)
	switch {
	case apierrors.IsNotFound(err):
		return r.degraded(ctx, adapter, st, "TargetNotFound",
			fmt.Sprintf("no InferaDeployment %q in this namespace", adapter.Spec.DeploymentRef))
	case err != nil:
		return ctrl.Result{}, err
	}

	svc, ok := idep.Spec.Services[adapter.Spec.ServiceName]
	if !ok {
		return r.degraded(ctx, adapter, st, "ServiceNotFound",
			fmt.Sprintf("InferaDeployment %q has no service %q",
				adapter.Spec.DeploymentRef, adapter.Spec.ServiceName))
	}

	// The selector must be a serialized string, not a structured selector --
	// that is what the scale subresource contract requires, and HPA rejects a
	// target without one.
	st.Selector = labels.SelectorFromSet(
		labelsFor(idep.Name, adapter.Spec.ServiceName)).String()

	name := idep.Name + "-" + adapter.Spec.ServiceName
	key := types.NamespacedName{Name: name, Namespace: idep.Namespace}
	if svc.NumberOfNodes > 1 {
		u := &unstructured.Unstructured{}
		u.SetGroupVersionKind(lwsGVK())
		if err := r.Get(ctx, key, u); err == nil {
			if v, ok, _ := unstructured.NestedInt64(u.Object, "status", "replicas"); ok {
				st.Replicas = int32(v)
			}
			if v, ok, _ := unstructured.NestedInt64(u.Object, "status", "readyReplicas"); ok {
				st.ReadyReplicas = int32(v)
			}
		}
	} else {
		dep := &appsv1.Deployment{}
		if err := r.Get(ctx, key, dep); err == nil {
			st.Replicas = dep.Status.Replicas
			st.ReadyReplicas = dep.Status.ReadyReplicas
		}
	}

	msg := "adapter is inert: spec.replicas unset, the InferaDeployment's own replicas apply"
	if adapter.Spec.Replicas != nil {
		msg = fmt.Sprintf("driving %s/%s to %d replica(s)",
			adapter.Spec.DeploymentRef, adapter.Spec.ServiceName, *adapter.Spec.Replicas)
	}
	setCondition(&st.Conditions, adapter.Generation, "Ready", metav1.ConditionTrue, "Resolved", msg)
	setCondition(&st.Conditions, adapter.Generation, "Degraded", metav1.ConditionFalse, "Resolved", msg)

	lg.V(1).Info("scaling adapter reconciled", "target", adapter.Spec.DeploymentRef,
		"service", adapter.Spec.ServiceName, "observed", st.Replicas)
	return r.writeStatus(ctx, adapter, st)
}

func (r *InferaScalingAdapterReconciler) degraded(
	ctx context.Context, a *inferav1alpha1.InferaScalingAdapter,
	st inferav1alpha1.InferaScalingAdapterStatus, reason, msg string,
) (ctrl.Result, error) {
	setCondition(&st.Conditions, a.Generation, "Ready", metav1.ConditionFalse, reason, msg)
	setCondition(&st.Conditions, a.Generation, "Degraded", metav1.ConditionTrue, reason, msg)
	res, err := r.writeStatus(ctx, a, st)
	if err != nil {
		return res, err
	}
	// A dangling adapter usually means the target has not been created yet, so
	// retry rather than waiting for an event on an object that does not exist.
	return ctrl.Result{RequeueAfter: 30 * time.Second}, nil
}

func (r *InferaScalingAdapterReconciler) writeStatus(
	ctx context.Context, a *inferav1alpha1.InferaScalingAdapter,
	st inferav1alpha1.InferaScalingAdapterStatus,
) (ctrl.Result, error) {
	if equalStatus(a.Status, st) {
		return ctrl.Result{}, nil
	}
	a.Status = st
	return ctrl.Result{}, r.Status().Update(ctx, a)
}

func equalStatus(a, b inferav1alpha1.InferaScalingAdapterStatus) bool {
	if a.Replicas != b.Replicas || a.ReadyReplicas != b.ReadyReplicas ||
		a.Selector != b.Selector || a.ObservedGeneration != b.ObservedGeneration ||
		len(a.Conditions) != len(b.Conditions) {
		return false
	}
	for i := range a.Conditions {
		if a.Conditions[i].Type != b.Conditions[i].Type ||
			a.Conditions[i].Status != b.Conditions[i].Status ||
			a.Conditions[i].Reason != b.Conditions[i].Reason ||
			a.Conditions[i].Message != b.Conditions[i].Message {
			return false
		}
	}
	return true
}

func setCondition(
	conds *[]metav1.Condition, gen int64, typ string,
	status metav1.ConditionStatus, reason, msg string,
) {
	for i := range *conds {
		if (*conds)[i].Type == typ {
			c := &(*conds)[i]
			if c.Status != status {
				c.LastTransitionTime = metav1.Now()
			}
			c.Status, c.Reason, c.Message, c.ObservedGeneration = status, reason, msg, gen
			return
		}
	}
	*conds = append(*conds, metav1.Condition{
		Type: typ, Status: status, Reason: reason, Message: msg,
		ObservedGeneration: gen, LastTransitionTime: metav1.Now(),
	})
}

func (r *InferaScalingAdapterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	// Watching the InferaDeployment matters as much as the adapter itself: a
	// scale write only changes `spec.replicas` here, and the workload does not
	// move until the other reconciler runs. Without this the adapter's status
	// would lag by a resync period after every scale.
	return ctrl.NewControllerManagedBy(mgr).
		For(&inferav1alpha1.InferaScalingAdapter{}).
		Watches(
			&inferav1alpha1.InferaDeployment{},
			handler.EnqueueRequestsFromMapFunc(r.adaptersForDeployment),
		).
		Complete(r)
}

func (r *InferaScalingAdapterReconciler) adaptersForDeployment(
	ctx context.Context, obj client.Object,
) []reconcile.Request {
	list := &inferav1alpha1.InferaScalingAdapterList{}
	if err := r.List(ctx, list, client.InNamespace(obj.GetNamespace())); err != nil {
		return nil
	}
	var out []reconcile.Request
	for i := range list.Items {
		if list.Items[i].Spec.DeploymentRef != obj.GetName() {
			continue
		}
		out = append(out, reconcile.Request{NamespacedName: types.NamespacedName{
			Name: list.Items[i].Name, Namespace: list.Items[i].Namespace,
		}})
	}
	return out
}
