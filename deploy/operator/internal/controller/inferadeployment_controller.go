/*
 * Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

package controller

import (
	"context"
	"sort"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

// InferaDeploymentReconciler reconciles a InferaDeployment object.
type InferaDeploymentReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=infera.amd.com,resources=inferadeployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=infera.amd.com,resources=inferadeployments/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=infera.amd.com,resources=inferadeployments/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments;statefulsets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=leaderworkerset.x-k8s.io,resources=leaderworkersets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=inference.networking.k8s.io,resources=inferencepools,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=gateway.networking.k8s.io,resources=httproutes,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=serviceaccounts,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=rbac.authorization.k8s.io,resources=roles;rolebindings,verbs=get;list;watch;create;update;patch;delete
// pods perms are required both to grant them in the per-IDEP discovery Role
// (RBAC escalation-prevention: a grantor must hold what it grants) and so a
// future operator path could read Pod status directly.
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;patch

func (r *InferaDeploymentReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	lg := log.FromContext(ctx)

	idep := &inferav1alpha1.InferaDeployment{}
	if err := r.Get(ctx, req.NamespacedName, idep); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// Being deleted: stop reconciling immediately so we do not recreate the
	// child Deployments/StatefulSets/Services that job-manager (or Kubernetes
	// GC) is tearing down. Children carry owner references, so GC handles their
	// cleanup; the operator holds no finalizer and has nothing else to do here.
	if !idep.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	// 0. Kubernetes-native discovery RBAC: a namespaced ServiceAccount + Role so
	// workers can patch their own Pod annotation and the server can list/watch
	// this deployment's worker Pods (no external etcd).
	if useK8sDiscovery(idep) {
		if err := r.applyObject(ctx, idep, buildDiscoverySA(idep)); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.applyObject(ctx, idep, buildDiscoveryRole(idep)); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.applyObject(ctx, idep, buildDiscoveryRoleBinding(idep)); err != nil {
			return ctrl.Result{}, err
		}
	}

	// 1. Operator-managed NATS (JetStream) for the KV-event plane.
	if natsEnabled(idep) {
		if err := r.applyObject(ctx, idep, buildNATSService(idep)); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.applyObject(ctx, idep, buildNATSStatefulSet(idep)); err != nil {
			return ctrl.Result{}, err
		}
	}

	// 2. Each service -> Deployment (single-node) or LeaderWorkerSet (multi-node).
	status := map[string]inferav1alpha1.ServiceStatus{}
	for _, name := range sortedKeys(idep.Spec.Services) {
		svc := idep.Spec.Services[name]
		if svc.NumberOfNodes > 1 {
			lws := buildLeaderWorkerSet(idep, name, svc)
			if err := r.applyUnstructured(ctx, idep, lws); err != nil {
				return ctrl.Result{}, err
			}
			status[name] = r.lwsStatus(ctx, idep, name, svc)
		} else {
			dep := buildDeployment(idep, name, svc)
			if err := r.applyObject(ctx, idep, dep); err != nil {
				return ctrl.Result{}, err
			}
			status[name] = r.deploymentStatus(ctx, idep, name, svc)
		}
		// The router/server gets a ClusterIP Service for ingress.
		if svc.ComponentType == inferav1alpha1.ComponentTypeServer {
			if err := r.applyObject(ctx, idep, buildServerService(idep, name, svc)); err != nil {
				return ctrl.Result{}, err
			}
		}
	}

	// 2b. GAIE: Endpoint Picker (ext_proc) + InferencePool + HTTPRoute so a
	// Kubernetes Inference Gateway routes by Infera's kv-aware policy. The
	// per-worker frontend sidecar is injected in podTemplate above.
	if gaieEnabled(idep) {
		if err := r.applyObject(ctx, idep, buildEPPService(idep)); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.applyObject(ctx, idep, buildEPPDeployment(idep)); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.applyUnstructured(ctx, idep, buildInferencePool(idep)); err != nil {
			return ctrl.Result{}, err
		}
		if err := r.applyUnstructured(ctx, idep, buildHTTPRoute(idep)); err != nil {
			return ctrl.Result{}, err
		}
	}

	// 3. Status roll-up.
	idep.Status.ObservedGeneration = idep.Generation
	idep.Status.Services = status
	if gaieEnabled(idep) {
		idep.Status.GAIE = r.gaieStatus(ctx, idep)
	} else {
		idep.Status.GAIE = nil
	}
	idep.Status.State = rollupState(status)
	if err := r.Status().Update(ctx, idep); err != nil {
		lg.Error(err, "status update failed")
		return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
	}
	return ctrl.Result{RequeueAfter: 15 * time.Second}, nil
}

// applyObject create-or-updates a typed object, setting the owner reference.
func (r *InferaDeploymentReconciler) applyObject(ctx context.Context, idep *inferav1alpha1.InferaDeployment, desired client.Object) error {
	// Build a fresh empty object of the same kind keyed by name/namespace.
	existing := desired.DeepCopyObject().(client.Object)
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, existing, func() error {
		copySpec(existing, desired)
		existing.SetLabels(desired.GetLabels())
		return controllerutil.SetControllerReference(idep, existing, r.Scheme)
	})
	return err
}

// applyUnstructured create-or-updates the unstructured LeaderWorkerSet.
func (r *InferaDeploymentReconciler) applyUnstructured(ctx context.Context, idep *inferav1alpha1.InferaDeployment, desired *unstructured.Unstructured) error {
	existing := &unstructured.Unstructured{}
	existing.SetGroupVersionKind(desired.GroupVersionKind())
	existing.SetName(desired.GetName())
	existing.SetNamespace(desired.GetNamespace())
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, existing, func() error {
		spec, _, _ := unstructured.NestedMap(desired.Object, "spec")
		_ = unstructured.SetNestedMap(existing.Object, spec, "spec")
		existing.SetLabels(desired.GetLabels())
		return controllerutil.SetControllerReference(idep, existing, r.Scheme)
	})
	return err
}

func (r *InferaDeploymentReconciler) deploymentStatus(ctx context.Context, idep *inferav1alpha1.InferaDeployment, name string, svc inferav1alpha1.ServiceSpec) inferav1alpha1.ServiceStatus {
	st := inferav1alpha1.ServiceStatus{Kind: "Deployment", Replicas: replicasOf(svc)}
	dep := &appsv1.Deployment{}
	if err := r.Get(ctx, client.ObjectKey{Name: idep.Name + "-" + name, Namespace: idep.Namespace}, dep); err == nil {
		st.ReadyReplicas = dep.Status.ReadyReplicas
	}
	return st
}

func (r *InferaDeploymentReconciler) lwsStatus(ctx context.Context, idep *inferav1alpha1.InferaDeployment, name string, svc inferav1alpha1.ServiceSpec) inferav1alpha1.ServiceStatus {
	st := inferav1alpha1.ServiceStatus{Kind: "LeaderWorkerSet", Replicas: replicasOf(svc)}
	u := &unstructured.Unstructured{}
	u.SetGroupVersionKind(lwsGVK())
	if err := r.Get(ctx, client.ObjectKey{Name: idep.Name + "-" + name, Namespace: idep.Namespace}, u); err == nil {
		if v, ok, _ := unstructured.NestedInt64(u.Object, "status", "readyReplicas"); ok {
			st.ReadyReplicas = int32(v)
		}
	}
	return st
}

// copySpec copies the .Spec of the desired object onto existing for the kinds
// the operator manages (Deployment, StatefulSet, Service).
func copySpec(existing, desired client.Object) {
	switch d := desired.(type) {
	case *appsv1.Deployment:
		existing.(*appsv1.Deployment).Spec = d.Spec
	case *appsv1.StatefulSet:
		e := existing.(*appsv1.StatefulSet)
		// VolumeClaimTemplates are immutable after creation; only set on create.
		if e.CreationTimestamp.IsZero() {
			e.Spec = d.Spec
		} else {
			tmpl := e.Spec.VolumeClaimTemplates
			e.Spec = d.Spec
			e.Spec.VolumeClaimTemplates = tmpl
		}
	case *corev1.Service:
		// Preserve the immutable ClusterIP across updates.
		e := existing.(*corev1.Service)
		clusterIP := e.Spec.ClusterIP
		e.Spec = d.Spec
		if clusterIP != "" {
			e.Spec.ClusterIP = clusterIP
		}
	case *rbacv1.Role:
		existing.(*rbacv1.Role).Rules = d.Rules
	case *rbacv1.RoleBinding:
		e := existing.(*rbacv1.RoleBinding)
		// RoleRef is immutable after creation; only Subjects can change.
		e.Subjects = d.Subjects
	case *corev1.ServiceAccount:
		// No spec to copy; owner ref + labels handled by applyObject.
	}
}

func rollupState(svcs map[string]inferav1alpha1.ServiceStatus) inferav1alpha1.DeploymentState {
	if len(svcs) == 0 {
		return inferav1alpha1.StatePending
	}
	allReady := true
	for _, s := range svcs {
		if s.ReadyReplicas < s.Replicas || s.Replicas == 0 {
			allReady = false
		}
	}
	if allReady {
		return inferav1alpha1.StateReady
	}
	return inferav1alpha1.StatePending
}

func sortedKeys(m map[string]inferav1alpha1.ServiceSpec) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// SetupWithManager registers the controller.
func (r *InferaDeploymentReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&inferav1alpha1.InferaDeployment{}).
		Owns(&appsv1.Deployment{}).
		Owns(&appsv1.StatefulSet{}).
		Owns(&corev1.Service{}).
		Complete(r)
}
