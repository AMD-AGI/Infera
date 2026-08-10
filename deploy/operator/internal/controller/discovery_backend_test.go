/*
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/

package controller

import (
	"context"
	"errors"
	"strings"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

// An operator-managed deployment is by definition running in Kubernetes, and
// there the orchestrator is what knows a worker is going away: a condemned Pod
// carries deletionTimestamp before the process is even signalled, and the
// registry drops it from routing then.
//
// Pointing such a deployment at an external etcd throws that away. The router
// stops watching Pods, so nothing reads the deletionTimestamp, and the only
// remaining signal is the worker announcing DRAINING after SIGTERM -- which
// arrives once the preStop delay the operator itself injects has elapsed. The
// combination keeps that delay and loses the early notice it exists to give,
// so for its whole duration the router keeps handing new work to a Pod that is
// already condemned. Refusing is better than rendering a deployment whose
// drain is worse than either backend alone.
func TestAnOperatorDeploymentRefusesTheExternalEtcdBackend(t *testing.T) {
	s := scaleScheme(t)
	idep := idepWith(1)
	idep.Spec.DiscoveryBackend = "etcd"
	idep.Spec.EtcdEndpoint = "etcd:2379"
	cl := fake.NewClientBuilder().WithScheme(s).
		WithObjects(idep).WithStatusSubresource(idep).Build()

	r := &InferaDeploymentReconciler{Client: cl, Scheme: s}
	res, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "qwen", Namespace: "ns"},
	})
	if err == nil {
		t.Fatal("reconcile accepted discoveryBackend=etcd; it must be refused")
	}
	if !strings.Contains(err.Error(), "discoveryBackend") {
		t.Fatalf("error should name the field so the cause is obvious, got: %v", err)
	}

	// No amount of retrying changes a spec field, and controller-runtime
	// re-queues a plain error with exponential backoff forever -- two error
	// logs and a status write on every attempt, and reconcile_errors_total
	// climbing until someone edits the CR. A terminal error is recorded once
	// and dropped.
	if !errors.Is(err, reconcile.TerminalError(nil)) {
		t.Errorf("error must be terminal, or the request is re-queued forever: %v", err)
	}
	if res.Requeue || res.RequeueAfter != 0 { //nolint:staticcheck // Requeue kept for clarity
		t.Errorf("refusal must not ask to be retried, got %+v", res)
	}

	// Refusing must not leave a half-built deployment behind.
	dep := &appsv1.Deployment{}
	key := types.NamespacedName{Name: "qwen-decode", Namespace: "ns"}
	if err := cl.Get(context.Background(), key, dep); err == nil {
		t.Fatal("a child workload was created for a configuration that was refused")
	}

	// The reason belongs on the object, not only in the operator's log.
	got := &inferav1alpha1.InferaDeployment{}
	if err := cl.Get(context.Background(), types.NamespacedName{Name: "qwen", Namespace: "ns"}, got); err != nil {
		t.Fatalf("get idep: %v", err)
	}
	if got.Status.State != inferav1alpha1.StateFailed {
		t.Errorf("status.state = %q, want %q so `kubectl get idep` shows it",
			got.Status.State, inferav1alpha1.StateFailed)
	}
}

// The default and an explicit "kubernetes" both reconcile normally.
func TestTheKubernetesBackendIsAccepted(t *testing.T) {
	for _, backend := range []string{"", "kubernetes"} {
		s := scaleScheme(t)
		idep := idepWith(1)
		idep.Spec.DiscoveryBackend = backend
		cl := fake.NewClientBuilder().WithScheme(s).
			WithObjects(idep).WithStatusSubresource(idep).Build()
		reconcileOnce(t, cl, s)

		dep := &appsv1.Deployment{}
		key := types.NamespacedName{Name: "qwen-decode", Namespace: "ns"}
		if err := cl.Get(context.Background(), key, dep); err != nil {
			t.Fatalf("discoveryBackend=%q: child Deployment not created: %v", backend, err)
		}
	}
}
