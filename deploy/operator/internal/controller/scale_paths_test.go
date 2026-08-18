/*
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/

package controller

import (
	"context"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

// Which of the three ways to write a replica count actually reaches the pods.
// These are easy to conflate and they behave differently on purpose, so the
// distinction is pinned here rather than left to a README.

func scaleScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := testScheme(t)
	if err := rbacv1.AddToScheme(s); err != nil {
		t.Fatalf("add rbac scheme: %v", err)
	}
	return s
}

func idepWith(replicas int32) *inferav1alpha1.InferaDeployment {
	r := replicas
	return &inferav1alpha1.InferaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "qwen", Namespace: "ns"},
		Spec: inferav1alpha1.InferaDeploymentSpec{
			Image: "infera:test",
			Services: map[string]inferav1alpha1.ServiceSpec{
				"decode": {
					ComponentType: inferav1alpha1.ComponentTypeWorker,
					Replicas:      &r,
					NumberOfNodes: 1,
				},
			},
		},
	}
}

func reconcileOnce(t *testing.T, cl client.Client, s *runtime.Scheme) {
	t.Helper()
	r := &InferaDeploymentReconciler{Client: cl, Scheme: s}
	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "qwen", Namespace: "ns"},
	})
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}
}

func childReplicas(t *testing.T, cl client.Client) int32 {
	t.Helper()
	dep := &appsv1.Deployment{}
	key := types.NamespacedName{Name: "qwen-decode", Namespace: "ns"}
	if err := cl.Get(context.Background(), key, dep); err != nil {
		t.Fatalf("get child Deployment: %v", err)
	}
	if dep.Spec.Replicas == nil {
		t.Fatal("child Deployment has no replicas set")
	}
	return *dep.Spec.Replicas
}

// Editing the CR is the normal path and it works: the CR is the desired state,
// so a change to it is what reconciliation exists to propagate.
func TestEditingTheCRScales(t *testing.T) {
	s := scaleScheme(t)
	idep := idepWith(2)
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(idep).
		WithStatusSubresource(idep).Build()

	reconcileOnce(t, cl, s)
	if got := childReplicas(t, cl); got != 2 {
		t.Fatalf("initial: child has %d replicas, want 2", got)
	}

	// The user edits the CR.
	live := &inferav1alpha1.InferaDeployment{}
	key := types.NamespacedName{Name: "qwen", Namespace: "ns"}
	if err := cl.Get(context.Background(), key, live); err != nil {
		t.Fatalf("get idep: %v", err)
	}
	five := int32(5)
	svc := live.Spec.Services["decode"]
	svc.Replicas = &five
	live.Spec.Services["decode"] = svc
	if err := cl.Update(context.Background(), live); err != nil {
		t.Fatalf("update idep: %v", err)
	}

	reconcileOnce(t, cl, s)
	if got := childReplicas(t, cl); got != 5 {
		t.Fatalf("after editing the CR: child has %d replicas, want 5", got)
	}
}

// Editing the *child* is the path that does not survive, and that is the
// intended behaviour of any operator: the child is derived state, so the next
// pass restores it from the CR. The write succeeds and nothing reports an
// error, which is why pointing an autoscaler at the generated Deployment looks
// like it works right up until the next reconcile.
func TestEditingTheChildIsReverted(t *testing.T) {
	s := scaleScheme(t)
	idep := idepWith(2)
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(idep).
		WithStatusSubresource(idep).Build()

	reconcileOnce(t, cl, s)

	// Something scales the generated Deployment directly.
	dep := &appsv1.Deployment{}
	key := types.NamespacedName{Name: "qwen-decode", Namespace: "ns"}
	if err := cl.Get(context.Background(), key, dep); err != nil {
		t.Fatalf("get child: %v", err)
	}
	three := int32(3)
	dep.Spec.Replicas = &three
	if err := cl.Update(context.Background(), dep); err != nil {
		t.Fatalf("update child: %v", err)
	}

	reconcileOnce(t, cl, s)
	if got := childReplicas(t, cl); got != 2 {
		t.Fatalf("child edit survived reconciliation: %d, want it reverted to the CR's 2", got)
	}
}

// Scaling down through the CR has to land on pods that drain, not pods that
// get cut. The two features are built separately -- the replica count comes
// from the CR, the graceful shutdown from what the operator injects into the
// pod template -- so this checks they meet: a Deployment produced by a normal
// reconcile carries the preStop delay and a grace period long enough to cover
// the whole shutdown.
//
// Without preStop the router keeps assigning work for the entire termination;
// without the grace covering preStop + drain + teardown the kubelet SIGKILLs
// mid-drain. Either one silently turns a graceful scale-down back into a kill.
func TestScalingDownThroughTheCRLandsOnDrainablePods(t *testing.T) {
	s := scaleScheme(t)
	idep := idepWith(3)
	// A long drain, the case where a fixed grace period used to fall short.
	svc := idep.Spec.Services["decode"]
	svc.Args = []string{"--drain-timeout", "300"}
	idep.Spec.Services["decode"] = svc

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(idep).
		WithStatusSubresource(idep).Build()
	reconcileOnce(t, cl, s)

	dep := &appsv1.Deployment{}
	key := types.NamespacedName{Name: "qwen-decode", Namespace: "ns"}
	if err := cl.Get(context.Background(), key, dep); err != nil {
		t.Fatalf("get child: %v", err)
	}
	spec := dep.Spec.Template.Spec

	if len(spec.Containers) == 0 {
		t.Fatal("no containers in the pod template")
	}
	lc := spec.Containers[0].Lifecycle
	if lc == nil || lc.PreStop == nil || lc.PreStop.Exec == nil {
		t.Fatal("no preStop hook: the router would keep routing to a condemned pod")
	}

	if spec.TerminationGracePeriodSeconds == nil {
		t.Fatal("no terminationGracePeriodSeconds: the kubelet default is 30s, far under a 300s drain")
	}
	want := int64(workerPreStopDrainSeconds + 300 + workerTeardownHeadroomSeconds)
	if got := *spec.TerminationGracePeriodSeconds; got != want {
		t.Fatalf("grace = %ds, want %ds (preStop %d + drain 300 + teardown %d)",
			got, want, workerPreStopDrainSeconds, workerTeardownHeadroomSeconds)
	}

	// And the scale-down itself still works on that same object.
	live := &inferav1alpha1.InferaDeployment{}
	if err := cl.Get(context.Background(), types.NamespacedName{Name: "qwen", Namespace: "ns"}, live); err != nil {
		t.Fatalf("get idep: %v", err)
	}
	one := int32(1)
	svc = live.Spec.Services["decode"]
	svc.Replicas = &one
	live.Spec.Services["decode"] = svc
	if err := cl.Update(context.Background(), live); err != nil {
		t.Fatalf("update idep: %v", err)
	}
	reconcileOnce(t, cl, s)
	if got := childReplicas(t, cl); got != 1 {
		t.Fatalf("scale down through the CR: child has %d replicas, want 1", got)
	}
}

// Multi-node workers are torn down a whole group at a time, so the same
// guarantees have to hold on the LWS path -- where the pod template travels
// through a different builder.
func TestMultiNodePodsAlsoDrain(t *testing.T) {
	s := scaleScheme(t)
	s.AddKnownTypeWithName(lwsGVK(), &unstructured.Unstructured{})

	idep := idepWith(2)
	svc := idep.Spec.Services["decode"]
	svc.NumberOfNodes = 3
	svc.Args = []string{"--drain-timeout", "180"}
	idep.Spec.Services["decode"] = svc

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(idep).
		WithStatusSubresource(idep).Build()
	reconcileOnce(t, cl, s)

	u := &unstructured.Unstructured{}
	u.SetGroupVersionKind(lwsGVK())
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "qwen-decode", Namespace: "ns"}, u); err != nil {
		t.Fatalf("get child LWS: %v", err)
	}

	grace, found, err := unstructured.NestedInt64(u.Object,
		"spec", "leaderWorkerTemplate", "workerTemplate", "spec", "terminationGracePeriodSeconds")
	if err != nil || !found {
		t.Fatalf("LWS pod template has no terminationGracePeriodSeconds (found=%v, err=%v)", found, err)
	}
	want := int64(workerPreStopDrainSeconds + 180 + workerTeardownHeadroomSeconds)
	if grace != want {
		t.Fatalf("LWS grace = %ds, want %ds", grace, want)
	}

	containers, found, err := unstructured.NestedSlice(u.Object,
		"spec", "leaderWorkerTemplate", "workerTemplate", "spec", "containers")
	if err != nil || !found || len(containers) == 0 {
		t.Fatalf("LWS pod template has no containers (found=%v, err=%v)", found, err)
	}
	c, _ := containers[0].(map[string]any)
	if _, ok := c["lifecycle"]; !ok {
		t.Fatal("LWS container has no lifecycle/preStop: a condemned group keeps receiving work")
	}
}

// A LeaderWorkerSet carries a real scale subresource of its own, so an HPA can
// be pointed straight at the generated LWS and the write will succeed. It still
// does not work, for the same reason it does not work on the generated
// Deployment: reconciliation assigns the whole child spec every pass, replicas
// included. The scale write lands, and the next reconcile overwrites it.
//
// This is worth pinning because the LWS case looks different from the outside
// -- `kubectl get lws` shows a scale subresource, HPA reports success, nothing
// errors -- and the only symptom is a replica count that keeps snapping back.
func TestEditingTheChildLWSIsAlsoReverted(t *testing.T) {
	s := scaleScheme(t)
	s.AddKnownTypeWithName(lwsGVK(), &unstructured.Unstructured{})
	s.AddKnownTypeWithName(lwsGVK().GroupVersion().WithKind(lwsKind+"List"),
		&unstructured.UnstructuredList{})

	idep := idepWith(2)
	svc := idep.Spec.Services["decode"]
	svc.NumberOfNodes = 3 // multi-node -> LeaderWorkerSet instead of Deployment
	idep.Spec.Services["decode"] = svc

	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(idep).
		WithStatusSubresource(idep).Build()
	reconcileOnce(t, cl, s)

	get := func() *unstructured.Unstructured {
		u := &unstructured.Unstructured{}
		u.SetGroupVersionKind(lwsGVK())
		key := types.NamespacedName{Name: "qwen-decode", Namespace: "ns"}
		if err := cl.Get(context.Background(), key, u); err != nil {
			t.Fatalf("get child LWS: %v", err)
		}
		return u
	}
	replicas := func(u *unstructured.Unstructured) int64 {
		v, found, err := unstructured.NestedInt64(u.Object, "spec", "replicas")
		if err != nil || !found {
			t.Fatalf("LWS has no spec.replicas (found=%v, err=%v)", found, err)
		}
		return v
	}

	lws := get()
	if got := replicas(lws); got != 2 {
		t.Fatalf("initial: LWS has %d groups, want 2", got)
	}

	// An HPA scales the LWS directly -- exactly what its scale subresource
	// invites, and exactly what does not survive.
	if err := unstructured.SetNestedField(lws.Object, int64(6), "spec", "replicas"); err != nil {
		t.Fatalf("set replicas: %v", err)
	}
	if err := cl.Update(context.Background(), lws); err != nil {
		t.Fatalf("update child LWS: %v", err)
	}
	if got := replicas(get()); got != 6 {
		t.Fatalf("precondition: the scale write itself must land, got %d", got)
	}

	reconcileOnce(t, cl, s)

	if got := replicas(get()); got != 2 {
		t.Fatalf("LWS scale survived reconciliation: %d groups, want it reverted to 2", got)
	}
}

// The server's scaling API writes replica counts back to the CR, which needs a
// grant the discovery identity did not previously carry. Every Pod in a
// deployment shares that identity, so the grant has to name the one CR it may
// touch: without ResourceNames a worker could resize the fleet it belongs to,
// or any other deployment in the namespace.
func TestTheDiscoveryRoleCanWriteOnlyItsOwnDeployment(t *testing.T) {
	idep := idepWith(2)
	role := buildDiscoveryRole(idep)

	var found *rbacv1.PolicyRule
	for i := range role.Rules {
		for _, res := range role.Rules[i].Resources {
			if res == "inferadeployments" {
				found = &role.Rules[i]
			}
		}
	}
	if found == nil {
		t.Fatal("no grant for inferadeployments: the scaling API would 403")
	}
	if len(found.ResourceNames) != 1 || found.ResourceNames[0] != idep.Name {
		t.Fatalf("ResourceNames = %v, want exactly [%s]: an unscoped grant lets "+
			"any Pod here resize any deployment in the namespace",
			found.ResourceNames, idep.Name)
	}
	for _, verb := range found.Verbs {
		switch verb {
		case "get", "patch":
		default:
			t.Errorf("verb %q is more than the scaling API needs", verb)
		}
	}
}

// Pods are a separate rule and must stay unscoped: the server lists and watches
// every worker Pod, which ResourceNames cannot express.
func TestThePodGrantIsUnchanged(t *testing.T) {
	role := buildDiscoveryRole(idepWith(1))
	for _, rule := range role.Rules {
		for _, res := range rule.Resources {
			if res != "pods" {
				continue
			}
			if len(rule.ResourceNames) != 0 {
				t.Fatal("the Pod grant must not be scoped by name; discovery lists all of them")
			}
		}
	}
}
