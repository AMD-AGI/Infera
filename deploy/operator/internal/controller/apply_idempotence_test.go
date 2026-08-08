/*
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/

package controller

import (
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

// The operator writes three fields of a LeaderWorkerSet; the API server fills
// in the rest from the CRD's defaults -- startupPolicy, rolloutStrategy,
// leaderWorkerTemplate.restartPolicy and more, nine of them on LWS v1.
//
// Replacing the whole .spec strips every one of those on each pass, the API
// server puts them back, and the next pass strips them again. That was a
// wasted write every resync; it becomes a hot loop now that the reconciler
// watches LeaderWorkerSet, because the write it just made enqueues the
// request that makes the next one.
//
// So: reconciling an object that is already in the desired state must not
// write to it.
func TestApplyingTheSameLwsTwiceDoesNotWriteAgain(t *testing.T) {
	s := testScheme(t)
	if err := inferav1alpha1.AddToScheme(s); err != nil {
		t.Fatalf("scheme: %v", err)
	}
	c := fake.NewClientBuilder().WithScheme(s).Build()
	r := &InferaDeploymentReconciler{Client: c, Scheme: s}
	ctx := context.Background()

	idep := &inferav1alpha1.InferaDeployment{}
	idep.Name = "qwen"
	idep.Namespace = "default"
	idep.UID = "uid-1"

	desired := func() *unstructured.Unstructured {
		u := &unstructured.Unstructured{}
		u.SetGroupVersionKind(lwsGVK())
		u.SetName("qwen-worker")
		u.SetNamespace("default")
		_ = unstructured.SetNestedField(u.Object, int64(2), "spec", "replicas")
		_ = unstructured.SetNestedField(u.Object, int64(2),
			"spec", "leaderWorkerTemplate", "size")
		return u
	}

	if err := r.applyUnstructured(ctx, idep, desired()); err != nil {
		t.Fatalf("first apply: %v", err)
	}

	// Stand in for the API server defaulting the fields the operator omits.
	live := &unstructured.Unstructured{}
	live.SetGroupVersionKind(lwsGVK())
	if err := c.Get(ctx, types.NamespacedName{Name: "qwen-worker", Namespace: "default"}, live); err != nil {
		t.Fatalf("get after create: %v", err)
	}
	_ = unstructured.SetNestedField(live.Object, "LeaderCreated", "spec", "startupPolicy")
	_ = unstructured.SetNestedField(live.Object, "RollingUpdate", "spec", "rolloutStrategy", "type")
	_ = unstructured.SetNestedField(live.Object, "RecreateGroupOnPodRestart",
		"spec", "leaderWorkerTemplate", "restartPolicy")
	if err := c.Update(ctx, live); err != nil {
		t.Fatalf("apply defaults: %v", err)
	}
	before := live.GetResourceVersion()

	if err := r.applyUnstructured(ctx, idep, desired()); err != nil {
		t.Fatalf("second apply: %v", err)
	}

	after := &unstructured.Unstructured{}
	after.SetGroupVersionKind(lwsGVK())
	if err := c.Get(ctx, types.NamespacedName{Name: "qwen-worker", Namespace: "default"}, after); err != nil {
		t.Fatalf("get after reconcile: %v", err)
	}

	if got := after.GetResourceVersion(); got != before {
		t.Errorf("reconcile rewrote an unchanged object (resourceVersion %s -> %s); "+
			"with the LWS watch registered this is a write loop", before, got)
	}
	for _, f := range [][]string{
		{"spec", "startupPolicy"},
		{"spec", "rolloutStrategy", "type"},
		{"spec", "leaderWorkerTemplate", "restartPolicy"},
	} {
		if v, ok, _ := unstructured.NestedString(after.Object, f...); !ok || v == "" {
			t.Errorf("%v was stripped; the API server will re-default it and the "+
				"next pass strips it again", f)
		}
	}
}

// Merging must not turn into "never update": a genuine spec change still has
// to reach the child, or scaling through the CR would silently do nothing.
func TestApplyStillPushesAChangedField(t *testing.T) {
	s := testScheme(t)
	c := fake.NewClientBuilder().WithScheme(s).Build()
	r := &InferaDeploymentReconciler{Client: c, Scheme: s}
	ctx := context.Background()

	idep := &inferav1alpha1.InferaDeployment{}
	idep.Name = "qwen"
	idep.Namespace = "default"
	idep.UID = "uid-1"

	build := func(replicas int64) *unstructured.Unstructured {
		u := &unstructured.Unstructured{}
		u.SetGroupVersionKind(lwsGVK())
		u.SetName("qwen-worker")
		u.SetNamespace("default")
		_ = unstructured.SetNestedField(u.Object, replicas, "spec", "replicas")
		return u
	}

	if err := r.applyUnstructured(ctx, idep, build(2)); err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := r.applyUnstructured(ctx, idep, build(5)); err != nil {
		t.Fatalf("scale: %v", err)
	}

	got := &unstructured.Unstructured{}
	got.SetGroupVersionKind(lwsGVK())
	if err := c.Get(ctx, types.NamespacedName{Name: "qwen-worker", Namespace: "default"}, got); err != nil {
		t.Fatalf("get: %v", err)
	}
	if v, _, _ := unstructured.NestedInt64(got.Object, "spec", "replicas"); v != 5 {
		t.Fatalf("replicas = %d, want 5 -- scaling through the CR did not land", v)
	}
}
