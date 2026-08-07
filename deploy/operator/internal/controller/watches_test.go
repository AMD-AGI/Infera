/*
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/

package controller

import (
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

func testScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	if err := inferav1alpha1.AddToScheme(s); err != nil {
		t.Fatalf("add infera scheme: %v", err)
	}
	if err := appsv1.AddToScheme(s); err != nil {
		t.Fatalf("add apps scheme: %v", err)
	}
	if err := corev1.AddToScheme(s); err != nil {
		t.Fatalf("add core scheme: %v", err)
	}
	return s
}

// The LWS watch is registered only when the CRD is served. controller-runtime
// builds an informer per watched type at startup and one for an unserved kind
// fails the manager, so a single-node cluster without LWS installed must not
// have the operator refuse to start.
func TestLwsInstalled(t *testing.T) {
	empty := meta.NewDefaultRESTMapper(nil)
	if lwsInstalled(empty) {
		t.Fatal("no LWS CRD: reported installed, the manager would fail to start")
	}

	withLWS := meta.NewDefaultRESTMapper([]schema.GroupVersion{lwsGVK().GroupVersion()})
	withLWS.Add(lwsGVK(), meta.RESTScopeNamespace)
	if !lwsInstalled(withLWS) {
		t.Fatal("LWS CRD present: reported missing, multi-node status would lag a resync")
	}
}
