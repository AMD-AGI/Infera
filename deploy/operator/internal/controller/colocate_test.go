/*
 * Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

package controller

import (
	"context"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

// routerWorkerIDEP returns a server plus one mixed worker with the given replicas.
func routerWorkerIDEP(workerReplicas int32) *inferav1alpha1.InferaDeployment {
	one := int32(1)
	r := workerReplicas
	return &inferav1alpha1.InferaDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "qwen", Namespace: "ns"},
		Spec: inferav1alpha1.InferaDeploymentSpec{
			Image: "infera:test",
			Services: map[string]inferav1alpha1.ServiceSpec{
				"role0": {ComponentType: inferav1alpha1.ComponentTypeServer, Replicas: &one},
				"role1": {
					ComponentType: inferav1alpha1.ComponentTypeWorker,
					Role:          inferav1alpha1.WorkerRoleMixed,
					Replicas:      &r,
					NumberOfNodes: 1,
					Resources:     &inferav1alpha1.Resources{CPU: "8"},
				},
			},
		},
	}
}

func TestColocationFor(t *testing.T) {
	if c, ok := colocationFor(routerWorkerIDEP(1)); !ok || c.server != "role0" || c.worker != "role1" {
		t.Fatalf("router + one worker: got %+v ok=%v", c, ok)
	}
	if _, ok := colocationFor(routerWorkerIDEP(2)); ok {
		t.Fatal("worker replicas=2 must not colocate")
	}

	multi := routerWorkerIDEP(1)
	w := multi.Spec.Services["role1"]
	w.NumberOfNodes = 2
	multi.Spec.Services["role1"] = w
	if _, ok := colocationFor(multi); ok {
		t.Fatal("multi-node worker must not colocate")
	}

	pd := routerWorkerIDEP(1)
	w = pd.Spec.Services["role1"]
	w.Role = inferav1alpha1.WorkerRolePrefill
	pd.Spec.Services["role1"] = w
	pd.Spec.Services["role2"] = inferav1alpha1.ServiceSpec{
		ComponentType: inferav1alpha1.ComponentTypeWorker, Role: inferav1alpha1.WorkerRoleDecode,
	}
	if _, ok := colocationFor(pd); ok {
		t.Fatal("router + 1P1D must not colocate")
	}
}

func TestColocatedWorkerTemplate(t *testing.T) {
	idep := routerWorkerIDEP(1)
	tmpl := podTemplate(idep, "role1", idep.Spec.Services["role1"])
	if len(tmpl.Spec.Containers) != 2 {
		t.Fatalf("want worker + server containers, got %d", len(tmpl.Spec.Containers))
	}
	server := tmpl.Spec.Containers[1]
	if server.Name != colocatedServerContainer {
		t.Fatalf("second container name = %q", server.Name)
	}
	if len(server.Ports) != 1 || server.Ports[0].ContainerPort != defaultServerPort {
		t.Fatalf("server ports = %+v", server.Ports)
	}
	if len(server.Resources.Requests) != 0 || len(server.Resources.Limits) != 0 {
		t.Fatalf("colocated server must not request resources: %+v", server.Resources)
	}
	if tmpl.Labels[routerLabel] != "true" {
		t.Fatal("colocated worker pod must carry the router label")
	}

	split := routerWorkerIDEP(2)
	wt := podTemplate(split, "role1", split.Spec.Services["role1"])
	if len(wt.Spec.Containers) != 1 || wt.Labels[routerLabel] != "" {
		t.Fatal("non-colocated worker must stay a single container without router label")
	}
	st := podTemplate(split, "role0", split.Spec.Services["role0"])
	if st.Labels[routerLabel] != "true" {
		t.Fatal("standalone server pod must carry the router label")
	}
}

func TestColocatedExtraPodSpecVolumes(t *testing.T) {
	idep := routerWorkerIDEP(1)
	srv := idep.Spec.Services["role0"]
	srv.ExtraPodSpec = &corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main", Image: "router:x",
			VolumeMounts: []corev1.VolumeMount{{Name: "launcher", MountPath: "/shared-data"}}}},
		Volumes: []corev1.Volume{{Name: "launcher"}, {Name: "boot-config"}},
	}
	idep.Spec.Services["role0"] = srv
	tmpl := podTemplate(idep, "role1", idep.Spec.Services["role1"])
	if tmpl.Spec.Containers[1].Image != "router:x" {
		t.Fatalf("server image = %q", tmpl.Spec.Containers[1].Image)
	}
	count := map[string]int{}
	for _, v := range tmpl.Spec.Volumes {
		count[v.Name]++
	}
	if count["launcher"] != 1 || count["boot-config"] != 1 {
		t.Fatalf("volumes must merge by name: %v", count)
	}
}

func TestColocationPortConflict(t *testing.T) {
	idep := routerWorkerIDEP(1)
	srv := idep.Spec.Services["role0"]
	srv.Port = defaultWorkerPort
	idep.Spec.Services["role0"] = srv
	c, _ := colocationFor(idep)
	if colocationPortConflict(idep, c) == nil {
		t.Fatal("server port equal to worker port must conflict")
	}
	ok := routerWorkerIDEP(1)
	c, _ = colocationFor(ok)
	if err := colocationPortConflict(ok, c); err != nil {
		t.Fatalf("default ports must not conflict: %v", err)
	}
}

func TestReconcileColocated(t *testing.T) {
	s := scaleScheme(t)
	idep := routerWorkerIDEP(1)
	stale := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: "qwen-role0", Namespace: "ns"}}
	idep.UID = "uid-qwen"
	ctrlTrue := true
	stale.OwnerReferences = []metav1.OwnerReference{{
		APIVersion: inferav1alpha1.GroupVersion.String(), Kind: "InferaDeployment",
		Name: "qwen", UID: idep.UID, Controller: &ctrlTrue,
	}}
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(idep, stale).WithStatusSubresource(idep).Build()
	reconcileOnce(t, cl, s)

	ctx := context.Background()
	err := cl.Get(ctx, types.NamespacedName{Name: "qwen-role0", Namespace: "ns"}, &appsv1.Deployment{})
	if !apierrors.IsNotFound(err) {
		t.Fatalf("server Deployment must be removed when colocated, err=%v", err)
	}
	dep := &appsv1.Deployment{}
	if err := cl.Get(ctx, types.NamespacedName{Name: "qwen-role1", Namespace: "ns"}, dep); err != nil {
		t.Fatalf("get worker Deployment: %v", err)
	}
	if len(dep.Spec.Template.Spec.Containers) != 2 {
		t.Fatalf("worker pod containers = %d", len(dep.Spec.Template.Spec.Containers))
	}
	svc := &corev1.Service{}
	if err := cl.Get(ctx, types.NamespacedName{Name: "qwen-role0", Namespace: "ns"}, svc); err != nil {
		t.Fatalf("get server Service: %v", err)
	}
	if svc.Spec.Selector[labelKeyService] != "role1" {
		t.Fatalf("server Service must select the worker pod: %v", svc.Spec.Selector)
	}
}

func TestReconcileColocatedPortConflictRefused(t *testing.T) {
	s := scaleScheme(t)
	idep := routerWorkerIDEP(1)
	srv := idep.Spec.Services["role0"]
	srv.Port = defaultWorkerPort
	idep.Spec.Services["role0"] = srv
	cl := fake.NewClientBuilder().WithScheme(s).WithObjects(idep).WithStatusSubresource(idep).Build()
	r := &InferaDeploymentReconciler{Client: cl, Scheme: s}
	_, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Name: "qwen", Namespace: "ns"},
	})
	if err == nil {
		t.Fatal("port conflict must be refused")
	}
	got := &inferav1alpha1.InferaDeployment{}
	_ = cl.Get(context.Background(), client.ObjectKeyFromObject(idep), got)
	if got.Status.State != inferav1alpha1.StateFailed {
		t.Fatalf("state = %q", got.Status.State)
	}
}
