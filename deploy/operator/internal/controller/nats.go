/*
 * Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

package controller

import (
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

func natsLabels(idepName string) map[string]string {
	return map[string]string{
		"app.kubernetes.io/managed-by": "infera-operator",
		"infera.amd.com/deployment":  idepName,
		"infera.amd.com/component":   "nats",
	}
}

func natsImage(idep *inferav1alpha1.InferaDeployment) string {
	if idep.Spec.NATS != nil && idep.Spec.NATS.Image != "" {
		return idep.Spec.NATS.Image
	}
	return "nats:latest"
}

func natsStorageSize(idep *inferav1alpha1.InferaDeployment) string {
	if idep.Spec.NATS != nil && idep.Spec.NATS.StorageSize != "" {
		return idep.Spec.NATS.StorageSize
	}
	return "4Gi"
}

// buildNATSService is the headless service so pods reach NATS at <name>-nats:4222.
func buildNATSService(idep *inferav1alpha1.InferaDeployment) *corev1.Service {
	lbls := natsLabels(idep.Name)
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: natsName(idep.Name), Namespace: idep.Namespace, Labels: lbls},
		Spec: corev1.ServiceSpec{
			ClusterIP: corev1.ClusterIPNone,
			Selector:  lbls,
			Ports: []corev1.ServicePort{
				{Name: "client", Port: natsClientPort},
				{Name: "monitor", Port: natsMonitorPort},
			},
		},
	}
}

// buildNATSStatefulSet creates a single-replica JetStream NATS whose store_dir
// is driven by NATS_STORE_DIR pointing at the mounted PVC (survives restarts).
// Scale to an odd replica count for RAFT quorum in production.
func buildNATSStatefulSet(idep *inferav1alpha1.InferaDeployment) *appsv1.StatefulSet {
	lbls := natsLabels(idep.Name)
	one := int32(1)
	storage := resource.MustParse(natsStorageSize(idep))
	pvcSpec := corev1.PersistentVolumeClaimSpec{
		AccessModes: []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
		Resources:   corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{corev1.ResourceStorage: storage}},
	}
	if idep.Spec.NATS != nil && idep.Spec.NATS.StorageClassName != "" {
		sc := idep.Spec.NATS.StorageClassName
		pvcSpec.StorageClassName = &sc
	}
	return &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: natsName(idep.Name), Namespace: idep.Namespace, Labels: lbls},
		Spec: appsv1.StatefulSetSpec{
			ServiceName: natsName(idep.Name),
			Replicas:    &one,
			Selector:    &metav1.LabelSelector{MatchLabels: lbls},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: lbls},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "nats",
						Image: natsImage(idep),
						// K8s expands $(NATS_STORE_DIR) in args from the env below.
						Args: []string{"-js", "-sd", "$(NATS_STORE_DIR)", "-m", "8222"},
						Env:  []corev1.EnvVar{{Name: "NATS_STORE_DIR", Value: "/data/jetstream"}},
						Ports: []corev1.ContainerPort{
							{Name: "client", ContainerPort: natsClientPort},
							{Name: "monitor", ContainerPort: natsMonitorPort},
						},
						VolumeMounts: []corev1.VolumeMount{{Name: "data", MountPath: "/data"}},
					}},
				},
			},
			VolumeClaimTemplates: []corev1.PersistentVolumeClaim{{
				ObjectMeta: metav1.ObjectMeta{Name: "data"},
				Spec:       pvcSpec,
			}},
		},
	}
}

// toUnstructured converts a typed object to a map for embedding in an
// unstructured resource (used for the LeaderWorkerSet pod template).
func toUnstructured(obj any) (map[string]any, error) {
	m, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// lwsGVK is the GroupVersionKind for LeaderWorkerSet (unstructured).
func lwsGVK() schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: "leaderworkerset.x-k8s.io", Version: "v1", Kind: lwsKind}
}
