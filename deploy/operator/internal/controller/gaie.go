/*
 * Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Gateway API Inference Extension (GAIE) resources: the operator deploys an
 * Endpoint Picker (EPP, `python -m infera.gaie`), injects a direct-mode
 * frontend sidecar into worker pods, and wires an InferencePool + HTTPRoute so
 * a Kubernetes Inference Gateway routes by Infera's kv-aware policy.
 */

package controller

import (
	"context"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

const (
	eppGRPCPort       int32 = 9002
	eppHealthPort     int32 = 9003
	defaultFrontendPt int32 = 8000

	inferencePoolAPIVersion = "inference.networking.k8s.io/v1"
	inferencePoolKind       = "InferencePool"
	httpRouteAPIVersion     = "gateway.networking.k8s.io/v1"
	httpRouteKind           = "HTTPRoute"

	// gaieFrontendLabel marks worker pods that carry the direct-mode frontend
	// sidecar; the InferencePool selects on it so it never picks up the EPP or
	// other non-serving pods of the same deployment.
	gaieFrontendLabel = "infera.amd.com/gaie-frontend"
)

func gaieEnabled(idep *inferav1alpha1.InferaDeployment) bool {
	return idep.Spec.GAIE != nil && idep.Spec.GAIE.Enabled
}

func eppName(idepName string) string           { return idepName + "-epp" }
func inferencePoolName(idepName string) string { return idepName + "-pool" }
func httpRouteName(idepName string) string     { return idepName + "-route" }

func gaieFrontendPort(idep *inferav1alpha1.InferaDeployment) int32 {
	if idep.Spec.GAIE != nil && idep.Spec.GAIE.FrontendPort != 0 {
		return idep.Spec.GAIE.FrontendPort
	}
	return defaultFrontendPt
}

func gaieImage(idep *inferav1alpha1.InferaDeployment) string {
	if idep.Spec.GAIE != nil && idep.Spec.GAIE.EPPImage != "" {
		return idep.Spec.GAIE.EPPImage
	}
	return idep.Spec.Image
}

func gaieGatewayNamespace(idep *inferav1alpha1.InferaDeployment) string {
	if idep.Spec.GAIE != nil && idep.Spec.GAIE.GatewayNamespace != "" {
		return idep.Spec.GAIE.GatewayNamespace
	}
	return idep.Namespace
}

// gaieDiscoveryArgs are the discovery flags shared by the EPP and the frontend
// sidecar (both reuse the server's kubernetes/etcd discovery).
func gaieDiscoveryArgs(idep *inferav1alpha1.InferaDeployment) []string {
	if useK8sDiscovery(idep) {
		return []string{
			"--discovery-backend", "kubernetes",
			"--k8s-label-selector", discoveryLabelSelector(idep.Name),
		}
	}
	return []string{"--etcd-endpoint", idep.Spec.EtcdEndpoint}
}

// gaieEnv builds the env shared by EPP + frontend sidecar (NATS broker + pod
// identity for kubernetes discovery).
func gaieEnv(idep *inferav1alpha1.InferaDeployment) []corev1.EnvVar {
	env := append([]corev1.EnvVar{}, idep.Spec.Envs...)
	if natsEnabled(idep) {
		env = append(env, corev1.EnvVar{Name: "NATS_SERVER", Value: natsURL(idep.Name)})
	}
	if useK8sDiscovery(idep) {
		env = append(env,
			corev1.EnvVar{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
			corev1.EnvVar{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"}}},
		)
	}
	return env
}

// applyGAIEFrontendSidecar injects a `infera.server --router-mode direct`
// sidecar into a worker pod template and labels it so the InferencePool can
// select it. No-op unless GAIE is enabled and the service is a worker. The
// sidecar shares the pod network with the engine, so it reaches the local
// engine over localhost and remote PD peers via their registered URLs.
func applyGAIEFrontendSidecar(
	idep *inferav1alpha1.InferaDeployment,
	svc inferav1alpha1.ServiceSpec,
	tmpl *corev1.PodTemplateSpec,
) {
	if !gaieEnabled(idep) || svc.ComponentType != inferav1alpha1.ComponentTypeWorker {
		return
	}
	port := gaieFrontendPort(idep)
	cmd := []string{
		"python3", "-m", "infera.server",
		"--router-mode", "direct",
		"--host", "0.0.0.0",
		"--port", fmt.Sprintf("%d", port),
		"--router-tokenizer-path", idep.Spec.GAIE.TokenizerPath,
	}
	cmd = append(cmd, gaieDiscoveryArgs(idep)...)
	if natsEnabled(idep) {
		cmd = append(cmd,
			"--kv-event-transport", "nats", "--nats-server", natsURL(idep.Name),
			"--request-transport", "nats")
	} else {
		cmd = append(cmd, "--request-transport", "http")
	}
	env := gaieEnv(idep)
	if useK8sDiscovery(idep) {
		env = append(env, corev1.EnvVar{
			Name: "INFERA_K8S_LABEL_SELECTOR", Value: discoveryLabelSelector(idep.Name)})
	}
	sidecar := corev1.Container{
		Name:    "infera-frontend",
		Image:   gaieImage(idep),
		Command: cmd,
		Env:     env,
		Ports:   []corev1.ContainerPort{{Name: "frontend", ContainerPort: port}},
		ReadinessProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: "/v1/models", Port: intstr.FromInt32(port),
				},
			},
			InitialDelaySeconds: 5,
			PeriodSeconds:       10,
		},
	}
	tmpl.Spec.Containers = append(tmpl.Spec.Containers, sidecar)
	if tmpl.ObjectMeta.Labels == nil {
		tmpl.ObjectMeta.Labels = map[string]string{}
	}
	tmpl.ObjectMeta.Labels[gaieFrontendLabel] = "true"
}

// buildEPPDeployment runs the GAIE Endpoint Picker (ext_proc) that scores
// workers with the same kv-aware policy as the server.
func buildEPPDeployment(idep *inferav1alpha1.InferaDeployment) *appsv1.Deployment {
	lbls := labelsFor(idep.Name, "epp")
	one := int32(1)
	cmd := []string{
		"python3", "-m", "infera.gaie",
		"--router-tokenizer-path", idep.Spec.GAIE.TokenizerPath,
		"--grpc-port", fmt.Sprintf("%d", eppGRPCPort),
		"--grpc-health-port", fmt.Sprintf("%d", eppHealthPort),
	}
	cmd = append(cmd, gaieDiscoveryArgs(idep)...)
	if natsEnabled(idep) {
		cmd = append(cmd, "--kv-event-transport", "nats", "--nats-server", natsURL(idep.Name))
	}
	podSpec := corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:    "epp",
			Image:   gaieImage(idep),
			Command: cmd,
			Env:     gaieEnv(idep),
			Ports: []corev1.ContainerPort{
				{Name: "grpc", ContainerPort: eppGRPCPort},
				{Name: "grpc-health", ContainerPort: eppHealthPort},
			},
			ReadinessProbe: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromInt32(eppGRPCPort)},
				},
				InitialDelaySeconds: 5,
				PeriodSeconds:       10,
			},
		}},
	}
	if useK8sDiscovery(idep) {
		podSpec.ServiceAccountName = discoverySAName(idep.Name)
	}
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: eppName(idep.Name), Namespace: idep.Namespace, Labels: lbls},
		Spec: appsv1.DeploymentSpec{
			Replicas: &one,
			Selector: &metav1.LabelSelector{MatchLabels: lbls},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: lbls},
				Spec:       podSpec,
			},
		},
	}
}

// buildEPPService is the ClusterIP the InferencePool.endpointPickerRef targets.
func buildEPPService(idep *inferav1alpha1.InferaDeployment) *corev1.Service {
	lbls := labelsFor(idep.Name, "epp")
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: eppName(idep.Name), Namespace: idep.Namespace, Labels: lbls},
		Spec: corev1.ServiceSpec{
			Selector: lbls,
			Ports: []corev1.ServicePort{{
				Name: "grpc-ext-proc", Port: eppGRPCPort, TargetPort: intstr.FromInt32(eppGRPCPort),
			}},
		},
	}
}

// buildInferencePool selects the worker pods (frontend sidecar port) and points
// the gateway at the EPP for per-request endpoint selection. Unstructured to
// avoid a compile-time dependency on the GAIE CRD module (the CRD must be
// installed in the cluster).
func buildInferencePool(idep *inferav1alpha1.InferaDeployment) *unstructured.Unstructured {
	port := gaieFrontendPort(idep)
	u := &unstructured.Unstructured{}
	u.SetAPIVersion(inferencePoolAPIVersion)
	u.SetKind(inferencePoolKind)
	u.SetName(inferencePoolName(idep.Name))
	u.SetNamespace(idep.Namespace)
	u.SetLabels(labelsFor(idep.Name, "epp"))
	spec := map[string]any{
		"targetPorts": []any{map[string]any{"number": int64(port)}},
		"selector": map[string]any{
			"matchLabels": map[string]any{
				"infera.amd.com/deployment": idep.Name,
				gaieFrontendLabel:             "true",
			},
		},
		"endpointPickerRef": map[string]any{
			"group": "",
			"kind":  "Service",
			"name":  eppName(idep.Name),
			"port":  map[string]any{"number": int64(eppGRPCPort)},
		},
	}
	_ = unstructured.SetNestedMap(u.Object, spec, "spec")
	return u
}

// gaieStatus reads back the EPP Deployment, InferencePool, and HTTPRoute to
// report observed readiness. Best-effort: a missing object reads as not-ready
// rather than erroring (the next requeue retries).
func (r *InferaDeploymentReconciler) gaieStatus(
	ctx context.Context, idep *inferav1alpha1.InferaDeployment,
) *inferav1alpha1.GAIEStatus {
	st := &inferav1alpha1.GAIEStatus{
		InferencePoolName: inferencePoolName(idep.Name),
		HTTPRouteName:     httpRouteName(idep.Name),
	}

	epp := &appsv1.Deployment{}
	if err := r.Get(ctx,
		client.ObjectKey{Name: eppName(idep.Name), Namespace: idep.Namespace}, epp); err == nil {
		st.EPPReplicas = epp.Status.Replicas
		st.EPPReadyReplicas = epp.Status.ReadyReplicas
	}

	pool := &unstructured.Unstructured{}
	pool.SetAPIVersion(inferencePoolAPIVersion)
	pool.SetKind(inferencePoolKind)
	if err := r.Get(ctx,
		client.ObjectKey{Name: inferencePoolName(idep.Name), Namespace: idep.Namespace},
		pool); err == nil {
		// Accepted condition if reported; otherwise existence is enough (some
		// gateways don't write InferencePool conditions).
		st.InferencePoolReady = !hasAnyCondition(pool) || conditionTrue(pool, "Accepted")
	}

	route := &unstructured.Unstructured{}
	route.SetAPIVersion(httpRouteAPIVersion)
	route.SetKind(httpRouteKind)
	if err := r.Get(ctx,
		client.ObjectKey{Name: httpRouteName(idep.Name), Namespace: idep.Namespace},
		route); err == nil {
		st.HTTPRouteAccepted = conditionTrue(route, "Accepted")
	}

	st.Ready = st.EPPReadyReplicas > 0 && st.InferencePoolReady && st.HTTPRouteAccepted
	return st
}

// hasAnyCondition reports whether the object carries any status conditions
// (top-level or per-parent), so existence-only readiness can be distinguished
// from a gateway that does report conditions.
func hasAnyCondition(u *unstructured.Unstructured) bool {
	if conds, ok, _ := unstructured.NestedSlice(u.Object, "status", "conditions"); ok && len(conds) > 0 {
		return true
	}
	if parents, ok, _ := unstructured.NestedSlice(u.Object, "status", "parents"); ok {
		for _, p := range parents {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			if conds, ok, _ := unstructured.NestedSlice(pm, "conditions"); ok && len(conds) > 0 {
				return true
			}
		}
	}
	return false
}

// conditionTrue scans both the top-level status.conditions and the
// gateway-style status.parents[].conditions for condType with status "True".
func conditionTrue(u *unstructured.Unstructured, condType string) bool {
	if conds, ok, _ := unstructured.NestedSlice(u.Object, "status", "conditions"); ok {
		if conditionsContainTrue(conds, condType) {
			return true
		}
	}
	if parents, ok, _ := unstructured.NestedSlice(u.Object, "status", "parents"); ok {
		for _, p := range parents {
			pm, ok := p.(map[string]any)
			if !ok {
				continue
			}
			conds, ok, _ := unstructured.NestedSlice(pm, "conditions")
			if ok && conditionsContainTrue(conds, condType) {
				return true
			}
		}
	}
	return false
}

func conditionsContainTrue(conds []any, condType string) bool {
	for _, c := range conds {
		cm, ok := c.(map[string]any)
		if !ok {
			continue
		}
		t, _, _ := unstructured.NestedString(cm, "type")
		s, _, _ := unstructured.NestedString(cm, "status")
		if t == condType && s == "True" {
			return true
		}
	}
	return false
}

// buildHTTPRoute attaches the InferencePool to an existing Inference Gateway.
func buildHTTPRoute(idep *inferav1alpha1.InferaDeployment) *unstructured.Unstructured {
	u := &unstructured.Unstructured{}
	u.SetAPIVersion(httpRouteAPIVersion)
	u.SetKind(httpRouteKind)
	u.SetName(httpRouteName(idep.Name))
	u.SetNamespace(idep.Namespace)
	u.SetLabels(labelsFor(idep.Name, "epp"))

	parentRef := map[string]any{
		"group":     "gateway.networking.k8s.io",
		"kind":      "Gateway",
		"name":      idep.Spec.GAIE.GatewayName,
		"namespace": gaieGatewayNamespace(idep),
	}
	backendRef := map[string]any{
		"group":  "inference.networking.k8s.io",
		"kind":   "InferencePool",
		"name":   inferencePoolName(idep.Name),
		"port":   int64(gaieFrontendPort(idep)),
		"weight": int64(1),
	}
	rule := map[string]any{
		"backendRefs": []any{backendRef},
		"matches": []any{map[string]any{
			"path": map[string]any{"type": "PathPrefix", "value": "/"},
		}},
	}
	spec := map[string]any{
		"parentRefs": []any{parentRef},
		"rules":      []any{rule},
	}
	if idep.Spec.GAIE != nil && len(idep.Spec.GAIE.Hostnames) > 0 {
		hosts := make([]any, 0, len(idep.Spec.GAIE.Hostnames))
		for _, h := range idep.Spec.GAIE.Hostnames {
			hosts = append(hosts, h)
		}
		spec["hostnames"] = hosts
	}
	_ = unstructured.SetNestedMap(u.Object, spec, "spec")
	return u
}
