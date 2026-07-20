/*
 * Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

package controller

import (
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/intstr"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

const (
	defaultServerPort int32 = 8000
	defaultWorkerPort int32 = 30000
	defaultGPUType          = "amd.com/gpu"
	natsClientPort    int32 = 4222
	natsMonitorPort   int32 = 8222

	lwsAPIVersion = "leaderworkerset.x-k8s.io/v1"
	lwsKind       = "LeaderWorkerSet"

	// Graceful rolling-upgrade tuning for GPU worker pods.
	workerPreStopDrainSeconds           = 15  // preStop sleep: let the router drop us before SIGTERM
	workerTerminationGraceSeconds int64 = 120 // must exceed preStop + the worker --drain-timeout
)

// labelsFor returns the selector/identity labels for a service's workload.
func labelsFor(idepName, svcName string) map[string]string {
	return map[string]string{
		"app.kubernetes.io/managed-by": "infera-operator",
		"infera.amd.com/deployment":  idepName,
		"infera.amd.com/service":     svcName,
	}
}

// podLabelsFor returns the operator's selector labels merged with any
// caller-supplied ServiceSpec.PodLabels (e.g. an external orchestrator's
// workload-id label used by its pod syncer). Operator selector labels always
// win on key conflict so Service selection and ownership remain intact.
func podLabelsFor(idepName, svcName string, svc inferav1alpha1.ServiceSpec) map[string]string {
	base := labelsFor(idepName, svcName)
	if len(svc.PodLabels) == 0 {
		return base
	}
	merged := make(map[string]string, len(base)+len(svc.PodLabels))
	for k, v := range svc.PodLabels {
		merged[k] = v
	}
	for k, v := range base {
		merged[k] = v // operator labels take precedence
	}
	return merged
}

func natsName(idepName string) string { return idepName + "-nats" }

// useK8sDiscovery reports whether the deployment uses Kubernetes-native worker
// discovery (the default) rather than an external etcd.
func useK8sDiscovery(idep *inferav1alpha1.InferaDeployment) bool {
	return idep.Spec.DiscoveryBackend != "etcd"
}

// discoverySAName is the ServiceAccount the operator provisions for k8s
// discovery (workers patch their own Pod, the server lists/watches Pods).
func discoverySAName(idepName string) string { return idepName + "-disc" }

// discoveryLabelSelector scopes the server's Pod watch to this deployment's
// workers (all pods of an IDEP carry infera.amd.com/deployment=<name>).
func discoveryLabelSelector(idepName string) string {
	return "infera.amd.com/deployment=" + idepName
}

func natsURL(idepName string) string {
	return fmt.Sprintf("nats://%s:%d", natsName(idepName), natsClientPort)
}

func servicePort(svc inferav1alpha1.ServiceSpec) int32 {
	if svc.Port != 0 {
		return svc.Port
	}
	if svc.ComponentType == inferav1alpha1.ComponentTypeServer {
		return defaultServerPort
	}
	return defaultWorkerPort
}

func imageFor(idep *inferav1alpha1.InferaDeployment, svc inferav1alpha1.ServiceSpec) string {
	if svc.Image != "" {
		return svc.Image
	}
	return idep.Spec.Image
}

func replicasOf(svc inferav1alpha1.ServiceSpec) int32 {
	if svc.Replicas != nil {
		return *svc.Replicas
	}
	return 1
}

// containerCommand builds the infera entrypoint + operator-injected flags,
// then appends the user's free-form Args (model-path, tokenizer, tp-size, ...).
func containerCommand(idep *inferav1alpha1.InferaDeployment, svc inferav1alpha1.ServiceSpec) []string {
	port := servicePort(svc)
	k8sDisc := useK8sDiscovery(idep)
	cmd := []string{"python3", "-m"}
	if svc.ComponentType == inferav1alpha1.ComponentTypeServer {
		cmd = append(cmd, "infera.server",
			"--host", "0.0.0.0",
			"--port", fmt.Sprintf("%d", port),
		)
		if k8sDisc {
			cmd = append(cmd, "--discovery-backend", "kubernetes",
				"--k8s-label-selector", discoveryLabelSelector(idep.Name))
		} else {
			cmd = append(cmd, "--etcd-endpoint", idep.Spec.EtcdEndpoint)
		}
	} else {
		cmd = append(cmd, "infera.engine."+backend(idep),
			"--host", "0.0.0.0",
			"--port", fmt.Sprintf("%d", port),
		)
		if k8sDisc {
			cmd = append(cmd, "--discovery-backend", "kubernetes")
		} else {
			cmd = append(cmd, "--etcd-endpoint", idep.Spec.EtcdEndpoint)
		}
		if svc.Role == inferav1alpha1.WorkerRolePrefill || svc.Role == inferav1alpha1.WorkerRoleDecode {
			cmd = append(cmd, "--disaggregation-mode", string(svc.Role))
		}
	}
	// KV-event plane over the managed NATS broker when enabled.
	if natsEnabled(idep) {
		cmd = append(cmd, "--kv-event-transport", "nats", "--nats-server", natsURL(idep.Name))
	}
	return append(cmd, svc.Args...)
}

func backend(idep *inferav1alpha1.InferaDeployment) string {
	if idep.Spec.BackendFramework == "vllm" {
		return "vllm"
	}
	return "sglang"
}

func natsEnabled(idep *inferav1alpha1.InferaDeployment) bool {
	return idep.Spec.NATS == nil || idep.Spec.NATS.Deploy
}

func envFor(idep *inferav1alpha1.InferaDeployment, svc inferav1alpha1.ServiceSpec) []corev1.EnvVar {
	env := []corev1.EnvVar{}
	env = append(env, idep.Spec.Envs...)
	if natsEnabled(idep) {
		env = append(env, corev1.EnvVar{Name: "NATS_SERVER", Value: natsURL(idep.Name)})
	}
	if useK8sDiscovery(idep) {
		// Pod identity for self-registration (worker) + selector for the server.
		env = append(env,
			corev1.EnvVar{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
			corev1.EnvVar{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"}}},
		)
		if svc.ComponentType == inferav1alpha1.ComponentTypeServer {
			env = append(env, corev1.EnvVar{
				Name: "INFERA_K8S_LABEL_SELECTOR", Value: discoveryLabelSelector(idep.Name)})
		}
	}
	env = append(env, svc.Env...)
	return env
}

func resourceRequirements(svc inferav1alpha1.ServiceSpec) corev1.ResourceRequirements {
	req := corev1.ResourceRequirements{
		Requests: corev1.ResourceList{},
		Limits:   corev1.ResourceList{},
	}
	if svc.Resources == nil {
		return req
	}
	if svc.Resources.CPU != "" {
		req.Requests[corev1.ResourceCPU] = resource.MustParse(svc.Resources.CPU)
	}
	if svc.Resources.Memory != "" {
		req.Requests[corev1.ResourceMemory] = resource.MustParse(svc.Resources.Memory)
	}
	if svc.Resources.GPU > 0 {
		gpuType := svc.Resources.GPUType
		if gpuType == "" {
			gpuType = defaultGPUType
		}
		q := resource.MustParse(fmt.Sprintf("%d", svc.Resources.GPU))
		req.Requests[corev1.ResourceName(gpuType)] = q
		req.Limits[corev1.ResourceName(gpuType)] = q
	}
	return req
}

// mainContainerNames are the container names an external orchestrator may use
// for the primary infera container inside ExtraPodSpec.
var mainContainerNames = map[string]struct{}{"main": {}, "infera": {}}

// injectWorkerRolloutDefaults adds graceful rolling-upgrade knobs to a worker
// pod that the template did not already set: a preStop drain delay on the
// primary container and a termination grace long enough to drain in-flight
// generations, plus a /health readiness probe for single-node workers (skipped
// for multi-node LWS groups whose follower ranks > 0 do not serve /health).
// Existing values are preserved; the grace is only raised, never lowered.
func injectWorkerRolloutDefaults(spec *corev1.PodSpec, idx int, port int32, addReadiness bool) {
	if idx < 0 || idx >= len(spec.Containers) {
		return
	}
	c := &spec.Containers[idx]
	if addReadiness && c.ReadinessProbe == nil {
		// SGLang's /health runs a tiny prefill self-check that often takes
		// >1s, so a 1s probe timeout (the k8s default) flaps the pod between
		// Ready/NotReady. Use a generous timeout + higher failure threshold so
		// a healthy-but-busy engine is not marked NotReady.
		c.ReadinessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromInt32(port)},
			},
			InitialDelaySeconds: 15,
			PeriodSeconds:       15,
			TimeoutSeconds:      10,
			FailureThreshold:    6,
		}
	}
	if c.Lifecycle == nil {
		c.Lifecycle = &corev1.Lifecycle{}
	}
	if c.Lifecycle.PreStop == nil {
		c.Lifecycle.PreStop = &corev1.LifecycleHandler{
			Exec: &corev1.ExecAction{
				Command: []string{"/bin/sh", "-c", fmt.Sprintf("sleep %d", workerPreStopDrainSeconds)},
			},
		}
	}
	if spec.TerminationGracePeriodSeconds == nil || *spec.TerminationGracePeriodSeconds < workerTerminationGraceSeconds {
		grace := workerTerminationGraceSeconds
		spec.TerminationGracePeriodSeconds = &grace
	}
}

// podTemplateFromExtra passes a caller-supplied PodSpec through verbatim,
// merging in the service selector labels and ensuring the primary container
// exposes the service port (so buildServerService has a target). Used when
// ServiceSpec.ExtraPodSpec is set (an external orchestrator renders the full
// pod template).
func podTemplateFromExtra(idep *inferav1alpha1.InferaDeployment, svcName string, svc inferav1alpha1.ServiceSpec) corev1.PodTemplateSpec {
	spec := *svc.ExtraPodSpec.DeepCopy()
	port := servicePort(svc)
	// Locate the primary container (named main/infera, else the first one)
	// and guarantee it advertises the service port for Service targeting.
	idx := 0
	for i := range spec.Containers {
		if _, ok := mainContainerNames[spec.Containers[i].Name]; ok {
			idx = i
			break
		}
	}
	if len(spec.Containers) > 0 {
		hasPort := false
		for _, p := range spec.Containers[idx].Ports {
			if p.ContainerPort == port {
				hasPort = true
				break
			}
		}
		if !hasPort {
			spec.Containers[idx].Ports = append(spec.Containers[idx].Ports,
				corev1.ContainerPort{ContainerPort: port})
		}
		// k8s discovery: the server reads its watch scope from an env var so we
		// don't have to rewrite the externally-supplied entrypoint command.
		if useK8sDiscovery(idep) && svc.ComponentType == inferav1alpha1.ComponentTypeServer {
			spec.Containers[idx].Env = append(spec.Containers[idx].Env, corev1.EnvVar{
				Name: "INFERA_K8S_LABEL_SELECTOR", Value: discoveryLabelSelector(idep.Name)})
		}
	}
	// Bind the discovery ServiceAccount (workers patch their own Pod; the
	// server lists/watches Pods) unless the pod template already set one.
	if useK8sDiscovery(idep) && spec.ServiceAccountName == "" {
		spec.ServiceAccountName = discoverySAName(idep.Name)
	}
	// Graceful rolling-upgrade defaults for worker pods rendered by an external
	// template: inject readiness/preStop/grace the template omitted.
	if svc.ComponentType == inferav1alpha1.ComponentTypeWorker {
		injectWorkerRolloutDefaults(&spec, idx, port, svc.NumberOfNodes <= 1 && !svc.SkipReadinessProbe)
	}
	return corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Labels: podLabelsFor(idep.Name, svcName, svc)},
		Spec:       spec,
	}
}

func podTemplate(idep *inferav1alpha1.InferaDeployment, svcName string, svc inferav1alpha1.ServiceSpec) corev1.PodTemplateSpec {
	if svc.ExtraPodSpec != nil {
		tmpl := podTemplateFromExtra(idep, svcName, svc)
		applyGAIEFrontendSidecar(idep, svc, &tmpl)
		return tmpl
	}
	port := servicePort(svc)
	volumes := []corev1.Volume{}
	mounts := []corev1.VolumeMount{}
	if svc.Resources != nil && svc.Resources.SharedMemory != "" {
		sz := resource.MustParse(svc.Resources.SharedMemory)
		volumes = append(volumes, corev1.Volume{
			Name: "dshm",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{Medium: corev1.StorageMediumMemory, SizeLimit: &sz},
			},
		})
		mounts = append(mounts, corev1.VolumeMount{Name: "dshm", MountPath: "/dev/shm"})
	}
	// Host kernel config so ais-check reads "Kernel P2PDMA support" correctly.
	// Engine images ship no /boot/config-*, which false-negatives GPU-direct
	// (kvd then silently CPU-bounces L3 loads). hostPath auto-matches the node
	// kernel since the pod and node share it.
	bootHostPathType := corev1.HostPathDirectory
	volumes = append(volumes, corev1.Volume{
		Name: "boot-config",
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{Path: "/boot", Type: &bootHostPathType},
		},
	})
	mounts = append(mounts, corev1.VolumeMount{Name: "boot-config", MountPath: "/boot", ReadOnly: true})
	c := corev1.Container{
		Name:         "infera",
		Image:        imageFor(idep, svc),
		Command:      containerCommand(idep, svc),
		Env:          envFor(idep, svc),
		Resources:    resourceRequirements(svc),
		VolumeMounts: mounts,
		Ports:        []corev1.ContainerPort{{ContainerPort: port}},
	}
	podSpec := corev1.PodSpec{
		Containers: []corev1.Container{c},
		Volumes:    volumes,
	}
	if useK8sDiscovery(idep) {
		podSpec.ServiceAccountName = discoverySAName(idep.Name)
	}
	// Graceful rolling-upgrade defaults for GPU workers (readiness/preStop/grace);
	// readiness is skipped for multi-node LWS groups (follower ranks have no
	// /health). The server (CPU-only) keeps the default fast shutdown.
	if svc.ComponentType == inferav1alpha1.ComponentTypeWorker {
		injectWorkerRolloutDefaults(&podSpec, 0, port, svc.NumberOfNodes <= 1 && !svc.SkipReadinessProbe)
	}
	tmpl := corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Labels: podLabelsFor(idep.Name, svcName, svc)},
		Spec:       podSpec,
	}
	applyGAIEFrontendSidecar(idep, svc, &tmpl)
	return tmpl
}

func buildDeployment(idep *inferav1alpha1.InferaDeployment, svcName string, svc inferav1alpha1.ServiceSpec) *appsv1.Deployment {
	reps := replicasOf(svc)
	lbls := labelsFor(idep.Name, svcName)
	// Worker services use surge-free RollingUpdate (maxSurge=0, maxUnavailable=1):
	// the default RollingUpdate brings up a surge pod first, which on a
	// GPU-saturated cluster has no free GPU until the old pod is torn down — so
	// an image change deadlocks (new pod Pending, old never removed). maxSurge=0
	// tears an old pod down first (freeing its GPU) before creating the new one,
	// so it never deadlocks; and unlike Recreate it rolls one pod at a time, so a
	// multi-replica worker keeps serving (reduced capacity) instead of a full
	// outage. (A single-replica worker still has an unavoidable gap with no spare
	// GPU.) Gate on componentType==worker (not the flat Resources.GPU, which is
	// empty when GPUs are declared inside extraPodSpec); the
	// server (CPU-only) keeps the default RollingUpdate for zero-downtime surge.
	strategy := appsv1.DeploymentStrategy{}
	if svc.ComponentType == inferav1alpha1.ComponentTypeWorker {
		maxSurge := intstr.FromInt32(0)
		maxUnavailable := intstr.FromInt32(1)
		strategy = appsv1.DeploymentStrategy{
			Type: appsv1.RollingUpdateDeploymentStrategyType,
			RollingUpdate: &appsv1.RollingUpdateDeployment{
				MaxSurge:       &maxSurge,
				MaxUnavailable: &maxUnavailable,
			},
		}
	}
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: idep.Name + "-" + svcName, Namespace: idep.Namespace, Labels: lbls},
		Spec: appsv1.DeploymentSpec{
			Replicas: &reps,
			Selector: &metav1.LabelSelector{MatchLabels: lbls},
			Strategy: strategy,
			Template: podTemplate(idep, svcName, svc),
		},
	}
}

// buildLeaderWorkerSet returns an unstructured LeaderWorkerSet so the operator
// does not take a compile-time dependency on the LWS Go module (keeps Infera
// self-contained; the LWS CRD must be installed in the cluster).
func buildLeaderWorkerSet(idep *inferav1alpha1.InferaDeployment, svcName string, svc inferav1alpha1.ServiceSpec) *unstructured.Unstructured {
	reps := replicasOf(svc)
	lbls := labelsFor(idep.Name, svcName)
	tmpl := podTemplate(idep, svcName, svc)
	// Convert the typed PodTemplateSpec to a map for embedding.
	podMap, _ := toUnstructured(&tmpl)
	u := &unstructured.Unstructured{}
	u.SetAPIVersion(lwsAPIVersion)
	u.SetKind(lwsKind)
	u.SetName(idep.Name + "-" + svcName)
	u.SetNamespace(idep.Namespace)
	u.SetLabels(lbls)
	_ = unstructured.SetNestedField(u.Object, int64(reps), "spec", "replicas")
	_ = unstructured.SetNestedField(u.Object, int64(svc.NumberOfNodes), "spec", "leaderWorkerTemplate", "size")
	// workerTemplate IS a core/v1 PodTemplateSpec ({metadata, spec}); podMap is
	// exactly that, so it goes directly at workerTemplate (NOT workerTemplate.spec,
	// which would nest metadata/spec one level too deep and leave
	// workerTemplate.spec.containers null). podMap already carries metadata.labels.
	_ = unstructured.SetNestedMap(u.Object, podMap, "spec", "leaderWorkerTemplate", "workerTemplate")
	return u
}

// buildDiscoverySA / Role / RoleBinding provision the least-privilege identity
// for Kubernetes-native discovery: workers patch their own Pod annotation and
// the server lists/watches this deployment's worker Pods. Namespaced Role so
// the grant is scoped to the workload namespace.
func buildDiscoverySA(idep *inferav1alpha1.InferaDeployment) *corev1.ServiceAccount {
	return &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      discoverySAName(idep.Name),
			Namespace: idep.Namespace,
			Labels:    labelsFor(idep.Name, "disc"),
		},
	}
}

func buildDiscoveryRole(idep *inferav1alpha1.InferaDeployment) *rbacv1.Role {
	return &rbacv1.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      discoverySAName(idep.Name),
			Namespace: idep.Namespace,
			Labels:    labelsFor(idep.Name, "disc"),
		},
		Rules: []rbacv1.PolicyRule{{
			APIGroups: []string{""},
			Resources: []string{"pods"},
			Verbs:     []string{"get", "list", "watch", "patch"},
		}},
	}
}

func buildDiscoveryRoleBinding(idep *inferav1alpha1.InferaDeployment) *rbacv1.RoleBinding {
	return &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      discoverySAName(idep.Name),
			Namespace: idep.Namespace,
			Labels:    labelsFor(idep.Name, "disc"),
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "Role",
			Name:     discoverySAName(idep.Name),
		},
		Subjects: []rbacv1.Subject{{
			Kind:      "ServiceAccount",
			Name:      discoverySAName(idep.Name),
			Namespace: idep.Namespace,
		}},
	}
}

func buildServerService(idep *inferav1alpha1.InferaDeployment, svcName string, svc inferav1alpha1.ServiceSpec) *corev1.Service {
	port := servicePort(svc)
	lbls := labelsFor(idep.Name, svcName)
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: idep.Name + "-" + svcName, Namespace: idep.Namespace, Labels: lbls},
		Spec: corev1.ServiceSpec{
			Selector: lbls,
			Ports:    []corev1.ServicePort{{Name: "http", Port: port, TargetPort: intstr.FromInt32(port)}},
		},
	}
}
