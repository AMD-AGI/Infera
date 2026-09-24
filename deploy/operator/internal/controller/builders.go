/*
 * Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

package controller

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
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
	workerPreStopDrainSeconds        = 15 // preStop sleep: let the router drop us before SIGTERM
	workerDefaultDrainTimeoutSeconds = 30 // matches the worker's --drain-timeout default
	// Teardown after the drain finishes: stopping the KV plane and
	// engine.stop(), which SIGTERMs the engine's process group and waits up
	// to 30s before escalating to SIGKILL.
	workerTeardownHeadroomSeconds = 50
	// Floor, so short drain timeouts still leave room for a slow engine exit.
	workerTerminationGraceSeconds int64 = 120

	// The worker reads this as the default for --drain-timeout, so it sets the
	// drain just as effectively as the flag does.
	drainTimeoutEnvVar = "INFERA_DRAIN_TIMEOUT"

	// Readiness is probed on the worker's own port rather than the engine's
	// /health. The engine answers as soon as its weights are loaded, which is
	// before the PD barrier has moved a real KV block and before the worker
	// has registered -- so a surge rollout keyed on /health retires the pod
	// that is serving in favour of one the router cannot reach yet. The worker
	// opens this port after registering and closes it when shutdown begins.
	workerReadinessPort int32 = 30090
	readinessPortEnvVar       = "INFERA_READINESS_PORT"
	// readinessProbePath is the path of the injected probe; a probe on the
	// readiness port with any other path is not the registration signal.
	readinessProbePath = "/ready"
)

// drainSeconds parses a worker --drain-timeout value. The worker takes a
// float; round up so a fractional value never shortens the budget.
// Ceiling on a parsed drain timeout, and therefore on the grace period derived
// from it. An hour is far past any real generation; beyond that the value is
// more likely a typo than an intention, and it is written into
// terminationGracePeriodSeconds, where too large means a stuck Pod that only
// `--force` can delete.
const maxDrainTimeoutSeconds = 3600

func drainSeconds(v string) (int, bool) {
	f, err := strconv.ParseFloat(v, 64)
	// NaN fails every comparison, so `f <= 0` does not catch it, and neither it
	// nor an infinity survives the conversion below: Go leaves out-of-range
	// float-to-int implementation-defined, and on amd64 both land on minInt64,
	// which the floor then quietly turns back into the default budget. Refusing
	// them means the fallback is at least a deliberate one.
	if err != nil || math.IsNaN(f) || math.IsInf(f, 0) || f <= 0 {
		return 0, false
	}
	if f > maxDrainTimeoutSeconds {
		return maxDrainTimeoutSeconds, true
	}
	return int(math.Ceil(f)), true
}

// graceSecondsFor sizes terminationGracePeriodSeconds so the kubelet cannot
// SIGKILL a worker in the middle of shutting down.
//
// The budget is preStop + the worker's --drain-timeout + teardown. That last
// term is not small: engine.stop() alone waits up to 30s for the engine's
// process group before escalating. Leaving the grace at a fixed 120s was fine
// for the default 30s drain, but --drain-timeout lives in free-form args that
// nothing here parsed -- so raising it for long generations (the exact reason
// anyone raises it) silently pushed shutdown past the grace and turned a
// graceful drain back into a kill.
//
// The drain can be set two ways and both have to be read. The worker takes
// $INFERA_DRAIN_TIMEOUT as the flag's *default*, so an env var raises the drain
// exactly as effectively as the flag does -- and parsing only the flag left the
// same silent overrun through a different door.
//
// Sources are given in increasing priority, and precedence is resolved per
// source rather than globally. It has to be: on the extraPodSpec path the
// template is passed through verbatim, so a --drain-timeout in ServiceSpec.Args
// is never rendered into the container and does not affect the drain at all.
// Letting that inert flag outrank the variable the container really reads sizes
// the budget for a drain that never happens, while the real one runs long and
// is killed partway through -- the exact failure this function exists to stop.
func graceSecondsFor(sources ...drainSource) int64 {
	drain := workerDefaultDrainTimeoutSeconds
	for _, s := range sources {
		if d, ok := s.drainTimeout(); ok {
			drain = d
		}
	}
	need := int64(workerPreStopDrainSeconds + drain + workerTeardownHeadroomSeconds)
	if need < workerTerminationGraceSeconds {
		return workerTerminationGraceSeconds
	}
	return need
}

// drainSource is one place a drain timeout can be configured: a set of args and
// env vars that travel together, either both from ServiceSpec or both from the
// container itself.
type drainSource struct {
	args []string
	env  []corev1.EnvVar
}

// drainTimeout resolves this source alone, reporting whether it set anything.
// Env first so an explicit flag overrides it, matching argparse: the variable
// supplies the default, the flag replaces it.
func (s drainSource) drainTimeout() (int, bool) {
	out, found := 0, false
	for _, e := range s.env {
		if e.Name != drainTimeoutEnvVar {
			continue
		}
		// A valueFrom reference is resolved by the kubelet, not here, so its
		// value is unknowable at build time and the budget falls back to the
		// flag or the default. Worth knowing if a drain is ever cut short
		// despite a ConfigMap saying otherwise.
		if d, ok := drainSeconds(e.Value); ok {
			out, found = d, true
		}
	}
	for i, a := range s.args {
		v := ""
		if a == "--drain-timeout" && i+1 < len(s.args) {
			v = s.args[i+1]
		} else if strings.HasPrefix(a, "--drain-timeout=") {
			v = strings.TrimPrefix(a, "--drain-timeout=")
		}
		if v == "" {
			continue
		}
		if d, ok := drainSeconds(v); ok {
			out, found = d, true
		}
	}
	return out, found
}

// Identity labels on every workload this operator builds. They are the only
// link back from a Deployment/LeaderWorkerSet to the CR and service that
// produced it, so the watch handlers that map a workload event to the objects
// interested in it read these rather than re-deriving the name.
const (
	labelKeyDeployment = "infera.amd.com/deployment"
	labelKeyService    = "infera.amd.com/service"
)

// labelsFor returns the selector/identity labels for a service's workload.
func labelsFor(idepName, svcName string) map[string]string {
	return map[string]string{
		"app.kubernetes.io/managed-by": "infera-operator",
		labelKeyDeployment:             idepName,
		labelKeyService:                svcName,
	}
}

// lwsInstalled reports whether the LeaderWorkerSet CRD is served by the API.
//
// It gates registering a watch on LWS: controller-runtime builds an informer
// for every watched type at startup, and one for a kind the API server does not
// serve fails the manager outright. LWS is an optional dependency here -- only
// multi-node services use it -- so a single-node cluster without the CRD must
// still be able to run the operator.
//
// The check runs once, at setup. Installing the CRD afterwards therefore needs
// an operator restart to pick up the watch; until then multi-node status still
// refreshes on the reconciler's periodic resync, just not immediately.
func lwsInstalled(mapper meta.RESTMapper) bool {
	_, err := mapper.RESTMapping(
		schema.GroupKind{Group: lwsGVK().Group, Kind: lwsGVK().Kind}, lwsGVK().Version)
	return err == nil
}

// lwsObject returns an empty LeaderWorkerSet for use as a watch target.
func lwsObject() *unstructured.Unstructured {
	u := &unstructured.Unstructured{}
	u.SetGroupVersionKind(lwsGVK())
	return u
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
		// POD_IP is what a worker advertises: it binds 0.0.0.0, and the address
		// it registers is the one the router dials, so without this it
		// registers its bind host and every request to it is unreachable.
		env = append(env,
			corev1.EnvVar{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
			corev1.EnvVar{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"}}},
			corev1.EnvVar{Name: "POD_IP", ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.podIP"}}},
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

// hasEnv reports whether the container already declares the named variable.
func hasEnv(c *corev1.Container, name string) bool {
	for _, e := range c.Env {
		if e.Name == name {
			return true
		}
	}
	return false
}

// readinessPortFrom returns the port the worker will open, and whether the
// operator can know it.
//
// A valueFrom source is resolved by the kubelet, not here, so the value is
// unknowable at render time. Guessing the default would pin the probe to a
// port the worker may not bind, and with maxUnavailable=0 that is an
// unrecoverable stall -- so the caller skips the probe instead, which falls
// back to surge-free rolling.
//
// Stricter than the worker's int(), which also accepts surrounding whitespace
// and digit underscores: such a value is treated as unknowable, so the probe
// is skipped and the worker rolls surge-free rather than being probed on a
// port read differently from the worker's. Read from the container rather than
// ServiceSpec.Env because an extraPodSpec template is passed through verbatim
// and is the more specific source.
func readinessPortFrom(c *corev1.Container) (int32, bool) {
	for _, e := range c.Env {
		if e.Name != readinessPortEnvVar {
			continue
		}
		if e.ValueFrom != nil {
			return 0, false
		}
		if e.Value != strings.TrimSpace(e.Value) || strings.Contains(e.Value, "_") {
			return 0, false
		}
		n, err := strconv.Atoi(e.Value)
		if err != nil || n <= 0 || n >= 65536 {
			return 0, false
		}
		return int32(n), true //nolint:gosec // bounded above
	}
	return workerReadinessPort, true
}

// injectWorkerRolloutDefaults adds graceful rolling-upgrade knobs to a worker
// pod that the template did not already set: a preStop drain delay on the
// primary container and a termination grace long enough to drain in-flight
// generations, plus a readiness probe on the worker's registration port for
// single-node workers (skipped for multi-node LWS groups, whose follower ranks
// do not serve). Existing values are preserved; the grace is only raised,
// never lowered.
func injectWorkerRolloutDefaults(
	spec *corev1.PodSpec, idx int, addReadiness bool,
	args []string, env []corev1.EnvVar,
) {
	if idx < 0 || idx >= len(spec.Containers) {
		return
	}
	c := &spec.Containers[idx]
	// The drain can arrive two ways: via ServiceSpec.Args/Env on the rendered
	// path, or written straight into the container by an extraPodSpec template,
	// which is passed through verbatim. Reading only the first would miss
	// exactly the deployments most likely to have tuned it. They stay separate
	// sources, listed in increasing priority, because the container's is what
	// the process actually reads -- see graceSecondsFor.
	fromService := drainSource{args: args, env: env}
	fromContainer := drainSource{
		args: append(append([]string{}, c.Command...), c.Args...),
		env:  c.Env,
	}
	// The worker resolves its readiness port from the environment, so the
	// entrypoint -- which operators write by hand -- does not have to change.
	// A value already on the container wins, and the probe below follows it,
	// so the two cannot disagree.
	readyPort, portKnown := readinessPortFrom(c)
	if !hasEnv(c, readinessPortEnvVar) {
		c.Env = append(c.Env, corev1.EnvVar{
			Name:  readinessPortEnvVar,
			Value: strconv.Itoa(int(readyPort)),
		})
	}
	if addReadiness && portKnown && c.ReadinessProbe == nil {
		// Probed on the readiness port, not the engine's /health: only the
		// former means "registered, and the router can reach me". The port
		// simply is not open before then, so the probe fails closed, which is
		// what holds a surge rollout back until the replacement can serve.
		c.ReadinessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{
					Path: readinessProbePath, Port: intstr.FromInt32(readyPort),
				},
			},
			// Weight loading and the PD barrier both happen before the port
			// opens, and either can take minutes, so the probe has to tolerate
			// a long stretch of refused connections without ever being fatal.
			// It is a readiness probe only -- nothing restarts the pod.
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
	if want := graceSecondsFor(fromService, fromContainer); spec.TerminationGracePeriodSeconds == nil ||
		*spec.TerminationGracePeriodSeconds < want {
		grace := want
		spec.TerminationGracePeriodSeconds = &grace
	}
}

// podTemplateFromExtra passes a caller-supplied PodSpec through verbatim,
// merging in the service selector labels and ensuring the primary container
// exposes the service port (so buildServerService has a target). Used when
// ServiceSpec.ExtraPodSpec is set (an external orchestrator renders the full
// pod template).
// appendEnvIfAbsent adds each variable the container does not already declare.
//
// A template supplied by an external orchestrator may well set these itself,
// and a duplicate name in a container's env is not an error -- the last one
// wins, silently overriding what the author wrote.
func appendEnvIfAbsent(env []corev1.EnvVar, add ...corev1.EnvVar) []corev1.EnvVar {
	present := make(map[string]bool, len(env))
	for _, e := range env {
		present[e.Name] = true
	}
	for _, e := range add {
		if !present[e.Name] {
			env = append(env, e)
		}
	}
	return env
}

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
		if useK8sDiscovery(idep) {
			// Pod identity, for both component types: a worker registers by
			// patching its own Pod annotation, and the server finds the
			// deployment it belongs to from its own Pod labels. Rendering the
			// template elsewhere does not change that either needs to know
			// which Pod it is.
			spec.Containers[idx].Env = appendEnvIfAbsent(spec.Containers[idx].Env,
				corev1.EnvVar{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
				corev1.EnvVar{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.namespace"}}},
				corev1.EnvVar{Name: "POD_IP", ValueFrom: &corev1.EnvVarSource{
					FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.podIP"}}},
			)
			// The server reads its watch scope from an env var so we don't have
			// to rewrite the externally-supplied entrypoint command.
			if svc.ComponentType == inferav1alpha1.ComponentTypeServer {
				spec.Containers[idx].Env = appendEnvIfAbsent(spec.Containers[idx].Env,
					corev1.EnvVar{
						Name:  "INFERA_K8S_LABEL_SELECTOR",
						Value: discoveryLabelSelector(idep.Name),
					})
			}
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
		injectWorkerRolloutDefaults(&spec, idx, svc.NumberOfNodes <= 1 && !svc.SkipReadinessProbe, svc.Args, svc.Env)
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
		injectWorkerRolloutDefaults(&podSpec, 0, svc.NumberOfNodes <= 1 && !svc.SkipReadinessProbe, svc.Args, svc.Env)
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
	// A worker's rolling strategy follows whether it has a readiness probe,
	// because that is the only signal telling the rollout the replacement can
	// serve.
	//
	// With a probe: maxSurge=1/maxUnavailable=0. The replacement is created
	// first and the old pod is retired only once the new one is Ready — which,
	// probed on the readiness port, means it has loaded its weights, passed
	// the PD barrier and registered. A single-replica worker keeps serving
	// across an image or template change, which surge-free rolling cannot do.
	// The cost is a spare GPU per rolling pod: without one the replacement
	// stays Pending and the roll does not finish, a stall with the old pod
	// still serving rather than an outage.
	//
	// Without one (skipReadinessProbe): maxSurge=0/maxUnavailable=1, the
	// historical behaviour. Surging here would be worse than not surging —
	// every pod counts as Ready the moment it is Running, so the rollout would
	// retire the pod that is still serving in favour of one still loading
	// weights. Rolling old-pod-first has a gap, but it is bounded and the
	// rollout always completes, which also keeps whole-node pinned workers
	// (replicas=1, nodeSelector, all GPUs on the host) upgradeable: no surge
	// pod could ever schedule for them.
	//
	// Gate on componentType==worker (not the flat Resources.GPU, which is empty
	// when GPUs are declared inside extraPodSpec); the server (CPU-only) keeps
	// the apps/v1 default, which already surges.
	tmpl := podTemplate(idep, svcName, svc)
	strategy := appsv1.DeploymentStrategy{}
	if svc.ComponentType == inferav1alpha1.ComponentTypeWorker {
		surge, unavailable := int32(1), int32(0)
		if !probedOnReadinessPort(&tmpl) {
			surge, unavailable = 0, 1
		}
		maxSurge := intstr.FromInt32(surge)
		maxUnavailable := intstr.FromInt32(unavailable)
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
			Template: tmpl,
		},
	}
}

// probedOnReadinessPort reports whether a container is probed on the port the
// worker opens once it has registered.
//
// Read off the rendered template rather than re-deriving the conditions, so
// every reason a pod ends up without that probe lands here: skipReadinessProbe,
// or an extraPodSpec that supplies a probe of its own. The latter is treated
// as "no readiness signal" on purpose -- a hand-written /health probe answers
// while the engine is still starting, which is exactly what makes surging
// unsafe. Only the port this operator injects carries the guarantee that Ready
// means registered.
func probedOnReadinessPort(tmpl *corev1.PodTemplateSpec) bool {
	for i := range tmpl.Spec.Containers {
		c := &tmpl.Spec.Containers[i]
		p := c.ReadinessProbe
		if p == nil || p.HTTPGet == nil || p.HTTPGet.Port.Type != intstr.Int ||
			p.HTTPGet.Path != readinessProbePath {
			continue
		}
		if port, ok := readinessPortFrom(c); ok && p.HTTPGet.Port.IntVal == port {
			return true
		}
	}
	return false
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
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{""},
				Resources: []string{"pods"},
				Verbs:     []string{"get", "list", "watch", "patch"},
			},
			{
				// The server's scaling API writes replica counts back to the CR
				// it belongs to. Named via ResourceNames so the grant reaches
				// exactly this deployment: every Pod here shares one identity,
				// so an unrestricted grant would also let any worker resize the
				// fleet, and a worker has no business doing that.
				//
				// Present whether or not --enable-scaling-api is set. The flag
				// lives on the server's command line and the operator does not
				// parse it; a permission nothing exercises costs nothing, while
				// discovering it is absent only after enabling the feature
				// costs a redeploy.
				APIGroups:     []string{inferav1alpha1.GroupVersion.Group},
				Resources:     []string{"inferadeployments"},
				ResourceNames: []string{idep.Name},
				Verbs:         []string{"get", "patch"},
			},
		},
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
