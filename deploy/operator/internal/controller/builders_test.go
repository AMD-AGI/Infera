/*
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/

package controller

import (
	"strconv"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

// The grace period is the only thing standing between a graceful drain and a
// SIGKILL halfway through one. It has to cover preStop, the worker's own
// --drain-timeout, and the teardown that follows -- of which engine.stop()
// alone can take 30s waiting on the engine's process group.
//
// The failure this guards against is quiet: raising --drain-timeout is exactly
// what an operator does when generations are long, and until the grace was
// derived from it that made shutdown *less* graceful, not more.
func TestGraceSecondsFor(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want int64
	}{
		{"no args uses the floor", nil, 120},
		{"default drain stays at the floor", []string{"--drain-timeout", "30"}, 120},
		{
			"a long drain raises the grace above the floor",
			[]string{"--model-path", "/m", "--drain-timeout", "120"},
			185, // 15 preStop + 120 drain + 50 teardown
		},
		{"equals form is parsed too", []string{"--drain-timeout=120"}, 185},
		{
			"fractional values round up rather than shortening the budget",
			[]string{"--drain-timeout", "60.5"},
			126, // 15 + 61 + 50
		},
		{"a short drain does not lower the floor", []string{"--drain-timeout", "1"}, 120},
		{"garbage falls back to the default", []string{"--drain-timeout", "abc"}, 120},
		{"a trailing flag with no value is ignored", []string{"--drain-timeout"}, 120},
		{"non-positive is ignored", []string{"--drain-timeout", "0"}, 120},
		{"the last occurrence wins", []string{"--drain-timeout", "5", "--drain-timeout", "200"}, 265},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := graceSecondsFor(drainSource{args: c.args}); got != c.want {
				t.Fatalf("graceSecondsFor(%v) = %d, want %d", c.args, got, c.want)
			}
		})
	}
}

// The budget must actually hold, not merely be larger than the old constant.
func TestGraceCoversTheWholeShutdown(t *testing.T) {
	for _, drain := range []int{30, 60, 120, 300} {
		args := []string{"--drain-timeout", itoa(drain)}
		grace := graceSecondsFor(drainSource{args: args})
		need := int64(workerPreStopDrainSeconds + drain + workerTeardownHeadroomSeconds)
		if grace < need {
			t.Fatalf("drain=%d: grace %d < required %d -- kubelet would SIGKILL mid-drain",
				drain, grace, need)
		}
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

// extraPodSpec templates are passed through verbatim, so --drain-timeout may
// live on the container rather than in ServiceSpec.Args. Reading only the
// latter would miss precisely the deployments that tuned it.
func TestGraceReadsDrainTimeoutFromTheContainerToo(t *testing.T) {
	spec := &corev1.PodSpec{Containers: []corev1.Container{{
		Name:    "main",
		Command: []string{"python3", "-m", "infera.engine.sglang"},
		Args:    []string{"--model-path", "/m", "--drain-timeout", "240"},
	}}}
	injectWorkerRolloutDefaults(spec, 0, false, nil, nil)
	if spec.TerminationGracePeriodSeconds == nil {
		t.Fatal("grace not set")
	}
	want := int64(workerPreStopDrainSeconds + 240 + workerTeardownHeadroomSeconds)
	if *spec.TerminationGracePeriodSeconds != want {
		t.Fatalf("grace = %d, want %d", *spec.TerminationGracePeriodSeconds, want)
	}
}

// The worker takes $INFERA_DRAIN_TIMEOUT as the default for --drain-timeout, so
// setting it raises the drain exactly as the flag does. Sizing the grace from
// the flag alone left the same silent overrun through a different door: the
// worker would drain for its full timeout and be SIGKILLed partway through.
func TestGraceReadsDrainTimeoutFromTheEnvironment(t *testing.T) {
	env := []corev1.EnvVar{
		{Name: "HF_HOME", Value: "/models"},
		{Name: drainTimeoutEnvVar, Value: "300"},
	}
	want := int64(workerPreStopDrainSeconds + 300 + workerTeardownHeadroomSeconds)
	if got := graceSecondsFor(drainSource{env: env}); got != want {
		t.Fatalf("env-set drain: grace = %d, want %d", got, want)
	}
}

// argparse reads the variable as the flag's default, so an explicit flag wins.
// Sizing the budget off the larger of the two would be safe but wrong, and
// wrong here means a pod that lingers minutes longer than its config says.
func TestGraceFlagOverridesTheEnvironment(t *testing.T) {
	env := []corev1.EnvVar{{Name: drainTimeoutEnvVar, Value: "300"}}
	args := []string{"--drain-timeout", "60"}
	want := int64(workerPreStopDrainSeconds + 60 + workerTeardownHeadroomSeconds)
	if got := graceSecondsFor(drainSource{args: args, env: env}); got != want {
		t.Fatalf("flag with env set: grace = %d, want the flag's %d", got, want)
	}
}

func TestGraceIgnoresUnreadableEnvValues(t *testing.T) {
	// valueFrom resolves in the kubelet; nothing is readable here, so the
	// budget has to fall back rather than treat the empty value as zero.
	from := []corev1.EnvVar{{
		Name: drainTimeoutEnvVar,
		ValueFrom: &corev1.EnvVarSource{
			ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Key: "drain"},
		},
	}}
	if got := graceSecondsFor(drainSource{env: from}); got != workerTerminationGraceSeconds {
		t.Fatalf("valueFrom: grace = %d, want the floor %d", got, workerTerminationGraceSeconds)
	}
	for _, v := range []string{"", "abc", "0", "-5"} {
		env := []corev1.EnvVar{{Name: drainTimeoutEnvVar, Value: v}}
		if got := graceSecondsFor(drainSource{env: env}); got != workerTerminationGraceSeconds {
			t.Fatalf("env %q: grace = %d, want the floor %d", v, got, workerTerminationGraceSeconds)
		}
	}
}

// An extraPodSpec template is passed through verbatim, so the variable may sit
// on the container rather than in ServiceSpec.Env -- the same asymmetry the
// flag has, and the deployments most likely to have tuned the drain.
func TestGraceReadsDrainEnvFromTheContainerToo(t *testing.T) {
	spec := &corev1.PodSpec{Containers: []corev1.Container{{
		Name: "main",
		Env:  []corev1.EnvVar{{Name: drainTimeoutEnvVar, Value: "240"}},
	}}}
	injectWorkerRolloutDefaults(spec, 0, false, nil, nil)
	if spec.TerminationGracePeriodSeconds == nil {
		t.Fatal("grace not set")
	}
	want := int64(workerPreStopDrainSeconds + 240 + workerTeardownHeadroomSeconds)
	if *spec.TerminationGracePeriodSeconds != want {
		t.Fatalf("grace = %d, want %d", *spec.TerminationGracePeriodSeconds, want)
	}
}

// On the extraPodSpec path the template is passed through verbatim, so
// ServiceSpec.Args is never rendered into the container -- a --drain-timeout
// sitting there is inert. It must not outrank the variable the container will
// actually read, or the budget is sized for a drain that never happens while
// the real one runs long and gets SIGKILLed partway through. Precedence is by
// source: what the process sees wins, and only within a source does a flag
// beat a variable.
func TestAnInertServiceSpecFlagDoesNotOutrankTheContainer(t *testing.T) {
	spec := &corev1.PodSpec{Containers: []corev1.Container{{
		Name: "main",
		Env:  []corev1.EnvVar{{Name: drainTimeoutEnvVar, Value: "600"}},
	}}}
	injectWorkerRolloutDefaults(spec, 0, false, []string{"--drain-timeout", "30"}, nil)

	want := int64(workerPreStopDrainSeconds + 600 + workerTeardownHeadroomSeconds)
	if got := *spec.TerminationGracePeriodSeconds; got != want {
		t.Fatalf("grace = %d, want %d -- the container drains for 600s, so %d "+
			"leaves the kubelet killing it partway through", got, want, got)
	}
}

// The same precedence, the other way round: a flag the container really runs
// with beats a variable from ServiceSpec.
func TestTheContainerFlagBeatsAServiceSpecVariable(t *testing.T) {
	spec := &corev1.PodSpec{Containers: []corev1.Container{{
		Name: "main",
		Args: []string{"--drain-timeout=300"},
	}}}
	env := []corev1.EnvVar{{Name: drainTimeoutEnvVar, Value: "45"}}
	injectWorkerRolloutDefaults(spec, 0, false, nil, env)

	want := int64(workerPreStopDrainSeconds + 300 + workerTeardownHeadroomSeconds)
	if got := *spec.TerminationGracePeriodSeconds; got != want {
		t.Fatalf("grace = %d, want %d", got, want)
	}
}

// A drain timeout arrives as free-form text from args or an env var, so the
// parse has to survive whatever is there. Two directions matter.
//
// Below: Go leaves float-to-int conversion implementation-defined when the
// value does not fit, and on amd64 `inf` and `NaN` both land on minInt64. The
// floor then hides it, so a worker configured with an unusable value silently
// gets the default budget instead of anything signalling a mistake. Python's
// argparse accepts `inf` as a float, so this is reachable.
//
// Above: nothing bounded the result, so a typo like 86400 renders a Pod that
// takes a day to delete, and 9e18 overflows into a nonsensical grace period.
func TestDrainTimeoutRejectsValuesItCannotUse(t *testing.T) {
	for _, v := range []string{"inf", "+Inf", "-Inf", "NaN", "abc", "", "0", "-5"} {
		if got, ok := drainSeconds(v); ok {
			t.Errorf("drainSeconds(%q) = %d, accepted; an unusable value must be refused "+
				"so the budget falls back to the default", v, got)
		}
	}
}

func TestDrainTimeoutIsCappedAtSomethingSurvivable(t *testing.T) {
	// Finite but implausible: clamped rather than refused, since the intent is
	// legible even when the number is not. 9e18 also overflows an int, which is
	// what made an unbounded path dangerous rather than merely silly.
	for _, v := range []string{"86400", "1e30", "9e18"} {
		got, ok := drainSeconds(v)
		if !ok {
			t.Fatalf("drainSeconds(%q): a finite positive value should parse", v)
		}
		if got != maxDrainTimeoutSeconds {
			t.Errorf("drainSeconds(%q) = %d, want it clamped to %d: an unbounded grace "+
				"period leaves a stuck Pod deletable only with --force",
				v, got, maxDrainTimeoutSeconds)
		}
	}
}

func TestDrainTimeoutStillAcceptsOrdinaryValues(t *testing.T) {
	for _, c := range []struct {
		in   string
		want int
	}{{"30", 30}, {"0.5", 1}, {"120.4", 121}, {"300", 300}} {
		got, ok := drainSeconds(c.in)
		if !ok || got != c.want {
			t.Errorf("drainSeconds(%q) = %d,%v; want %d,true", c.in, got, ok, c.want)
		}
	}
}

// Pod identity is what k8s discovery is built on: a worker patches its own Pod
// annotation to register, and the server reads its own labels to find the
// deployment it belongs to. Both need POD_NAME, which the operator injects --
// on the path that renders the pod itself. A template supplied through
// extraPodSpec took a different path and got the watch selector but not the
// identity, so registration and the scaling API both failed on exactly the
// deployments the PD example tells people to write.
func TestExtraPodSpecStillGetsPodIdentity(t *testing.T) {
	for _, ct := range []inferav1alpha1.ComponentType{
		inferav1alpha1.ComponentTypeServer,
		inferav1alpha1.ComponentTypeWorker,
	} {
		idep := idepWith(1)
		idep.Spec.DiscoveryBackend = "kubernetes"
		svc := inferav1alpha1.ServiceSpec{
			ComponentType: ct,
			ExtraPodSpec: &corev1.PodSpec{
				Containers: []corev1.Container{{Name: "main", Image: "x"}},
			},
		}
		tmpl := podTemplateFromExtra(idep, "svc", svc)
		got := map[string]bool{}
		for _, e := range tmpl.Spec.Containers[0].Env {
			got[e.Name] = true
		}
		for _, want := range []string{"POD_NAME", "POD_NAMESPACE"} {
			if !got[want] {
				t.Errorf("%s: extraPodSpec container has no %s; "+
					"self-registration and the scaling API both need it", ct, want)
			}
		}
	}
}

// A template that sets these itself keeps its own values: a duplicate env name
// is not an error, the last one wins, and appending ours would silently
// override whatever the author had in mind.
func TestExtraPodSpecKeepsItsOwnPodIdentity(t *testing.T) {
	idep := idepWith(1)
	idep.Spec.DiscoveryBackend = "kubernetes"
	svc := inferav1alpha1.ServiceSpec{
		ComponentType: inferav1alpha1.ComponentTypeWorker,
		ExtraPodSpec: &corev1.PodSpec{Containers: []corev1.Container{{
			Name:  "main",
			Image: "x",
			Env:   []corev1.EnvVar{{Name: "POD_NAME", Value: "chosen-by-the-author"}},
		}}},
	}
	tmpl := podTemplateFromExtra(idep, "svc", svc)

	seen := 0
	for _, e := range tmpl.Spec.Containers[0].Env {
		if e.Name != "POD_NAME" {
			continue
		}
		seen++
		if e.Value != "chosen-by-the-author" {
			t.Errorf("POD_NAME = %q, want the template's own value", e.Value)
		}
	}
	if seen != 1 {
		t.Errorf("POD_NAME appears %d times, want 1", seen)
	}
}

// A worker binds 0.0.0.0 and advertises something else, because the address it
// registers is the one the router dials. Under k8s discovery it resolves that
// from POD_IP -- the logic is already there and reads the downward API -- so
// leaving the variable out makes the worker register 0.0.0.0 and every request
// to it fail with "worker unreachable".
//
// Measured before this was injected: the worker came up healthy, registered,
// and the router returned {"error":"worker 0.0.0.0:8080 unreachable"} for the
// first inference request.
func TestWorkersLearnTheirOwnAddress(t *testing.T) {
	idep := idepWith(1)
	idep.Spec.DiscoveryBackend = "kubernetes"

	check := func(t *testing.T, env []corev1.EnvVar, where string) {
		t.Helper()
		for _, e := range env {
			if e.Name != "POD_IP" {
				continue
			}
			if e.ValueFrom == nil || e.ValueFrom.FieldRef == nil ||
				e.ValueFrom.FieldRef.FieldPath != "status.podIP" {
				t.Errorf("%s: POD_IP is not read from the downward API", where)
			}
			return
		}
		t.Errorf("%s: no POD_IP; the worker would advertise its bind address", where)
	}

	svc := inferav1alpha1.ServiceSpec{ComponentType: inferav1alpha1.ComponentTypeWorker}
	check(t, envFor(idep, svc), "rendered pod")

	svc.ExtraPodSpec = &corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main", Image: "x"}},
	}
	tmpl := podTemplateFromExtra(idep, "worker", svc)
	check(t, tmpl.Spec.Containers[0].Env, "extraPodSpec")
}

// rollingOf returns the Deployment's rolling parameters, failing when the
// strategy is not RollingUpdate at all.
func rollingOf(t *testing.T, svc inferav1alpha1.ServiceSpec) (surge, unavailable int32) {
	t.Helper()
	dep := buildDeployment(idepWith(1), "w", svc)
	ru := dep.Spec.Strategy.RollingUpdate
	if dep.Spec.Strategy.Type != appsv1.RollingUpdateDeploymentStrategyType || ru == nil {
		t.Fatalf("strategy = %+v, want RollingUpdate with parameters", dep.Spec.Strategy)
	}
	return ru.MaxSurge.IntVal, ru.MaxUnavailable.IntVal
}

// A worker rolls surge-first so a single-replica prefill or decode keeps
// serving across an upgrade. maxUnavailable must be 0, not merely maxSurge 1
// -- with maxUnavailable=1 the controller is free to retire the only serving
// pod before the replacement is Ready, which is the outage this exists to
// remove.
func TestWorkersRollSurgeFirst(t *testing.T) {
	surge, unavailable := rollingOf(t, inferav1alpha1.ServiceSpec{
		ComponentType: inferav1alpha1.ComponentTypeWorker,
	})
	if surge != 1 {
		t.Errorf("maxSurge = %d, want 1: the replacement must start before the old pod goes", surge)
	}
	if unavailable != 0 {
		t.Fatalf("maxUnavailable = %d, want 0: any other value permits a gap with no pod serving",
			unavailable)
	}
}

// The server is CPU-only and the apps/v1 default already surges. Forcing a
// strategy here would only narrow it.
func TestTheServerKeepsTheDefaultStrategy(t *testing.T) {
	dep := buildDeployment(idepWith(1), "server", inferav1alpha1.ServiceSpec{
		ComponentType: inferav1alpha1.ComponentTypeServer,
	})
	if dep.Spec.Strategy.Type != "" || dep.Spec.Strategy.RollingUpdate != nil {
		t.Errorf("strategy = %+v, want the apps/v1 default", dep.Spec.Strategy)
	}
}

// Readiness decides when the old pod may go, so it has to mean "registered",
// not "sglang answered". The engine's /health is up well before the PD
// barrier has moved a KV block and before the worker registers; probing it
// would retire the serving pod in favour of one the router cannot reach.
func TestReadinessIsProbedOnTheRegistrationPort(t *testing.T) {
	spec := &corev1.PodSpec{Containers: []corev1.Container{{Name: "main"}}}
	injectWorkerRolloutDefaults(spec, 0, true, nil, nil)

	p := spec.Containers[0].ReadinessProbe
	if p == nil || p.HTTPGet == nil {
		t.Fatal("no readiness probe injected")
	}
	if got := p.HTTPGet.Port.IntVal; got != workerReadinessPort {
		t.Errorf("probe port = %d, want %d (the readiness port, not the engine)",
			got, workerReadinessPort)
	}
	if p.HTTPGet.Path == "/health" {
		t.Error("probing /health defeats the point: it answers before registration")
	}
	// The worker reads the port from the environment, so the two must agree
	// or the probe polls a port nothing listens on.
	var env string
	for _, e := range spec.Containers[0].Env {
		if e.Name == readinessPortEnvVar {
			env = e.Value
		}
	}
	if env != strconv.Itoa(int(workerReadinessPort)) {
		t.Errorf("%s = %q, want %d", readinessPortEnvVar, env, workerReadinessPort)
	}
}

// An extraPodSpec template is passed through verbatim and may already pin the
// port. The probe has to follow it, or it polls a port nothing listens on.
func TestReadinessProbeFollowsAnOverriddenPort(t *testing.T) {
	spec := &corev1.PodSpec{Containers: []corev1.Container{{
		Name: "main",
		Env:  []corev1.EnvVar{{Name: readinessPortEnvVar, Value: "31234"}},
	}}}
	injectWorkerRolloutDefaults(spec, 0, true, nil, nil)

	if got := spec.Containers[0].ReadinessProbe.HTTPGet.Port.IntVal; got != 31234 {
		t.Errorf("probe port = %d, want the container's own 31234", got)
	}
	count := 0
	for _, e := range spec.Containers[0].Env {
		if e.Name == readinessPortEnvVar {
			count++
		}
	}
	if count != 1 {
		t.Errorf("%s appears %d times, want 1", readinessPortEnvVar, count)
	}
}

// Surging is only safe while readiness means "registered". A worker whose
// probe was skipped reports Ready the moment it is Running, so surging would
// retire the pod that is still serving in favour of one still loading weights
// -- worse than the bounded gap of rolling old-pod-first. Every shipped
// example that sets skipReadinessProbe on a serving worker depends on this.
//
// It also keeps whole-node pinned workers upgradeable: replicas=1 with a
// nodeSelector and every GPU on the host leaves no room for a surge pod, so
// maxSurge=1 would never complete.
func TestAWorkerWithoutAProbeRollsSurgeFree(t *testing.T) {
	surge, unavailable := rollingOf(t, inferav1alpha1.ServiceSpec{
		ComponentType:      inferav1alpha1.ComponentTypeWorker,
		SkipReadinessProbe: true,
	})
	if surge != 0 || unavailable != 1 {
		t.Errorf("maxSurge/maxUnavailable = %d/%d, want 0/1: no probe means no safe surge",
			surge, unavailable)
	}
}

// A hand-written probe carries no guarantee that Ready means registered -- a
// /health probe answers while the engine is still starting -- so it does not
// buy the right to surge either.
func TestAnExtraPodSpecProbeDoesNotEnableSurge(t *testing.T) {
	surge, unavailable := rollingOf(t, inferav1alpha1.ServiceSpec{
		ComponentType: inferav1alpha1.ComponentTypeWorker,
		ExtraPodSpec: &corev1.PodSpec{Containers: []corev1.Container{{
			Name:  "main",
			Image: "x",
			ReadinessProbe: &corev1.Probe{ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromInt32(30000)},
			}},
		}}},
	})
	if surge != 0 || unavailable != 1 {
		t.Errorf("maxSurge/maxUnavailable = %d/%d, want 0/1 for a foreign probe",
			surge, unavailable)
	}
}

// A valueFrom port is resolved by the kubelet, so the operator cannot know it.
// Injecting a probe on the default would poll a port the worker may not bind,
// and with maxUnavailable=0 that is an unrecoverable stall -- so no probe is
// injected, which in turn falls back to surge-free rolling.
func TestAnUnknowableReadinessPortSkipsTheProbe(t *testing.T) {
	spec := &corev1.PodSpec{Containers: []corev1.Container{{
		Name: "main",
		Env: []corev1.EnvVar{{
			Name: readinessPortEnvVar,
			ValueFrom: &corev1.EnvVarSource{
				ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Key: "port"},
			},
		}},
	}}}
	injectWorkerRolloutDefaults(spec, 0, true, nil, nil)
	if p := spec.Containers[0].ReadinessProbe; p != nil {
		t.Errorf("probe injected for an unknowable port: %+v", p.HTTPGet)
	}
}

// An unknowable readiness port reads as port 0, and so does a named-port probe
// (Port.IntVal is 0 when Port.Type is String). Neither may be mistaken for the
// readiness probe, or a foreign probe enables surge.
func TestANamedPortProbeWithUnknowablePortDoesNotEnableSurge(t *testing.T) {
	surge, unavailable := rollingOf(t, inferav1alpha1.ServiceSpec{
		ComponentType: inferav1alpha1.ComponentTypeWorker,
		ExtraPodSpec: &corev1.PodSpec{Containers: []corev1.Container{{
			Name:  "main",
			Image: "x",
			Env: []corev1.EnvVar{{
				Name: readinessPortEnvVar,
				ValueFrom: &corev1.EnvVarSource{
					ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Key: "port"},
				},
			}},
			ReadinessProbe: &corev1.Probe{ProbeHandler: corev1.ProbeHandler{
				HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: intstr.FromString("http")},
			}},
		}}},
	})
	if surge != 0 || unavailable != 1 {
		t.Errorf("maxSurge/maxUnavailable = %d/%d, want 0/1 for a named-port probe",
			surge, unavailable)
	}
}

// Go's Atoi accepts forms Python's int() does not, and a divergence here means
// the probe and the worker disagree on the port.
func TestReadinessPortRejectsFormsThePythonSideWouldNotAccept(t *testing.T) {
	for _, raw := range []string{" 30091", "30_091", "30091 ", "", "abc", "0", "70000"} {
		c := &corev1.Container{Env: []corev1.EnvVar{{Name: readinessPortEnvVar, Value: raw}}}
		if _, ok := readinessPortFrom(c); ok {
			t.Errorf("%q was accepted; the worker would bind something else", raw)
		}
	}
	c := &corev1.Container{Env: []corev1.EnvVar{{Name: readinessPortEnvVar, Value: "30091"}}}
	port, ok := readinessPortFrom(c)
	if !ok || port != 30091 {
		t.Errorf("readinessPortFrom = %d/%v, want 30091/true", port, ok)
	}
}
