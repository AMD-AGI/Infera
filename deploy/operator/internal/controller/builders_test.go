/*
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/

package controller

import (
	"testing"

	corev1 "k8s.io/api/core/v1"

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
			if got := graceSecondsFor(c.args, nil); got != c.want {
				t.Fatalf("graceSecondsFor(%v) = %d, want %d", c.args, got, c.want)
			}
		})
	}
}

// The budget must actually hold, not merely be larger than the old constant.
func TestGraceCoversTheWholeShutdown(t *testing.T) {
	for _, drain := range []int{30, 60, 120, 300} {
		args := []string{"--drain-timeout", itoa(drain)}
		grace := graceSecondsFor(args, nil)
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
	injectWorkerRolloutDefaults(spec, 0, 8080, false, nil, nil)
	if spec.TerminationGracePeriodSeconds == nil {
		t.Fatal("grace not set")
	}
	want := int64(workerPreStopDrainSeconds + 240 + workerTeardownHeadroomSeconds)
	if *spec.TerminationGracePeriodSeconds != want {
		t.Fatalf("grace = %d, want %d", *spec.TerminationGracePeriodSeconds, want)
	}
}

// An adapter owns its service's replica count; everything else keeps using the
// CR's. Getting this wrong in either direction is bad: ignoring the adapter
// makes `/scale` a no-op, and applying it too broadly makes a single autoscaler
// silently resize pools nobody pointed it at.
func TestEffectiveReplicas(t *testing.T) {
	three := int32(3)
	svc := inferav1alpha1.ServiceSpec{Replicas: &three}

	if got := effectiveReplicas(svc, "worker", nil); got != 3 {
		t.Fatalf("no adapters: got %d, want the CR's 3", got)
	}
	if got := effectiveReplicas(svc, "worker", map[string]int32{"worker": 7}); got != 7 {
		t.Fatalf("adapter present: got %d, want 7", got)
	}
	if got := effectiveReplicas(svc, "worker", map[string]int32{"prefill": 7}); got != 3 {
		t.Fatalf("adapter for another service: got %d, want the CR's 3", got)
	}
	// Zero is a legitimate target, not "unset" -- an autoscaler scaling a pool
	// to zero must not silently fall back to the CR's count.
	if got := effectiveReplicas(svc, "worker", map[string]int32{"worker": 0}); got != 0 {
		t.Fatalf("adapter asking for 0: got %d, want 0", got)
	}
	// The CR default when it says nothing either.
	if got := effectiveReplicas(inferav1alpha1.ServiceSpec{}, "worker", nil); got != 1 {
		t.Fatalf("nothing set anywhere: got %d, want the default 1", got)
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
	if got := graceSecondsFor(nil, env); got != want {
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
	if got := graceSecondsFor(args, env); got != want {
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
	if got := graceSecondsFor(nil, from); got != workerTerminationGraceSeconds {
		t.Fatalf("valueFrom: grace = %d, want the floor %d", got, workerTerminationGraceSeconds)
	}
	for _, v := range []string{"", "abc", "0", "-5"} {
		env := []corev1.EnvVar{{Name: drainTimeoutEnvVar, Value: v}}
		if got := graceSecondsFor(nil, env); got != workerTerminationGraceSeconds {
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
	injectWorkerRolloutDefaults(spec, 0, 8080, false, nil, nil)
	if spec.TerminationGracePeriodSeconds == nil {
		t.Fatal("grace not set")
	}
	want := int64(workerPreStopDrainSeconds + 240 + workerTeardownHeadroomSeconds)
	if *spec.TerminationGracePeriodSeconds != want {
		t.Fatalf("grace = %d, want %d", *spec.TerminationGracePeriodSeconds, want)
	}
}
