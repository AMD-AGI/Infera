/*
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/

package controller

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
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
	injectWorkerRolloutDefaults(spec, 0, 8080, false, nil, nil)
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
	injectWorkerRolloutDefaults(spec, 0, 8080, false, nil, nil)
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
	injectWorkerRolloutDefaults(spec, 0, 8080, false, []string{"--drain-timeout", "30"}, nil)

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
	injectWorkerRolloutDefaults(spec, 0, 8080, false, nil, env)

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
