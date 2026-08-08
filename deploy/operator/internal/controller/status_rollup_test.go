/*
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/

package controller

import (
	"testing"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

// ServiceStatus.Replicas reports what the workload has, not what was asked
// for -- an autoscaler cannot tell a scale-up has not landed if `current` is
// the number it just requested. That makes it the wrong side of the readiness
// comparison: `ready < observed` only catches Pods that exist and are not
// ready, and says nothing about Pods that were never created at all.
//
// Which is the case that matters. A worker Pod that cannot be scheduled -- no
// GPU, quota exhausted, a node taint -- never reaches the ReplicaSet's
// status.replicas, so ready equals observed and the whole deployment reports
// itself ready on a fraction of its capacity. `.status.state` is a
// printcolumn and the natural readiness gate for anything orchestrating on
// top, so this is what decides whether traffic is sent.
func TestReadyNeedsTheReplicaCountThatWasAskedFor(t *testing.T) {
	three := int32(3)
	svcs := map[string]inferav1alpha1.ServiceSpec{
		"decode": {Replicas: &three},
	}

	cases := []struct {
		name     string
		observed inferav1alpha1.ServiceStatus
		want     inferav1alpha1.DeploymentState
	}{
		{
			name:     "every requested replica is up",
			observed: inferav1alpha1.ServiceStatus{Replicas: 3, ReadyReplicas: 3},
			want:     inferav1alpha1.StateReady,
		},
		{
			name:     "a replica could not be scheduled, so it is not in the workload at all",
			observed: inferav1alpha1.ServiceStatus{Replicas: 2, ReadyReplicas: 2},
			want:     inferav1alpha1.StatePending,
		},
		{
			name:     "all present, one still starting",
			observed: inferav1alpha1.ServiceStatus{Replicas: 3, ReadyReplicas: 2},
			want:     inferav1alpha1.StatePending,
		},
		{
			name:     "nothing up yet",
			observed: inferav1alpha1.ServiceStatus{Replicas: 0, ReadyReplicas: 0},
			want:     inferav1alpha1.StatePending,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := rollupState(map[string]inferav1alpha1.ServiceStatus{"decode": c.observed}, svcs)
			if got != c.want {
				t.Fatalf("state = %q, want %q (spec asked for %d, workload has %d/%d)",
					got, c.want, three, c.observed.ReadyReplicas, c.observed.Replicas)
			}
		})
	}
}

// A service the spec no longer mentions must not hold the deployment back,
// and one with no status yet must not read as satisfied.
func TestRollupHandlesServicesMissingFromEitherSide(t *testing.T) {
	one := int32(1)
	specs := map[string]inferav1alpha1.ServiceSpec{"decode": {Replicas: &one}}

	if got := rollupState(map[string]inferav1alpha1.ServiceStatus{}, specs); got != inferav1alpha1.StatePending {
		t.Fatalf("no status reported yet: state = %q, want pending", got)
	}

	// Status carries a service the spec dropped; the live one is satisfied.
	svcs := map[string]inferav1alpha1.ServiceStatus{
		"decode": {Replicas: 1, ReadyReplicas: 1},
		"stale":  {Replicas: 0, ReadyReplicas: 0},
	}
	if got := rollupState(svcs, specs); got != inferav1alpha1.StateReady {
		t.Fatalf("a service no longer in the spec blocked readiness: state = %q", got)
	}
}
