/*
Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT
*/

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// InferaScalingAdapterSpec points at one service inside an InferaDeployment and
// owns its replica count.
type InferaScalingAdapterSpec struct {
	// DeploymentRef is the InferaDeployment to scale, in this namespace.
	// +kubebuilder:validation:MinLength=1
	DeploymentRef string `json:"deploymentRef"`

	// ServiceName is the key in that deployment's `spec.services` map.
	// +kubebuilder:validation:MinLength=1
	ServiceName string `json:"serviceName"`

	// Replicas is the desired count, and the field `/scale` writes.
	//
	// Left unset the adapter is inert: the InferaDeployment's own
	// `spec.services[<name>].replicas` still applies. That makes adding an
	// adapter a safe no-op until something actually scales, so an autoscaler
	// can be attached and observed before it is trusted.
	// +optional
	// +kubebuilder:validation:Minimum=0
	Replicas *int32 `json:"replicas,omitempty"`
}

// InferaScalingAdapterStatus carries what `/scale` reads back.
type InferaScalingAdapterStatus struct {
	// Replicas is the count observed on the workload -- not the desired count
	// echoed back.
	//
	// The distinction is the whole reason this field exists. HorizontalPodAutoscaler
	// computes `desired = ceil(current * metric/target)`; if `current` is really
	// the desired value it never lags reality, so during a multi-minute model load
	// the autoscaler cannot tell that a scale-up has not landed yet and keeps
	// multiplying.
	// +optional
	Replicas int32 `json:"replicas"`

	// ReadyReplicas is how many of those are actually serving.
	// +optional
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// Selector is a serialized label selector matching the scaled pods.
	// HorizontalPodAutoscaler requires this to be a *string*, not a structured
	// selector, and refuses to scale a resource whose scale subresource does not
	// provide one.
	// +optional
	Selector string `json:"selector,omitempty"`

	// ObservedGeneration is the adapter generation this status reflects.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Conditions carries `Ready` (the target resolves and is being driven) and
	// `Degraded` (it does not).
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// InferaScalingAdapter gives one service of an InferaDeployment a standard
// Kubernetes `/scale` subresource.
//
// An InferaDeployment cannot carry `/scale` itself, and this is a property of
// its shape rather than a missing feature: `spec.services` is a map with
// user-chosen keys, while the scale subresource requires `specReplicasPath` to
// be a *static* dot-notation JSONPath under `.spec`. There is no way to write
// "the replicas of an arbitrary map entry".
//
// So scaling gets its own object, one per scalable service. That makes
// `kubectl scale`, HorizontalPodAutoscaler, KEDA and a custom planner all work
// through the same standard interface, with no per-tool support in this
// operator.
//
// While an adapter exists with `spec.replicas` set, it is the single writer of
// that service's replica count: the InferaDeployment reconciler reads the
// adapter instead of the CR's own `replicas`, so the two cannot fight. Delete
// the adapter, or clear `spec.replicas`, and the CR is back in charge.
//
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.replicas,selectorpath=.status.selector
// +kubebuilder:resource:shortName=isa
// +kubebuilder:printcolumn:name="Target",type=string,JSONPath=`.spec.deploymentRef`
// +kubebuilder:printcolumn:name="Service",type=string,JSONPath=`.spec.serviceName`
// +kubebuilder:printcolumn:name="Desired",type=integer,JSONPath=`.spec.replicas`
// +kubebuilder:printcolumn:name="Current",type=integer,JSONPath=`.status.replicas`
// +kubebuilder:printcolumn:name="Ready",type=integer,JSONPath=`.status.readyReplicas`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
type InferaScalingAdapter struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   InferaScalingAdapterSpec   `json:"spec,omitempty"`
	Status InferaScalingAdapterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
type InferaScalingAdapterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []InferaScalingAdapter `json:"items"`
}

func init() {
	SchemeBuilder.Register(&InferaScalingAdapter{}, &InferaScalingAdapterList{})
}
