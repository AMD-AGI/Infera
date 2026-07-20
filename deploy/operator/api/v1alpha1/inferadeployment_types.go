/*
 * Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Infera operator API: a single CRD describing the inference graph
 * (a router/server plus one or more worker pools), reconciled into
 * Deployments (single-node) or LeaderWorkerSets (multi-node), with AMD GPUs
 * (amd.com/gpu) and an optional operator-managed NATS for the KV-event plane.
 */

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ComponentType is the role of a service in the inference graph.
// +kubebuilder:validation:Enum=server;worker
type ComponentType string

const (
	// ComponentTypeServer is the infera.server router/frontend (no GPU).
	ComponentTypeServer ComponentType = "server"
	// ComponentTypeWorker is a infera.engine.{sglang,vllm} worker (GPU).
	ComponentTypeWorker ComponentType = "worker"
)

// WorkerRole is the disaggregation role of a worker pool.
// +kubebuilder:validation:Enum=mixed;prefill;decode
type WorkerRole string

const (
	WorkerRoleMixed   WorkerRole = "mixed"
	WorkerRolePrefill WorkerRole = "prefill"
	WorkerRoleDecode  WorkerRole = "decode"
)

// DeploymentState is a high-level lifecycle status.
// +kubebuilder:validation:Enum=initializing;pending;ready;failed
type DeploymentState string

const (
	StateInitializing DeploymentState = "initializing"
	StatePending      DeploymentState = "pending"
	StateReady        DeploymentState = "ready"
	StateFailed       DeploymentState = "failed"
)

// Resources describes a service's compute request/limit knobs. GPUs are
// requested via the amd.com/gpu extended resource by default.
type Resources struct {
	// GPU is the number of GPUs per pod (mapped to the GPUType resource on
	// both requests and limits). 0/omitted means no GPU (e.g. the server).
	// +kubebuilder:validation:Minimum=0
	// +optional
	GPU int32 `json:"gpu,omitempty"`
	// GPUType is the extended resource name for GPUs. Defaults to amd.com/gpu.
	// +kubebuilder:default="amd.com/gpu"
	// +optional
	GPUType string `json:"gpuType,omitempty"`
	// CPU request (e.g. "8"). +optional
	CPU string `json:"cpu,omitempty"`
	// Memory request (e.g. "64Gi"). +optional
	Memory string `json:"memory,omitempty"`
	// SharedMemory sizes the /dev/shm emptyDir (e.g. "32Gi"); needed for
	// torch/NCCL/RDMA staging. +optional
	SharedMemory string `json:"sharedMemory,omitempty"`
}

// ServiceSpec is one node in the inference graph.
type ServiceSpec struct {
	// ComponentType selects server (router) or worker (engine).
	// +kubebuilder:validation:Required
	ComponentType ComponentType `json:"componentType"`
	// Role is the disaggregation role for workers (mixed/prefill/decode).
	// Ignored for the server. +optional
	// +kubebuilder:default=mixed
	Role WorkerRole `json:"role,omitempty"`
	// Replicas is the number of pods (single-node) or LeaderWorkerSet groups
	// (multi-node). +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=0
	Replicas *int32 `json:"replicas,omitempty"`
	// NumberOfNodes per replica. 1 => a Deployment; >1 => a LeaderWorkerSet
	// (multi-node tensor/pipeline parallel or PD group). +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	NumberOfNodes int32 `json:"numberOfNodes,omitempty"`
	// SkipReadinessProbe, when true, tells the operator NOT to inject its
	// default /health readiness probe. Use for idle-pod / SSH-managed workers
	// (e.g. the optimizer create-infera model) whose engine is (re)launched
	// out-of-band after deploy: a /health probe would never pass at deploy time,
	// leaving readyReplicas=0 and the InferaDeployment stuck in "pending". +optional
	SkipReadinessProbe bool `json:"skipReadinessProbe,omitempty"`
	// Image overrides spec.image for this service. +optional
	Image string `json:"image,omitempty"`
	// Port the container listens on (server default 8000, worker 30000). +optional
	Port int32 `json:"port,omitempty"`
	// Resources for each pod. +optional
	Resources *Resources `json:"resources,omitempty"`
	// Args appended to the container entrypoint (engine/server flags). +optional
	Args []string `json:"args,omitempty"`
	// Env extra environment variables for this service. +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// ExtraPodSpec, when set, is used verbatim as this service's pod spec
	// instead of the operator building one from the flat fields above. It is
	// the integration seam for an external orchestrator that renders a
	// full pod template — image/command/resources/env are injected into the
	// container named "main"/"infera", and pod-level fields (initContainers,
	// volumes, affinity, schedulerName, hostNetwork, ...) are honored as-is.
	// ComponentType / Role / Replicas / NumberOfNodes still drive the workload
	// kind (Deployment vs LeaderWorkerSet) and Service creation. +optional
	ExtraPodSpec *corev1.PodSpec `json:"extraPodSpec,omitempty"`
	// PodLabels are extra labels merged onto this service's pod template
	// (in addition to the operator's own infera.amd.com/* selector labels).
	// This is the pass-through seam for an external orchestrator that needs
	// its own tracking label on the rendered pods so its pod syncer can
	// associate them to a workload. ExtraPodSpec is a bare corev1.PodSpec with
	// no metadata, so pod labels cannot ride along inside it; they are carried
	// here instead. +optional
	PodLabels map[string]string `json:"podLabels,omitempty"`
}

// NATSSpec controls the operator-managed NATS (JetStream) for the KV-event
// plane. etcd is referenced externally via spec.etcdEndpoint.
type NATSSpec struct {
	// Deploy toggles whether the operator creates a NATS StatefulSet+Service.
	// +kubebuilder:default=true
	Deploy bool `json:"deploy,omitempty"`
	// Image for the NATS server. +kubebuilder:default="nats:latest"
	Image string `json:"image,omitempty"`
	// StorageSize for the JetStream PVC. +kubebuilder:default="4Gi"
	StorageSize string `json:"storageSize,omitempty"`
	// StorageClassName for the JetStream PVC. +optional
	StorageClassName string `json:"storageClassName,omitempty"`
}

// GAIESpec configures Gateway API Inference Extension (GAIE) integration. When
// enabled the operator:
//   - injects a `infera.server --router-mode direct` frontend sidecar into
//     each worker pod (the gateway routes to it; it honours the EPP's
//     x-worker-instance-id / x-prefill-instance-id picks),
//   - deploys an Endpoint Picker (EPP, `python -m infera.gaie`) Deployment +
//     Service that scores workers with the same kv-aware policy as the server,
//   - creates an InferencePool selecting the worker pods (frontend port) and an
//     HTTPRoute attaching the pool to an existing Inference Gateway.
//
// This is the per-worker-sidecar topology used by the upstream GAIE
// integration. The standalone `server` service is not required in this mode.
type GAIESpec struct {
	// Enabled toggles creation of the EPP + InferencePool + HTTPRoute and the
	// per-worker frontend sidecar. +kubebuilder:default=false
	Enabled bool `json:"enabled,omitempty"`
	// EPPImage is the container image for the EPP and the frontend sidecar
	// (`python -m infera.gaie` / `infera.server`). Defaults to spec.Image.
	// +optional
	EPPImage string `json:"eppImage,omitempty"`
	// TokenizerPath is the HF model id or local tokenizer path the EPP uses for
	// token-aware kv scoring; must match the workers' tokenizer.
	// +kubebuilder:validation:Required
	TokenizerPath string `json:"tokenizerPath"`
	// GatewayName is the Inference Gateway the generated HTTPRoute attaches to.
	// +kubebuilder:validation:Required
	GatewayName string `json:"gatewayName"`
	// GatewayNamespace is the Gateway's namespace (defaults to the IDEP's). +optional
	GatewayNamespace string `json:"gatewayNamespace,omitempty"`
	// Hostnames optionally restricts the HTTPRoute to these hostnames. +optional
	Hostnames []string `json:"hostnames,omitempty"`
	// FrontendPort is the port the injected frontend sidecar listens on (and the
	// InferencePool target port). +kubebuilder:default=8000
	// +optional
	FrontendPort int32 `json:"frontendPort,omitempty"`
	// ModelName is the served model id, used for the EPP's /v1/models view and
	// route matching. Informational; routing is by header. +optional
	ModelName string `json:"modelName,omitempty"`
}

// InferaDeploymentSpec is the desired inference graph.
type InferaDeploymentSpec struct {
	// BackendFramework selects the engine module (sglang or vllm).
	// +kubebuilder:validation:Enum=sglang;vllm
	// +kubebuilder:default=sglang
	BackendFramework string `json:"backendFramework,omitempty"`
	// Image is the default container image for services built from the flat
	// ServiceSpec fields (overridable per service via ServiceSpec.Image). When a
	// service supplies ExtraPodSpec, the image lives on that pod's container
	// instead, so this top-level field may be empty. +optional
	Image string `json:"image,omitempty"`
	// DiscoveryBackend selects how the server discovers workers: "kubernetes"
	// (default) watches worker Pods via the in-cluster API server (no etcd; the
	// operator provisions a ServiceAccount + Role so workers can self-register
	// into their Pod annotation), or "etcd" (external etcd at EtcdEndpoint).
	// +kubebuilder:validation:Enum=kubernetes;etcd
	// +kubebuilder:default=kubernetes
	// +optional
	DiscoveryBackend string `json:"discoveryBackend,omitempty"`
	// EtcdEndpoint is the external etcd the workers/server use for discovery
	// (e.g. "etcd:2379"). Required only when DiscoveryBackend is "etcd".
	// +optional
	EtcdEndpoint string `json:"etcdEndpoint,omitempty"`
	// NATS configures the operator-managed JetStream broker for KV events. +optional
	NATS *NATSSpec `json:"nats,omitempty"`
	// GAIE configures Gateway API Inference Extension integration (EPP +
	// InferencePool + HTTPRoute + per-worker frontend sidecar). +optional
	GAIE *GAIESpec `json:"gaie,omitempty"`
	// Envs are environment variables applied to every service. +optional
	Envs []corev1.EnvVar `json:"envs,omitempty"`
	// Services is the named set of graph components (e.g. server, prefill, decode).
	// +kubebuilder:validation:Required
	Services map[string]ServiceSpec `json:"services"`
}

// ServiceStatus reports replica counts for one service.
type ServiceStatus struct {
	// Kind is the underlying workload kind ("Deployment" or "LeaderWorkerSet").
	Kind string `json:"kind,omitempty"`
	// Replicas is the desired replica count.
	Replicas int32 `json:"replicas"`
	// ReadyReplicas is the number of ready replicas/groups.
	ReadyReplicas int32 `json:"readyReplicas"`
}

// GAIEStatus reports the observed state of the Gateway API Inference Extension
// resources the operator manages for this deployment.
type GAIEStatus struct {
	// EPPReplicas is the desired Endpoint Picker replica count.
	EPPReplicas int32 `json:"eppReplicas"`
	// EPPReadyReplicas is the number of ready Endpoint Picker replicas.
	EPPReadyReplicas int32 `json:"eppReadyReplicas"`
	// InferencePoolName is the generated InferencePool. +optional
	InferencePoolName string `json:"inferencePoolName,omitempty"`
	// InferencePoolReady is true once the InferencePool exists and (if it
	// reports conditions) is Accepted. +optional
	InferencePoolReady bool `json:"inferencePoolReady,omitempty"`
	// HTTPRouteName is the generated HTTPRoute. +optional
	HTTPRouteName string `json:"httpRouteName,omitempty"`
	// HTTPRouteAccepted is true once a parent Gateway reports the route as
	// Accepted. +optional
	HTTPRouteAccepted bool `json:"httpRouteAccepted,omitempty"`
	// Ready is the roll-up: EPP has >=1 ready replica, the InferencePool is
	// ready, and the HTTPRoute is accepted. +optional
	Ready bool `json:"ready,omitempty"`
}

// InferaDeploymentStatus is the observed state.
type InferaDeploymentStatus struct {
	// ObservedGeneration is the most recent generation reconciled. +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
	// State is the high-level lifecycle state.
	// +kubebuilder:default=initializing
	State DeploymentState `json:"state,omitempty"`
	// Conditions are the latest observed conditions.
	// +optional
	// +patchMergeKey=type
	// +patchStrategy=merge
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
	// Services holds per-service replica status keyed by service name. +optional
	Services map[string]ServiceStatus `json:"services,omitempty"`
	// GAIE holds the status of the Gateway API Inference Extension resources,
	// set only when spec.gaie.enabled is true. +optional
	GAIE *GAIEStatus `json:"gaie,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=idep
// +kubebuilder:printcolumn:name="Backend",type="string",JSONPath=".spec.backendFramework"
// +kubebuilder:printcolumn:name="State",type="string",JSONPath=".status.state"
// +kubebuilder:printcolumn:name="GAIE",type="boolean",JSONPath=".status.gaie.ready",priority=1
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// InferaDeployment is the Schema for the inferadeployments API.
type InferaDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   InferaDeploymentSpec   `json:"spec,omitempty"`
	Status InferaDeploymentStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// InferaDeploymentList contains a list of InferaDeployment.
type InferaDeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []InferaDeployment `json:"items"`
}

func init() {
	SchemeBuilder.Register(&InferaDeployment{}, &InferaDeploymentList{})
}
