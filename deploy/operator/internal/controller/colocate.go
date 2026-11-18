/*
 * Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Server/worker colocation: a graph of one server and one single-node mixed
 * worker with one replica runs both in the worker pod, the server as a second
 * container. The server keeps its Service; it selects the worker pod instead.
 */

package controller

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"

	inferav1alpha1 "github.com/amd/infera/deploy/operator/api/v1alpha1"
)

const (
	// routerLabel marks every pod that runs the server, standalone or
	// colocated. An external Service that fronts the router selects on it.
	routerLabel = "infera.amd.com/router"
	// colocatedServerContainer is the server container added to the worker pod.
	colocatedServerContainer = "infera-server"
)

// colocation names the server and worker services that share one pod.
type colocation struct {
	server string
	worker string
}

// colocationFor reports the server/worker pair that runs in one pod.
//
// The graph must be exactly one server and one worker. The worker is mixed,
// single-node, and has one replica; the server is single-node with one
// replica. GAIE deployments keep their own per-worker frontend layout.
func colocationFor(idep *inferav1alpha1.InferaDeployment) (colocation, bool) {
	if gaieEnabled(idep) || len(idep.Spec.Services) != 2 {
		return colocation{}, false
	}
	var c colocation
	for _, name := range sortedKeys(idep.Spec.Services) {
		svc := idep.Spec.Services[name]
		if svc.NumberOfNodes > 1 || replicasOf(svc) != 1 {
			return colocation{}, false
		}
		switch svc.ComponentType {
		case inferav1alpha1.ComponentTypeServer:
			c.server = name
		case inferav1alpha1.ComponentTypeWorker:
			if svc.Role != "" && svc.Role != inferav1alpha1.WorkerRoleMixed {
				return colocation{}, false
			}
			c.worker = name
		}
	}
	return c, c.server != "" && c.worker != ""
}

// primaryContainerIndex returns the index of the main/infera container, else 0.
func primaryContainerIndex(spec *corev1.PodSpec) int {
	for i := range spec.Containers {
		if _, ok := mainContainerNames[spec.Containers[i].Name]; ok {
			return i
		}
	}
	return 0
}

// applyColocatedServer adds the server's primary container to the worker pod.
//
// The container is taken from the server's own rendered template, so flat
// fields and extraPodSpec behave the same. Its resource requests are dropped:
// the server is CPU-light and shares the worker pod budget, and adding them
// could make a whole-node worker pod unschedulable. Server volumes the worker
// pod does not already declare are carried over by name.
func applyColocatedServer(idep *inferav1alpha1.InferaDeployment, c colocation, tmpl *corev1.PodTemplateSpec) {
	serverTmpl := basePodTemplate(idep, c.server, idep.Spec.Services[c.server])
	if len(serverTmpl.Spec.Containers) == 0 {
		return
	}
	server := *serverTmpl.Spec.Containers[primaryContainerIndex(&serverTmpl.Spec)].DeepCopy()
	server.Name = colocatedServerContainer
	server.Resources = corev1.ResourceRequirements{}
	tmpl.Spec.Containers = append(tmpl.Spec.Containers, server)

	present := make(map[string]bool, len(tmpl.Spec.Volumes))
	for _, v := range tmpl.Spec.Volumes {
		present[v.Name] = true
	}
	for _, v := range serverTmpl.Spec.Volumes {
		if !present[v.Name] {
			tmpl.Spec.Volumes = append(tmpl.Spec.Volumes, v)
			present[v.Name] = true
		}
	}
	setRouterLabel(tmpl)
}

// setRouterLabel marks a pod template as running the server.
func setRouterLabel(tmpl *corev1.PodTemplateSpec) {
	if tmpl.ObjectMeta.Labels == nil {
		tmpl.ObjectMeta.Labels = map[string]string{}
	}
	tmpl.ObjectMeta.Labels[routerLabel] = "true"
}

// colocationPortConflict reports a port the server and worker would both bind
// in the shared pod network: the server port against every worker container
// port and the worker readiness port.
func colocationPortConflict(idep *inferav1alpha1.InferaDeployment, c colocation) error {
	serverPort := servicePort(idep.Spec.Services[c.server])
	workerSpec := basePodTemplate(idep, c.worker, idep.Spec.Services[c.worker]).Spec
	for i := range workerSpec.Containers {
		w := &workerSpec.Containers[i]
		for _, p := range w.Ports {
			if p.ContainerPort == serverPort {
				return fmt.Errorf("server %q port %d is also a port of worker %q container %q; "+
					"they share one pod network when colocated", c.server, serverPort, c.worker, w.Name)
			}
		}
	}
	idx := primaryContainerIndex(&workerSpec)
	if len(workerSpec.Containers) > 0 {
		if port, ok := readinessPortFrom(&workerSpec.Containers[idx]); ok && port == serverPort {
			return fmt.Errorf("server %q port %d equals worker %q readiness port; "+
				"they share one pod network when colocated", c.server, serverPort, c.worker)
		}
	}
	return nil
}
