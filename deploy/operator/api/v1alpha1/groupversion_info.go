/*
 * Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

// Package v1alpha1 contains the Infera operator API (group infera.amd.com).
// +kubebuilder:object:generate=true
// +groupName=infera.amd.com
package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/scheme"
)

var (
	// GroupVersion is group version used to register these objects.
	GroupVersion = schema.GroupVersion{Group: "infera.amd.com", Version: "v1alpha1"}

	// SchemeBuilder is used to add go types to the GroupVersionKind scheme.
	SchemeBuilder = &scheme.Builder{GroupVersion: GroupVersion}

	// AddToScheme adds the types in this group-version to the given scheme.
	AddToScheme = SchemeBuilder.AddToScheme
)
