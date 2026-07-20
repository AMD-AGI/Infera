{{/*
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
*/}}

{{- define "infera-operator.name" -}}
infera-operator
{{- end -}}

{{- define "infera-operator.fullname" -}}
{{- $name := include "infera-operator.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "infera-operator.labels" -}}
app.kubernetes.io/name: {{ include "infera-operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}

{{- define "infera-operator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "infera-operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "infera-operator.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "infera-operator.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}
