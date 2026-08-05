# infera-operator — minimal build/codegen targets.
# The manager is a tag of the shared rocm/infera repo, alongside the engine
# images, so one repo holds everything that runs. The chart is pushed to
# rocm/infera-operator instead -- a separate OCI repo, so image tags and chart
# versions cannot collide in the same namespace.
#
#   docker pull rocm/infera:operator-v0.1.0
#   helm  pull  oci://docker.io/rocm/infera-operator --version 0.1.0
#
# Builds publish to the staging repo, as the engine images do; promotion to
# rocm/infera is a separate step. Override on the CLI, e.g.
#   make docker-build docker-push IMG=docker.io/<ns>/infera:operator-v0.1.0
IMG ?= docker.io/inferaimage/infera:operator-v0.1.0
# OCI registry namespace the packaged Helm chart is pushed to (chart repo name
# comes from Chart.yaml: infera-operator). Staging, like IMG.
CHART_REGISTRY ?= oci://docker.io/inferaimage
CHART_DIR ?= helm/infera-operator
CRD_DIR ?= config/crd/bases
# The chart ships its own copy of the CRD, so `helm install` can create it
# without a separate kubectl apply. It is a mirror of CRD_DIR, never edited by
# hand -- `manifests` refreshes both so a chart cannot lag behind the types.
CHART_CRD_DIR ?= $(CHART_DIR)/crds
CONTROLLER_GEN_VERSION ?= v0.17.1
CONTROLLER_GEN = go run sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_GEN_VERSION)

.PHONY: generate manifests sync-chart-crd build vet test docker-build docker-push helm-package helm-push install deploy sample

## Generate DeepCopy methods.
generate:
	$(CONTROLLER_GEN) object paths=./api/...

## Generate CRD + RBAC manifests, then mirror the CRD into the chart.
manifests:
	$(CONTROLLER_GEN) crd paths=./api/... output:crd:dir=$(CRD_DIR)
	$(CONTROLLER_GEN) rbac:roleName=infera-operator-role paths=./internal/... output:rbac:dir=config/rbac
	$(MAKE) sync-chart-crd

## Copy the generated CRD into the chart. Splitting this out keeps it runnable
## on its own, for the case where the chart copy drifted without the types
## changing.
sync-chart-crd:
	mkdir -p $(CHART_CRD_DIR)
	cp $(CRD_DIR)/*.yaml $(CHART_CRD_DIR)/

build: generate
	go build -o bin/manager ./cmd

vet:
	go vet ./...

test:
	go test ./...

## Build the manager image. Depends on the generators because the Dockerfile
## compiles from the source tree as-is: a stale zz_generated.deepcopy.go would
## be baked into the image, and a stale CRD would ship a chart that rejects the
## fields the manager already understands.
docker-build: generate manifests
	docker build -t $(IMG) .

## Push the operator manager image (requires `docker login`).
docker-push:
	docker push $(IMG)

## Package the Helm chart into a local .tgz. Depends on manifests so a chart
## can never be cut against a stale CRD -- the failure that produces is quiet:
## the chart installs, and the operator rejects fields the CRD has not heard of.
helm-package: manifests
	helm package $(CHART_DIR)

## Push the packaged chart as an OCI artifact (requires `helm registry login`).
helm-push: helm-package
	helm push infera-operator-$(shell awk '/^version:/{print $$2}' $(CHART_DIR)/Chart.yaml).tgz $(CHART_REGISTRY)

## Apply the CRD to the current kube context.
install: manifests
	kubectl apply -f $(CRD_DIR)

## Apply a sample deployment.
sample:
	kubectl apply -f config/samples/
