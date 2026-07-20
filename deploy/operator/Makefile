# infera-operator — minimal build/codegen targets.
# Operator manager image + chart registry. Override on the CLI, e.g.
#   make docker-build docker-push IMG=docker.io/inferaimage/infera:0.1.0
IMG ?= docker.io/inferaimage/infera-operator-manager:0.1.1
# OCI registry namespace the packaged Helm chart is pushed to (chart repo name
# comes from Chart.yaml: infera-operator).
CHART_REGISTRY ?= oci://docker.io/inferaimage
CHART_DIR ?= helm/infera-operator
CONTROLLER_GEN_VERSION ?= v0.17.1
CONTROLLER_GEN = go run sigs.k8s.io/controller-tools/cmd/controller-gen@$(CONTROLLER_GEN_VERSION)

.PHONY: generate manifests build vet test docker-build docker-push helm-package helm-push install deploy sample

## Generate DeepCopy methods.
generate:
	$(CONTROLLER_GEN) object paths=./api/...

## Generate CRD + RBAC manifests.
manifests:
	$(CONTROLLER_GEN) crd paths=./api/... output:crd:dir=config/crd/bases
	$(CONTROLLER_GEN) rbac:roleName=infera-operator-role paths=./internal/... output:rbac:dir=config/rbac

build: generate
	go build -o bin/manager ./cmd

vet:
	go vet ./...

test:
	go test ./...

docker-build:
	docker build -t $(IMG) .

## Push the operator manager image (requires `docker login`).
docker-push:
	docker push $(IMG)

## Package the Helm chart into a local .tgz.
helm-package:
	helm package $(CHART_DIR)

## Push the packaged chart as an OCI artifact (requires `helm registry login`).
helm-push: helm-package
	helm push infera-operator-$(shell awk '/^version:/{print $$2}' $(CHART_DIR)/Chart.yaml).tgz $(CHART_REGISTRY)

## Apply the CRD to the current kube context.
install: manifests
	kubectl apply -f config/crd/bases

## Apply a sample deployment.
sample:
	kubectl apply -f config/samples/
