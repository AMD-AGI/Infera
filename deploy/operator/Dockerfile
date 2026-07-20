# Build the infera-operator manager binary.
FROM golang:1.25 AS builder
WORKDIR /workspace
COPY go.mod go.sum ./
RUN go mod download
COPY cmd/ cmd/
COPY api/ api/
COPY internal/ internal/
RUN CGO_ENABLED=0 GOOS=linux go build -a -o manager ./cmd

# Minimal runtime image.
FROM gcr.io/distroless/static:nonroot
WORKDIR /
COPY --from=builder /workspace/manager .
USER 65532:65532
ENTRYPOINT ["/manager"]
