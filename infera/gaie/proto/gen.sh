#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Regenerate the ext_proc gRPC/protobuf stubs from ext_proc.proto.
# Requires: pip install grpcio-tools
# The grpc stub's intra-package import is rewritten to a relative import so it
# resolves as infera.gaie.proto.ext_proc_pb2.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m grpc_tools.protoc \
  -I "$here" \
  --python_out="$here" \
  --grpc_python_out="$here" \
  "$here/ext_proc.proto"

# `import ext_proc_pb2` -> `from . import ext_proc_pb2`
sed -i 's/^import ext_proc_pb2 as/from . import ext_proc_pb2 as/' "$here/ext_proc_pb2_grpc.py"
echo "regenerated ext_proc_pb2.py + ext_proc_pb2_grpc.py"
