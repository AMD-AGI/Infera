#!/usr/bin/env bash
# Infera in a vLLM 0.34.0 container (the image the PD experiments use): the launcher's vllm serve argument parsing
# (tests/engine/vllm/test_disagg_allow_tcp_args.py, importorskip("vllm")) on origin/main, which is expected to fail
# on 0.34's moved make_arg_parser, and on the arg-parser branch, which should pass; plus the arg-parser and DP engine
# LB unit tests against the real vLLM.
# key: for d in main vllm-034-arg-parser vllm-dp-engine-lb; do git -C /home/liyingli/bench_agentx/upstream-20261119/iwt/$d rev-parse HEAD; done
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready v034 || exit 1
docker rm -f up-infera-v034 > /dev/null 2>&1
docker run --rm --name up-infera-v034 --network host -v "$U/iwt:/src:ro" -v "$OUT:/out" -v "$NL/pip:/root/.cache/pip" \
    -e PYTHONDONTWRITEBYTECODE=1 --entrypoint bash "$IMG_V034" -c "
pip install -q pytest pytest-asyncio 'httpx>=0.28' xxhash msgspec msgpack nats-py prometheus-client pyzmq > /out/pip.log 2>&1
python3 -c 'import vllm; print(\"vllm\", vllm.__version__)'
t=tests/engine/vllm/test_disagg_allow_tcp_args.py
echo '== main (expected to fail on vLLM 0.34)'
cd /src/main && PYTHONPATH=/src/main python3 -m pytest -p no:cacheprovider -q \$t --junitxml=/out/main.xml 2>&1 | tail -5
echo '== vllm-034-arg-parser'
cd /src/vllm-034-arg-parser && PYTHONPATH=/src/vllm-034-arg-parser python3 -m pytest -p no:cacheprovider -q \$t \
    tests/unit/engine/test_vllm_server_arg_parser.py --junitxml=/out/argparser.xml 2>&1 | tail -5
echo '== vllm-dp-engine-lb'
cd /src/vllm-dp-engine-lb && PYTHONPATH=/src/vllm-dp-engine-lb python3 -m pytest -p no:cacheprovider -q \
    tests/unit/engine/test_vllm_dp_identity.py --junitxml=/out/dplb.xml 2>&1 | tail -5
" 2>&1 | tee "$OUT/run.txt"
python3 "$U/checks/junit_diff.py" /dev/null "$OUT/argparser.xml" | tail -3 && \
python3 "$U/checks/junit_diff.py" /dev/null "$OUT/dplb.xml" | tail -3
