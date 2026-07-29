echo "### SGLANG COMMIT"
cd /sgl-workspace/sglang && git rev-parse HEAD
echo "### KEY LIBS"
python3 - <<'PY'
import importlib
for m in ["aiter","tilelang","sgl_kernel","triton","transformers","torch"]:
    try:
        x = importlib.import_module(m)
        print("  %-14s %s" % (m, getattr(x, "__version__", "no __version__")))
    except Exception as e:
        print("  %-14s IMPORT-ERR %s" % (m, type(e).__name__))
PY
echo "### PATCH STATE (0 everywhere = clean upstream)"
SG=/sgl-workspace/sglang/python/sglang/srt
for m in _q_mqa GLM52_BUG6 _glm52_match_page_table_rows GLM52_BUG2 VARIANT_B; do
  printf "  %-32s %s\n" "$m" \
    "$(grep -rho "$m" $SG/layers/attention/dsa/dsa_indexer.py \
        $SG/layers/attention/dsa_backend.py \
        $SG/speculative/base_spec_worker.py \
        $SG/speculative/eagle_worker_v2.py 2>/dev/null | wc -l)"
done
echo "### MODELS"
for d in /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 /shared_nfs/huggingface_models/zai-org/GLM-5.2-FP8; do
  printf "  %s\n" "$d"
  python3 -c "
import json,sys
c=json.load(open('$d/config.json'))
q=c.get('quantization_config') or {}
qm=q.get('quant_method') or (list(q.get('global_quant_config',{}).get('weight',{}).items())[:1] if q else '?')
print('     arch=%s layers=%s experts=%s index_topk=%s' % (c['architectures'][0], c['num_hidden_layers'], c.get('n_routed_experts'), c.get('index_topk')))
print('     quant=%s' % (q.get('quant_method') or 'quark/mxfp4'))
g=json.load(open('$d/generation_config.json'))
print('     generation_config: temperature=%s top_p=%s' % (g.get('temperature'), g.get('top_p')))
import os; print('     chat_template.jinja present:', os.path.exists('$d/chat_template.jinja'))
"
done
