#!/usr/bin/env bash
# Runs INSIDE the vllm-ascend image. Expects /fork = host checkout of the fork (ro),
# /out = host output dir (rw). Read-only with respect to the image.
set -u
OUT=/out
SRC=$(python3 - <<'PY'
import importlib.util as u
s = u.find_spec('vllm_ascend')
print(list(s.submodule_search_locations)[0] if s else '')
PY
)
ROOT=$(dirname "$SRC")
FRK=/fork/vllm_ascend

# ---------- 10: versions / environment
{
  echo "ROOT=$ROOT"; echo "SRC=$SRC"
  python3 -V 2>&1
  pip show vllm vllm-ascend 2>/dev/null | grep -E '^(Name|Version|Location)'
  python3 -c "import torch, torch_npu; print('torch', torch.__version__, '| torch_npu', torch_npu.__version__)" 2>&1 | tail -2
  echo "--- /usr/local/Ascend ---"; ls -d /usr/local/Ascend/* 2>/dev/null
  echo "--- py count ---"; find "$SRC" -name '*.py' | wc -l
  echo "--- so count ---";  find "$SRC" -name '*.so' | wc -l
  echo "--- package size ---"; du -sh "$SRC" 2>/dev/null
} > "$OUT/10_image_versions.txt" 2>&1

# ---------- 11/12: serve flags
vllm serve --help=all > /tmp/help.txt 2>/dev/null
{
  echo "help_lines=$(wc -l < /tmp/help.txt)"
  for f in enable-prefix-caching compilation-config speculative-config additional-config \
           engram-config async-scheduling swa-bounded-replay kv-cache-dtype tokenizer-mode \
           quantization safetensors-load-strategy max-num-batched-tokens max-num-seqs \
           data-parallel-size tensor-parallel-size enable-expert-parallel block-size \
           trust-remote-code enforce-eager; do
    printf '%-30s %s\n' "$f" "$(grep -c -- "$f" /tmp/help.txt)"
  done
} > "$OUT/11_flags_present.txt" 2>&1
grep -n -A3 -E "^ +--(tokenizer-mode|kv-cache-dtype|quantization|safetensors-load-strategy|kv-transfer-config) " \
  /tmp/help.txt > "$OUT/12_flags_values.txt" 2>&1

# ---------- 20-22: image tree vs fork tree (py lists + non-py artifacts)
(cd "$SRC" && find . -name '*.py' | sort > /tmp/img_py.txt)
if [ -d "$FRK" ]; then (cd "$FRK" && find . -name '*.py' | sort > /tmp/frk_py.txt); else : > /tmp/frk_py.txt; fi
{
  echo "img_py=$(wc -l < /tmp/img_py.txt)  fork_py=$(wc -l < /tmp/frk_py.txt)"
  echo "only_in_image=$(comm -23 /tmp/img_py.txt /tmp/frk_py.txt | wc -l)"
  echo "only_in_fork=$(comm -13 /tmp/img_py.txt /tmp/frk_py.txt | wc -l)"
} > "$OUT/20_tree_py_counts.txt" 2>&1
comm -23 /tmp/img_py.txt /tmp/frk_py.txt > "$OUT/21_py_only_in_image.txt"
comm -13 /tmp/img_py.txt /tmp/frk_py.txt > "$OUT/22_py_only_in_fork.txt"

(cd "$SRC" && find . -type f ! -name '*.py' ! -name '*.pyc' ! -path '*__pycache__*' | sort > /tmp/img_np.txt)
if [ -d "$FRK" ]; then (cd "$FRK" && find . -type f ! -name '*.py' ! -name '*.pyc' ! -path '*__pycache__*' | sort > /tmp/frk_np.txt); else : > /tmp/frk_np.txt; fi
{
  echo "img_non_py=$(wc -l < /tmp/img_np.txt)  fork_non_py=$(wc -l < /tmp/frk_np.txt)"
  echo "non_py_only_in_image=$(comm -23 /tmp/img_np.txt /tmp/frk_np.txt | wc -l)"
} > "$OUT/23_tree_artifact_counts.txt" 2>&1
comm -23 /tmp/img_np.txt /tmp/frk_np.txt > "$OUT/24_artifacts_only_in_image.txt"

# ---------- 30/31: the one file we patch
cp -a "$SRC/attention/dsa_v41.py" "$OUT/dsa_v41_image.py" 2>/dev/null
{
  md5sum "$SRC/attention/dsa_v41.py"
  wc -l "$SRC/attention/dsa_v41.py"
  [ -f "$FRK/attention/dsa_v41.py" ] && { md5sum "$FRK/attention/dsa_v41.py"; wc -l "$FRK/attention/dsa_v41.py"; }
} > "$OUT/30_dsa_v41_md5.txt" 2>&1
if [ -f "$FRK/attention/dsa_v41.py" ]; then
  diff -u "$FRK/attention/dsa_v41.py" "$SRC/attention/dsa_v41.py" > "$OUT/31_dsa_v41_fork_vs_image.diff" 2>&1
fi

# ---------- 40: which custom CANN ops ship in this image
{
  echo "=== _cann_ops_custom size ==="; du -sh "$SRC/_cann_ops_custom" 2>/dev/null
  echo "=== dirs (depth<=5) ==="; find "$SRC/_cann_ops_custom" -maxdepth 5 -type d 2>/dev/null | head -40
  echo "=== built kernel dirs (op names) ==="
  ls "$SRC/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel" 2>/dev/null | head -120
  echo "=== op proto json ==="
  find "$SRC/_cann_ops_custom" -name '*.json' 2>/dev/null | head -20
} > "$OUT/40_custom_ops.txt" 2>&1

echo "container-side done: $(ls "$OUT" | wc -l) files"
