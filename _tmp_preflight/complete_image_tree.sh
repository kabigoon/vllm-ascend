#!/usr/bin/env bash
# 把镜像里"fork 没有的 .py"（TBE 算子源码 + _version/_build_info）补进 fork，
# 再做一次**全量内容 diff**，确认 fork 的 vllm_ascend/ 与镜像一致 ⇒ 之后可以整树挂载。
# 只写 fork 的工作区（不 commit、不改镜像）；被 .git/info/exclude 忽略，git status 仍干净。
set -u
IMG=${IMG:-quay.io/ascend/vllm-ascend:v0.30.0-a3}
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SELF/.." && pwd)
echo "==> fork: $REPO/vllm_ascend"
docker run --rm -v "$REPO:/fork" "$IMG" bash -lc '
set -u
SRC=$(python3 -c "import importlib.util as u; s=u.find_spec(\"vllm_ascend\"); print(list(s.submodule_search_locations)[0])")
echo "SRC=$SRC"
n=0
while IFS= read -r f; do
  rel=${f#"$SRC"/}
  [ -e "/fork/vllm_ascend/$rel" ] && continue
  mkdir -p "/fork/vllm_ascend/$(dirname "$rel")"
  cp -a "$f" "/fork/vllm_ascend/$rel"
  n=$((n+1))
done < <(find "$SRC" -name "*.py" ! -path "*__pycache__*")
echo "added=$n"
echo "img_py=$(find "$SRC" -name "*.py" | wc -l)  fork_py=$(find /fork/vllm_ascend -name "*.py" | wc -l)"
echo
echo "=== full content diff (expect ONLY attention/dsa_v41.py, that is our probe) ==="
diff -rq --exclude=__pycache__ --exclude="*.so" --exclude="*.pyc" "$SRC" /fork/vllm_ascend
echo "=== end diff (rc=$?) ==="
'
echo
echo "==> 如果上面 diff 只列出 attention/dsa_v41.py，就可以用 FULLTREE=1 起服务："
echo "    PROBE=1 FULLTREE=1 bash _tmp_preflight/run_b.sh"
