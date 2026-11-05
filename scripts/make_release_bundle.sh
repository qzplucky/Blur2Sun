#!/usr/bin/env bash
# 打包 GitHub Release 权重资产（不在仓库内产生大文件，输出到仓库外目录）。
#
# 用法:
#   bash scripts/make_release_bundle.sh [SRC_OUTPUTS] [OUT_DIR]
# 默认 SRC=/home/zth24029/project/Blur2Sun_v5/outputs
# 默认 OUT=/home/zth24029/project/blur2sun_release
#
# 产物:
#   blur2sun_v5_weights_production.zip   7 波段生产权重（论文所有主表口径）
#   blur2sun_v5_weights_ablation.zip     消融/实验变体权重
#   SHA256SUMS.txt
set -euo pipefail

SRC=${1:-/home/zth24029/project/Blur2Sun_v5/outputs}
OUT=${2:-/home/zth24029/project/blur2sun_release}

rm -rf "$OUT"
mkdir -p "$OUT/staging_prod" "$OUT/staging_abl"

# ---- 生产权重: band_XX/map_full_12s_v5/models/best_model.pth（7 波段）----
for b in 94 131 171 193 211 304 335; do
  mkdir -p "$OUT/staging_prod/band_$b/map_full_12s_v5/models"
  cp "$SRC/band_$b/map_full_12s_v5/models/best_model.pth" \
     "$OUT/staging_prod/band_$b/map_full_12s_v5/models/"
done

cat > "$OUT/staging_prod/MANIFEST.txt" <<'EOF'
Blur2Sun v5 production weights (7 AIA EUV passbands)
====================================================
Layout: band_XX/map_full_12s_v5/models/best_model.pth   (XX = 094/131/171/193/211/304/335)

Usage:
  unzip blur2sun_v5_weights_production.zip -d outputs
  export BLUR2SUN_MODELS=$(pwd)/outputs
  python3 demo/run_demo.py

These checkpoints reproduce all main-text results (lunar transit, noise
tables, Hi-C FWHM, flare DEM, quiet-Sun DEM, 7x4 qualitative grids) with the
native-intensity evaluation scripts shipped in the repository. At inference NO
input offset is applied to standard-exposure frames. For count-depleted
short-exposure flare frames only, a hand-set per-channel calibration constant
(2/2/85/4/85/4/4 DN for 94/131/171/193/211/304/335 A) is added at the input
and subtracted at the output; see Section 4.4 of the paper.
EOF

# ---- 消融/变体权重 ----
for rel in \
  band_94/map_full_12s_v5_ab_no_prior \
  band_171/map_full_12s_v5_ab_no_prior \
  band_171/map_full_12s_v5_ab_fixed_k \
  band_171/map_full_12s_v5_priorstrong \
  band_171/map_full_12s_v5_1 ; do
  mkdir -p "$OUT/staging_abl/$rel/models"
  cp "$SRC/$rel/models/best_model.pth" "$OUT/staging_abl/$rel/models/"
done

cat > "$OUT/staging_abl/MANIFEST.txt" <<'EOF'
Blur2Sun v5 ablation / variant weights
======================================
map_full_12s_v5_ab_no_prior   : learned prior removed (ablation "No-Prior")
map_full_12s_v5_ab_fixed_k    : step size k frozen at 1.0 (ablation "Fixed-k")
map_full_12s_v5_priorstrong   : exploratory stronger-prior variant (not in the paper)
map_full_12s_v5_1             : intermediate v5 training snapshot (not in the paper)

Layout identical to the production bundle. Paper ablation tables use the
channels named in the manuscript; the two exploratory 171 A variants are
provided for completeness only.
EOF

# ---- 压缩（.pth 压缩收益很小，用快速档）----
( cd "$OUT/staging_prod" && zip -qr -1 "$OUT/blur2sun_v5_weights_production.zip" . )
( cd "$OUT/staging_abl"  && zip -qr -1 "$OUT/blur2sun_v5_weights_ablation.zip" . )

# ---- 校验和 ----
( cd "$OUT" && sha256sum blur2sun_v5_weights_production.zip blur2sun_v5_weights_ablation.zip > SHA256SUMS.txt )

echo "=== 产物 ==="
ls -lh "$OUT"/*.zip "$OUT/SHA256SUMS.txt"
