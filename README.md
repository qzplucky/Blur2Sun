### Pre-trained weights (Google Drive)
The pre-trained weights used in the paper are hosted on Google Drive:
[https://drive.google.com/drive/folders/1UUtPrxP9i2zw6udOLVLXMKcx9VN0khwV?usp=drive_link](https://drive.google.com/drive/folders/1UUtPrxP9i2zw6udOLVLXMKcx9VN0khwV?usp=drive_link)

| Asset | Size | Contents |
|---|---|---|
| `blur2sun_v5_weights_production.zip` | ~885 MB | 7 passbands, `band_XX/map_full_12s_v5/models/best_model.pth` — the checkpoints used for all main-text results |
| `blur2sun_v5_weights_ablation.zip` | ~628 MB | ablation / variant checkpoints (No-Prior, Fixed-k, exploratory 171 Å runs) |

SHA-256 checksums are in `SHA256SUMS.txt` alongside the assets.
After downloading, unzip the weights bundle into the `outputs` folder:
```bash
unzip blur2sun_v5_weights_production.zip -d outputs
export BLUR2SUN_MODELS=$(pwd)/outputs   # contains band_XX/map_full_12s_v5/models/best_model.pth
