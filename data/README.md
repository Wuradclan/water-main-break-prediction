# Data

See the root [README.md](../README.md) for the full project data contract, labeling strategy, and PU-bias discussion.

## Pinned files

| File | Description |
|------|-------------|
| `raw/Water_Main_Breaks.csv` | Kitchener–Waterloo water-main **break incidents** used for labeling |
| `SHA256SUMS` | Content hashes for reproducibility (logged into MLflow runs) |
| `processed/pipe_break_snapshots.csv` | Historical Snapshot / Windowing table (`break_within_horizon`) |

### Source

- Reference project: [js3lliott/water-main-break-prediction-KW](https://github.com/js3lliott/water-main-break-prediction-KW)
- Upstream portal: [Kitchener Water Main Breaks](https://open-kitchenergis.opendata.arcgis.com/datasets/KitchenerGIS::water-main-breaks/about)

### Features in scope

Breaks-only physical attributes: `ASSET_MATERIAL`, `ASSET_SIZE` (diameter mm), `ASSET_YEAR_INSTALLED`, plus time-aware features engineered at snapshot date `t` (`age_years`, `prior_break_count`, `years_since_last_break`).

**Deferred:** pipe length from the Water_Mains inventory (`Shape__Length`).

### Labeling (H = 5 years)

- Positive at `t = T − H` for each break time `T`
- Negatives at `t = T − H − k` for `k ∈ {1,2,3,4,5}` when `(t, t+H]` is break-free
- Label `0` ≠ permanently healthy (residual PU / selection bias)

### Regenerate processed snapshots

```bash
python -m src.labeling
# then refresh SHA256SUMS if you intentionally change the snapshot file
shasum -a 256 data/raw/Water_Main_Breaks.csv data/processed/pipe_break_snapshots.csv > data/SHA256SUMS
```

### Legacy

`Aiplane_BlueBook.csv` may still be present from the original aircraft project and is **not** used by the water-main pipeline.
