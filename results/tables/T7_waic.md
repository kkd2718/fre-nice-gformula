# 표 7. WAIC 모형 비교 (cluster-level + paired SE_diff)

_Cluster-level WAIC at the unique-subject level (N=15,549 random-effect clusters matching the hierarchical group structure). Paired SE_diff per Vehtari 2017 §4.2. Sign convention: ΔWAIC = (comparator WAIC) − (M3 WAIC), so positive values mean M3 fits better; abstract uses the opposite convention (M3 − comparator, negative for M3 advantage); absolute magnitudes are identical._

| Spec | WAIC | SE_obs | ΔWAIC vs M3 | SE_diff | ratio | ρ(elpd) |
|---|---|---|---|---|---|---|
| **M3 (primary)** | **16200** | 288 | 0 (ref) | — | — | — |
| M2 | 16837 | 294 | +636 | 32.0 | **19.9×** | 0.9941 |
| M1 | 17119 | 297 | +919 | 42.5 | **21.6×** | 0.9899 |
| M4 | 17268 | 297 | +1068 | 46.8 | **22.8×** | 0.9877 |

All comparisons exceed 10×SE_diff on cluster-level predictive density; M3 is preferred under this metric. Caveat: ~12.3% of clusters have p_waic_i > 0.4 (Vehtari 2017 §4.2 threshold), so PSIS-LOO or K-fold cross-validation is recommended as a future-work robustness check. Absolute WAIC magnitudes are a ranking auxiliary indicator, not definitive causal-contrast preference.

**Spline rank sensitivity (rank 4 vs rank 5 vs rank 6 NCS basis):**

| Spline rank | bin 12 conditional HR (95% CrI) |
|---|---|
| 4 | 1.393 (1.054–1.813) |
| **5 (primary, M3)** | 1.394 (1.040–1.853) |
| 6 | 1.404 (1.038–1.880) |