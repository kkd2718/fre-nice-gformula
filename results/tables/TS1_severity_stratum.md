# 표 S1. Berlin 중증도 stratum별 dose-response (M3)

_Per-stratum forward simulation. Marginal HR vs reference bin 7. P(HR > 1) is the Bayesian posterior probability for the primary bin 12 contrast._

| Stratum | Fit N (subjects) / Cohort stays | bin 9 HR | bin 12 HR (95% CrI) | bin 12 P(HR>1) | bin 16 HR | bin 19 HR |
|---|---|---|---|---|---|---|
| Mild (PF 200–300) | 7,019 / 7,373 | 1.21 | **1.53** (1.13–1.97) | 1.000 | 2.52 | 5.89 |
| Moderate (PF 100–200) | 6,727 / 7,329 | 1.29 | **1.84** (1.45–2.30) | 1.000 | 2.68 | 4.06 |
| Severe (PF ≤ 100) | 2,916 / 3,086 | 1.12 | **1.42** (1.11–1.88) | 1.000 | 1.83 | 2.38 |
| **Overall** | 15,549 fit / 17,788 cohort stays | 1.22 | **1.72** (1.48–1.99) | 1.000 | 2.63 | 4.40 |

Ceiling effect: severe stratum shows weakest marginal HR at all bins — baseline mortality already high.

**Note.** Overall fit N = 15,549 (full cohort M3 random-effect group count); stratum fit N sum = mild 7,019 + moderate 6,727 + severe 2,916 = 16,662, with 1,113 overlap from patients whose multiple ICU stays were classified into different severity strata.

_P(HR > 1) is the Bayesian credible-evidence analog to a frequentist p-value (fraction of MCMC posterior draws with HR > 1.0 at bin 12). Based on 200 forward-simulation subsamples per stratum._