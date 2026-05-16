# 표 2. 주 contrast — 4-spec 견고성 사다리 (조건부 HR, bin 12 vs ref bin 7)

_조건부 HR (sustained MP bin 12 ≈17.4 J/min vs reference bin 7 ≈9.3 J/min). Posterior 95% CrIs and posterior probability P(HR > 1 | data)._

| Specification | Y-RE | L-RE | Posterior samples | HR | 95% CrI | P(HR > 1) |
|---|---|---|---|---|---|---|
| M1 (no-RE) | none | none | 4,000 | 1.33 | (1.04–1.69) | 0.990 |
| M2 (scalar Y-RE) | constant (K=1) | none | 4,000 | 1.35 | (1.03–1.72) | 0.987 |
| **M3 (functional Y-RE, primary)** | **NCS (rank 5)** | none | 4,000 | **1.39** | **(1.04–1.85)** | **0.987** |
| M4 (joint scalar Y+L RE) | constant intercept (rank 1) | scalar intercept | 1,500 | 1.39 | (1.08–1.78) | 0.994 |

Primary contrast moves within 5% range (1.33 → 1.39) across 4 structurally distinct RE specifications. Empirical concordance, not mathematical proof of RE-independent identification.

_P(HR > 1) is the Bayesian credible-evidence analog to a frequentist p-value: the fraction of MCMC posterior draws with HR above 1.0. Values ≥ 0.975 correspond to 95% CrI excluding 1.0 (asterisk convention)._