-- ARDS analysis cohort: hierarchical MP, time-varying confounders, ECMO exclusion.
-- Builds on cohort_v5 (already has v4 corrected MP itemids + Charlson) and adds:
--   1. Hierarchical MP (Gattinoni VCV + Becher PCV surrogate)
--      Becher 2019 (Intensive Care Med); Chiumello 2020 (Crit Care)
--   2. Time-varying treatment confounders:
--        pressor_day  - day-level binary (NE/Vaso/Epi/Phenyl/Dobut active)
--        prone_day    - day-level binary (Position = 'Prone')
--        sofa_daily   - continuous, MIMIC-IV derived view
--   3. Static categorical: ards_cause (pneumonia/sepsis/aspiration/trauma/pancreatitis/other)
--   4. Outcome: mortality_28d (replaces mortality_30d to match study window)
--   5. Missingness indicator: bmi_missing
--   6. Exclusion: ECMO stays removed from cohort

WITH
daily_pressor AS (
  SELECT
    ie.stay_id,
    DATE(ie.starttime) AS d,
    1 AS pressor_day
  FROM `physionet-data.mimiciv_3_1_icu.inputevents` ie
  WHERE ie.itemid IN (221906, 222315, 221289, 221749, 221653, 221662)
    AND (ie.amount > 0 OR ie.rate > 0)
  GROUP BY ie.stay_id, DATE(ie.starttime)
),

daily_prone AS (
  SELECT
    ce.stay_id,
    DATE(ce.charttime) AS d,
    1 AS prone_day
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
  WHERE ce.itemid = 224093 AND LOWER(ce.value) LIKE '%prone%'
  GROUP BY ce.stay_id, DATE(ce.charttime)
),

daily_sofa AS (
  SELECT
    s.stay_id,
    DATE(s.starttime) AS d,
    AVG(s.sofa_24hours) AS sofa_daily
  FROM `physionet-data.mimiciv_3_1_derived.sofa` s
  GROUP BY s.stay_id, DATE(s.starttime)
),

ecmo_stays AS (
  SELECT DISTINCT ce.stay_id
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
  WHERE ce.itemid IN (224660, 229280, 229270, 229363, 229364, 229529, 229530)
),

-- Map ICD diagnoses to ARDS cause categories per Berlin definition.
-- Priority: pneumonia > sepsis > aspiration > trauma > pancreatitis > other.
ards_cause_per_hadm AS (
  SELECT
    di.hadm_id,
    CASE
      WHEN LOGICAL_OR(REGEXP_CONTAINS(di.icd_code,
        r'^(J12|J13|J14|J15|J16|J17|J18|J22|U071|480|481|482|483|484|485|486|487|488)')) THEN 'pneumonia'
      WHEN LOGICAL_OR(REGEXP_CONTAINS(di.icd_code,
        r'^(A40|A41|A42|R6520|R6521|99591|99592|0380|0381|0382|0388|0389)')) THEN 'sepsis'
      WHEN LOGICAL_OR(REGEXP_CONTAINS(di.icd_code, r'^(J69|5070|5078|99711)')) THEN 'aspiration'
      WHEN LOGICAL_OR(REGEXP_CONTAINS(di.icd_code, r'^(S27|S22|8610|8611|8612|8613)')) THEN 'trauma'
      WHEN LOGICAL_OR(REGEXP_CONTAINS(di.icd_code, r'^(K85|5770|5771|5772)')) THEN 'pancreatitis'
      ELSE 'other'
    END AS ards_cause
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` di
  GROUP BY di.hadm_id
),

-- Add Becher PCV surrogate MP for stay-days where Gattinoni is missing
-- because of unmeasured P_plat (PCV mode). Gattinoni is preserved separately
-- as `mp_gattinoni_only` for sensitivity analysis.
cohort_with_pcv AS (
  SELECT
    cb_in.*,
    CASE
      WHEN cb_in.mp_j_min IS NOT NULL THEN NULL                                       -- Gattinoni applicable
      WHEN cb_in.rr IS NULL OR cb_in.tidvol_obs IS NULL OR cb_in.ppeak IS NULL
           OR cb_in.peep IS NULL OR cb_in.pplat IS NOT NULL THEN NULL                    -- PCV requires P_plat null + P_peak present
      WHEN (0.098 * cb_in.rr * (cb_in.tidvol_obs / 1000.0) * 0.5 * (cb_in.ppeak + cb_in.peep))
            BETWEEN 0.1 AND 80
        THEN 0.098 * cb_in.rr * (cb_in.tidvol_obs / 1000.0) * 0.5 * (cb_in.ppeak + cb_in.peep)
      ELSE NULL
    END AS mp_pcv_surrogate
  FROM `pivotal-nebula-458401-j1.ards_v4.cohort_v5` cb_in
)

SELECT
  cb.* EXCEPT(mp_j_min, mp_pcv_surrogate),
  cb.mp_j_min                                  AS mp_gattinoni_only,
  cb.mp_pcv_surrogate                          AS mp_pcv_surrogate,
  COALESCE(cb.mp_j_min, cb.mp_pcv_surrogate)  AS mp_j_min,                       -- PRIMARY hierarchical
  COALESCE(dp.pressor_day, 0)                   AS pressor_day,
  COALESCE(dpr.prone_day, 0)                    AS prone_day,
  ds.sofa_daily,
  COALESCE(ac.ards_cause, 'other')              AS ards_cause,
  -- 28-day cumulative mortality (matches study endpoint, replaces mortality_30d)
  CASE
    WHEN cb.dod IS NOT NULL
     AND DATE_DIFF(DATE(cb.dod), cb.first_obs_date, DAY) <= 28
    THEN 1 ELSE 0
  END AS mortality_28d,
  -- BMI missingness indicator (informs imputation sensitivity)
  CASE WHEN cb.bmi_imputed IS NULL THEN 1 ELSE 0 END AS bmi_missing
FROM cohort_with_pcv cb
LEFT JOIN daily_pressor      dp  ON dp.stay_id  = cb.stay_id AND dp.d  = cb.day_date
LEFT JOIN daily_prone        dpr ON dpr.stay_id = cb.stay_id AND dpr.d = cb.day_date
LEFT JOIN daily_sofa         ds  ON ds.stay_id  = cb.stay_id AND ds.d  = cb.day_date
LEFT JOIN ards_cause_per_hadm ac ON ac.hadm_id  = cb.hadm_id
WHERE cb.stay_id NOT IN (SELECT stay_id FROM ecmo_stays)
ORDER BY cb.stay_id, cb.day_num
