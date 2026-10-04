# RR-interval AF screening: does the risk guarantee hold for every patient?

Deep models that detect atrial fibrillation from beat-to-beat (RR) intervals
usually report one number for accuracy and one for F1 on a pooled test set.
This project asks a different question: once you wrap such a model in a
conformal risk guarantee, does that guarantee hold inside clinically
meaningful subgroups, or only on average?

Short answer from our runs: only on average. The same calibrated model that
keeps its pooled missed-AF rate near 6% misses about 55% of AF in windows
where the ventricular response is regular.

Everything here is RR-only. No raw waveform is used at inference time, which
is the same constraint a wrist wearable operates under.

---

## What the code does

1. **Extraction.** Pulls R-peak and rhythm annotations from four public
   long-term Holter databases, converts them to RR intervals, clips to
   0.2-3.0 s, and cuts non-overlapping 30-beat windows. A window is labelled
   AF when the majority of its beats fall inside an AF episode.
2. **Training.** A four-layer dilated temporal convolutional network on the
   raw RR sequence, plus a gradient-boosting model on eleven HRV summary
   features as an architecture control.
3. **Calibration.** Split conformal risk control on the AF class: pick the
   threshold on held-out calibration patients so the missed-AF rate is at
   most alpha = 0.05.
4. **Evaluation.** Report the pooled miss rate, then re-report it inside
   strata of the per-window RR coefficient of variation
   `CV = std(RR) / mean(RR)`. CV costs nothing to compute; it comes from the
   same 30 numbers the model already sees.
5. **Mitigation.** Mondrian (per-stratum) calibration: one threshold per CV
   stratum instead of one global threshold.

All splits are at the patient level and asserted in code. Every headline
number is the mean and standard deviation over five random patient splits.
Confidence intervals are bootstrapped over patients, not windows, because
windows inside one recording are strongly correlated.

---

## Data

| Cohort | Source | Windows | Patients | AF % |
|---|---|---|---|---|
| SHDB-AF | PhysioNet | 344,941 | 93 | 23.7 |
| LTAFDB | PhysioNet | 299,818 | 84 | 58.3 |
| AFDB | PhysioNet | 40,705 | 25 | 42.6 |
| IRIDIA-AF | Zenodo | 1,038,230 | 152 | 32.2 |
| **Pooled** | | **1,723,694** | **354** | **35.3** |

All four are public and are used exactly as distributed, with no
re-annotation.

Two details that silently corrupt everything if you get them wrong, and that
the notebooks handle explicitly:

- SHDB-AF has 98 annotated recordings but only 93 distinct subjects. Five
  subjects contributed two recordings each. Group by `Subject_ID` from
  `AdditionalData.csv`, not by recording id, or those subjects leak across
  train and test.
- IRIDIA-AF ships RR in milliseconds, and its AF episodes are indexed per
  day-file. Day-files have to be concatenated with offset tracking before the
  episode mask is applied. Patient id and record id are also different (167
  records, 152 patients).

IRIDIA-AF is a 10.5 GB archive. We do not download it. `remotezip` reads the
ZIP central directory over HTTP range requests and pulls only the ~45 MB of
RR files we need.

---

## Results

Pooled, five patient-disjoint splits, 354 patients.

**Aggregate metrics at threshold 0.5**

| Metric | Value |
|---|---|
| AUC | 0.988 +/- 0.003 |
| Accuracy | 96.25 +/- 0.24 |
| F1 | 94.80 +/- 0.53 |
| Precision | 94.22 +/- 1.93 |
| Recall | 95.44 +/- 1.22 |
| Specificity | 96.80 +/- 0.74 |

**Missed-AF rate by CV stratum, global conformal threshold, alpha = 0.05**

| CV stratum | AF windows | Patients | Missed AF | 95% CI |
|---|---|---|---|---|
| [0.000, 0.060) | 4,126 | 39 | 0.554 +/- 0.161 | [0.365, 0.741] |
| [0.060, 0.082) | 1,369 | 39 | 0.360 +/- 0.128 | [0.158, 0.624] |
| [0.082, 0.200) | 72,569 | 81 | 0.053 +/- 0.024 | [0.031, 0.084] |
| [0.200, inf) | 79,337 | 81 | 0.036 +/- 0.011 | [0.028, 0.047] |
| Pooled | | | 0.060 +/- 0.019 | |

The pooled rate sits close to the 0.05 target. The lowest stratum is about
15 times worse, and nothing in the aggregate table shows it.

**Why it happens.** It is not a ranking failure. Inside the low-CV stratum
the model still scores true AF above sinus rhythm. What collapses is the
absolute score: AF is rare in that regime relative to the 35.3% pooled
prevalence, so the scores sit far below the global threshold. In one split
the global threshold is 0.812 while the thresholds needed for valid coverage
inside each stratum are 0.035, 0.008, 0.934 and 0.793. Swapping 0.812 for
0.035 in the lowest stratum drops its miss rate from 0.711 to 0.368 with no
retraining at all.

**It is not an artifact.**

- Leave one cohort out, so the model never sees the test cohort during
  training or calibration: the gap appears in all four, with low/high CV miss
  ratios of 9.0x, 11.6x, 11.5x and 35.1x at AUC 0.953-0.995.
- Gradient boosting on HRV features behaves the same as the TCN (blind miss
  0.627 vs 0.609 at comparable AUC), so it is not a property of the function
  class.
- Sweeping the stratification cut over [0.05, 0.15] moves the numbers but not
  the direction: low-CV miss goes from 0.565 down to 0.191, ratios from 11.9x
  to 5.9x.

**Mondrian calibration helps but does not fix it.** Worst-case stratum miss
falls from 0.554 to 0.230, a 2.4x improvement, while the stratum-averaged
alert burden rises from 0.335 to 0.444. The worst stratum still misses close
to a quarter of AF. The residual standard deviation is 0.113 on a mean of
0.230, which is half its own size. With roughly 4,100 and 1,400 AF windows in
the two low-CV strata, the 5% quantile fitted inside them is simply noisy, so
we read the residual as an estimation problem rather than as a hard limit of
the RR representation.

**Cost of the guarantee.** Holding the pooled miss rate near 5% means
flagging 35.6 +/- 2.2% of every recording, against a pooled AF prevalence of
35.3%.

---

## Repository layout

```
notebooks/
  01_cohort_extraction_and_first_experiments.ipynb
  02_qrs_detector_and_patient_level_calibration.ipynb
  03_mondrian_calibration_and_iridia_pooling.ipynb
  04_pooled_four_cohort_analysis.ipynb
  05_final_leakfree_analysis_and_figures.ipynb
requirements.txt
LICENSE
```

The notebooks are in the order we ran them, and they are kept as a working
record rather than rewritten into a tidy single pass. Earlier notebooks
contain experiments we later abandoned or contradicted. Where that happened,
the comments say so.

- **01** Cohort extraction and windowing, first conformal threshold, the
  cross-cohort transfer check, and the first look at CV as a stratifier.
- **02** RR re-derived from a real QRS detector (XQRS) instead of gold
  annotations, beat-level detector sensitivity and PPV, and patient-level
  calibration.
- **03** Mondrian calibration, the IRIDIA-AF extractor, and the first pooled
  four-cohort run.
- **04** Full pooled suite with bootstrap CIs computed across all five seeds.
- **05** The run everything above is superseded by: SHDB grouped by subject,
  354 patients, plus the figure script.

**If you only read one, read 05.** Numbers in 01-04 come from smaller cohort
sets or from the run with the SHDB grouping bug, and are kept for history
only.

---

## Running it

The notebooks were written on Kaggle with a GPU and internet enabled, and
they write to `/kaggle/working`. To run elsewhere, change `OUT` at the top of
each notebook and keep the directory stable across cells, because later cells
reload the `.npz` files earlier cells write.

```bash
pip install -r requirements.txt
```

Rough order:

1. Run the rebuild cell in notebook 05 to produce
   `{shdb,ltafdb,afdb,iridia}_W30.npz`. About 25-30 minutes, needs internet.
2. Run the experiment suite in the same session. About 40 minutes on a GPU,
   much longer on CPU.
3. Run the figure cell last. It reuses variables left in memory by the suite,
   so it has to be the same kernel session.

No hyperparameter search was done. The point is the calibration behaviour of
a model that already performs well, not squeezing out another AUC point.

---

## Things we did not settle

- The calibration unit is the window, not the patient. Windows inside a
  recording are not exchangeable, which is the same reason the bootstrap is
  over patients. Our pooled miss of 0.060 sitting above the 0.05 target is
  consistent with that. A patient-level loss or a cluster-aware conformal
  construction is the right fix and is the first thing we would do next. The
  stratified result is a ratio between strata under one threshold, so it does
  not depend on this.
- R-peaks come from the supplied annotations. A single-cohort check on
  SHDB-AF (800 spans) gave beat-level detector sensitivity of 0.97 against
  the annotations with no difference between AF and non-AF spans (p = 0.39),
  and the stratified gap survived when RR came from the detector. That check
  was not repeated on the pooled data, so treat the numbers as measured under
  favourable detection conditions.
- Rhythm definitions differ across cohorts. We take AF strictly from the
  `(AFIB` annotation for the PhysioNet cohorts, while IRIDIA-AF may group
  atrial flutter with AF. Flutter also produces regular RR, which may be part
  of why IRIDIA-AF shows the largest effect.
- Alert burden is pooled, not per patient. Whether the alarm is uninformative
  for an individual low-burden patient needs per-patient burden measured
  against that patient's own AF burden, which we did not compute.
- We make no claim about raw ECG. An earlier notebook contains an oracle
  experiment on waveform; a later experiment contradicted its conclusion, and
  we did not resolve it.

---

## Team

Four of us, splitting data extraction, modelling, calibration and analysis:

- Lalitaditya Tickoo
- Md Samar Aazmi
- Ayushman Mayank
- Tushar Biswas

---

## License

MIT, see `LICENSE`. The databases are not redistributed here and keep their
own licences: SHDB-AF, LTAFDB and AFDB from PhysioNet, IRIDIA-AF from Zenodo.
