# Evaluation & Audit — PhoBERT vs XLM-R on UIT-ViQuAD 2.0

**State audited:** `main` @ `c4c2698`
**Deliverable set:** tested pipeline (`src/mrc/`, `scripts/`) · evidence store (`results/`) ·
report (`report/`) · deck (`slides/`) · ViMRC Console (`demo/`)

**How this audit was performed.** The methods and results chapters were read in full. Every headline
metric was recomputed from `data/raw/viquad2_validation.json` and `results/predictions_*.json` with an
**independently written scorer** that imports nothing from `src/mrc/`. Predictions were additionally
re-derived from the raw per-window scores in `results/windows_*.json`. Paired significance tests and
the article-level bootstrap were re-implemented from scratch. Colour contrast was computed from the
design tokens in `demo/console/theme.py` and `slides/main.tex`. The test suite was executed.

---

## Verdict

**The measurements are sound, independently reproducible, and the claims are calibrated to what the
evidence supports.** Reported EM/F1 reproduce to within 0.06 points, paired statistics reproduce to
within rounding, and the dev-selected no-answer threshold survives an oracle-threshold stress test.
The central finding — that PhoBERT-base leads XLM-R-base by ≈6 EM in total score, and that this lead
is **abstention calibration rather than reading ability** — is stated at exactly the strength the
data carries, and I reproduced both halves of it independently.

**There are no FATAL and no SERIOUS findings.** The open items are two accessibility defects in the
console and a short list of documentation and scope gaps. The limits that remain on what the study
can conclude — unequal tuning budget, three-epoch budget, two seeds — are named in the text rather
than left implicit.

**Confidence in this audit:** **High** for everything recomputed from shipped artifacts — metrics,
paired tests, threshold behaviour, run configurations, contrast values, test results. **Medium** for
the training-instability diagnosis: the evidence chain is internally consistent, but nothing was
retrained. **Not assessable:** whether the instability occurs on CUDA; how much the `pyvi` segmenter
costs PhoBERT relative to the segmenter used during its pretraining; the per-model gradeability
ceiling — the window files store only the best span per window, so it cannot be computed without a
re-run.

---

## What the solution is

Two pretrained encoders — PhoBERT-base-v2 (monolingual, word-level) and XLM-R-base (multilingual,
subword) — are fine-tuned through one shared harness on UIT-ViQuAD 2.0 and scored once on the full
3,814-question validation set, which is the public test set of VLSP 2021. Four research questions are
posed: word-boundary sensitivity (RQ1), length effects (RQ2), unanswerable-question handling (RQ3),
and whether Transformers beat a non-neural baseline (RQ4).

The contribution is diagnostic rather than a leaderboard entry, and its sharpest result is
methodological: the apparent ranking of two SQuAD-2.0-style systems is governed by the no-answer
threshold, so comparing them at a fixed τ measures calibration rather than competence. Each model
must be thresholded on held-out data before any comparison means anything.

Three properties hold across the whole deliverable set.

- **Traceability with teeth.** No result number is typed by hand. `make_report_numbers.py` generates
  LaTeX macros from `results/`, and a value with no source renders as a red `??` rather than a guess.
  The deck consumes the same generated macros, so deck and report cannot disagree numerically.
  `demo/console/data.py` reads `results/*.json` directly, hardcodes no result values, and renders a
  missing value as `—` ("thiếu phải thấy được"). One evidence store feeds all three surfaces.
- **The two strongest diagnostics need no GPU.** The word-boundary measure bounds RQ1's predicted
  effect before any model runs; the stress-test audit establishes that the project's own 250-question
  diagnostic set cannot measure what it claims to.
- **A pre-registered hypothesis is reported as refuted.** RQ1 predicted PhoBERT would be *worse* at
  word boundaries. It is not, and the text says so — then goes on to say PhoBERT is not better at
  reading either, which is the harder admission and the correct one.

Reporting your own dataset as unusable is the most creditable act in this repository.

---

## Claim ↔ evidence

| # | Claim as stated | Evidence | Holds? |
|---|---|---|---|
| 1 | PhoBERT > XLM-R by +5.82 / +6.00 EM at dev-tuned τ, both seeds | `thresholds.json`, `diagnosis_validation_tuned.json` | **Yes.** Recomputed +5.85 / +6.00. |
| 2 | That lead is calibration, not reading: matched-window comparison gives −0.08 EM on answerable, CI [−2.24, 2.05], p = 0.97; all of +5.85 sits in unanswerable (+19.38) | `tab_paired_matched.tex` | **Yes.** Recomputed −0.04 EM, CI [−2.04, +2.00], p = 1.00; unanswerable +19.38. Independent agreement. |
| 3 | At τ = 0 with seed 42 alone the comparison reads as a tie with mirror-image profiles, and that reading is wrong | `diagnosis_validation.json` | **Yes.** Reproduced (−1.13, p = 0.22). |
| 4 | PhoBERT seed 42's non-abstention is a reproducible training event | `probe_resume.json`: gradient-norm spike 12,880 at lr 2.2e-5, absent at 1e-5 on identical batches | **Yes** as a symptom→trigger link. Root cause explicitly left open. |
| 5 | Window size is not XLM-R's handicap | `xlmr_256` run at 256/96 | **Yes** — the strongest control in the study, and the basis of claim 2. |
| 6 | Word segmentation is not decisive: 1.24% of answers, ceiling 0.75 EM | `segmentation_validation.json`, model-free | **Yes.** The ceiling bounds the effect a priori. |
| 7 | The 250-question stress-test is not a usable instrument | `stress_test_audit.json`: 15/120 answerable items gradeable | **Yes**, and acted upon rather than published. |

---

## Independent verification of the metrics

A scorer written from the SQuAD 2.0 definition, importing nothing from the project:

| Run | Reported EM / F1 | Recomputed EM / F1 | Δ |
|---|---:|---:|---:|
| PhoBERT s42 @ τ = 0 | 51.36 / 63.10 | 51.34 / 63.10 | −0.02 / 0.00 |
| PhoBERT s42 @ τ dev | 58.31 / 68.54 | 58.29 / 68.53 | −0.02 / −0.01 |
| PhoBERT s13 @ τ dev | 59.26 / 68.49 | 59.23 / 68.49 | −0.03 / 0.00 |
| XLM-R s42 @ τ dev | 52.49 / 61.99 | 52.44 / 61.99 | −0.05 / 0.00 |
| XLM-R s13 @ τ dev | 53.25 / 62.68 | 53.23 / 62.68 | −0.02 / 0.00 |
| PhoBERT lr 1e-5 @ τ dev | 53.85 / 64.69 | 53.83 / 64.68 | −0.02 / −0.01 |

Residual EM gaps of 0.02–0.05 correspond to one or two questions out of 3,814 — a normalisation
detail, not an error. Gold counts verified independently: **3,814 questions / 2,653 answerable /
1,161 unanswerable / 19 articles**, matching every reported figure. Rebuilding predictions from raw
window scores reproduces each published figure to ≤0.06. **The reported numbers are measurements,
not claims.**

The matched comparison that carries claim 2 was re-derived end to end:

| slice | published | recomputed |
|---|---|---|
| all | +5.85, CI [3.74, 7.85], p < 0.001 | +5.87, CI [3.82, 7.82], p ≈ 1e-12 |
| answerable | **−0.08**, CI [−2.24, 2.05], p = 0.968 | **−0.04**, CI [−2.04, 2.00], p = 1.00 |
| unanswerable | +19.38, CI [16.71, 22.76], p < 0.001 | +19.38, CI [16.79, 22.65], p ≈ 3e-34 |

Resampling **articles rather than questions** is the correct unit — questions within an article share
a passage and an annotator — and the project uses it and reports the wider interval.

**Test suite:** 244 tests, all passing, no GPU required.

---

## Methodology

**Test-set hygiene — passes.** The dev split used for epoch and threshold selection is carved from
*train*, split by passage, with a runtime assertion that train, dev and validation share no passage.
Validation is scored once. The official hidden test set is refused rather than mis-scored: its 7,301
questions carry empty answers with `is_impossible = False`, and `assert_gradeable` blocks them. **No
leakage path was found.**

**Tuning budget — unequal, and declared.** PhoBERT is run at three learning rates and XLM-R at one.
§Phương pháp states this directly, calls it a budget that is *không cân bằng*, explains that the extra
runs exist to diagnose the instability, and instructs the reader that PhoBERT's total-score advantage
must be read under that condition. That is the correct handling: the asymmetry is real and cannot be
removed without more compute, so it is named where a reader will meet it. It remains a genuine limit
on the comparison, not a defect in the reporting.

**Convergence — under-trained, and declared.** Six of the seven runs carry `"still_improving": true`
from the project's own curve detector, and §Phương pháp states that every run was still improving at
its final epoch. Final training losses differ systematically by family: 0.82 for PhoBERT seed 13
against 1.08 for XLM-R seed 13 at 384/128 and 0.85 at 256/96. At a fixed three-epoch budget,
"PhoBERT is better" and "PhoBERT converges faster" are not separable — which is why claim 2 is scoped
to calibration and not extended into an architectural statement.

**Confound control.** The `xlmr_256` run isolates window configuration and is what makes the matched
comparison possible. Segmenter drift — `pyvi` against the segmenter PhoBERT was pretrained with —
remains declared but not controlled.

---

## Statistical validity

**Threshold selection is sound under stress.** Sweeping τ across validation and comparing the
dev-selected value against the validation-optimal one:

| Run | τ (dev) | F1 @ τ dev | oracle τ | F1 @ oracle | regret |
|---|---:|---:|---:|---:|---:|
| PhoBERT s42 | +7.86 | 68.60 | +6.50 | 68.97 | 0.37 |
| PhoBERT s13 | +1.03 | 68.46 | −0.75 | 69.27 | 0.81 |
| XLM-R s42 | −0.47 | 61.99 | −1.50 | 62.20 | 0.21 |
| XLM-R s13 | −0.62 | 62.68 | −0.75 | 62.88 | 0.21 |
| PhoBERT lr 1e-5 | −0.39 | 64.68 | −0.25 | 64.70 | 0.02 |
| XLM-R 256/96 | −1.30 | 63.36 | −0.75 | 63.48 | 0.12 |

Maximum regret is 0.81 F1. **The total-score gap is not an artifact of a fortunate threshold** — it
holds when both models are handed their oracle τ. Adding this check to the report would cost one
paragraph and close the last plausible objection to claim 1.

**Effect size against training noise.** Answerable-only EM at each run's tuned τ:

| Family | runs | mean | within-family spread |
|---|---|---:|---:|
| PhoBERT | 60.54 / 57.67 / 59.22 | 59.14 | 2.86 |
| XLM-R | 54.54 / 57.82 / 57.71 | 56.69 | 3.28 |

The difference of means on reading skill is +2.45 EM, **smaller than the run-to-run spread inside
either family**. This is an independent route to the same conclusion the matched comparison reaches:
there is no reading-ability difference this study can resolve, and the text does not assert one.

**Error decomposition matches the headline.** `diagnosis_validation_tuned.json` evaluates the
taxonomy, abstention precision/recall and slice breakdowns at each system's dev-tuned τ — the same
operating point the results tables report. PhoBERT s42 at τ dev abstains on 24.96% of questions with
precision 64.81% and recall 53.14%, consistent with its published threshold row.

---

## Reproducibility

**Strong on provenance.** Every `eval_*.json` carries commit, timestamp, device, platform and
`torch`/`transformers` versions. Dependencies are pinned with `==`. Per-question predictions ship for
every system, which is what made this audit possible — most published work cannot be audited this way.

**Gaps.** There is no `uv.lock` despite `uv` being the documented installer, and no `Makefile` or
single entry point: reproduction is roughly twelve manual steps plus two overnight queue scripts. MPS
is non-deterministic, so bit-exact reproduction is impossible even for the authors — correctly
declared. `results/README.md` annotates `misaligned_labels.json` with an open review task, and that
file is the sole evidence for RQ1's sub-claim that most boundary misses are segmenter defects rather
than genuine compound-word boundaries.

---

## Presentation layer

**Report.** LaTeX, all numbers injected from generated macros. The key box, README and §Kết luận agree
with the body: RQ1 is answered as "not worse at word boundaries, and not better at reading either",
with the matched-comparison interval quoted inline.

**Deck.** Beamer, 24 pages, consuming the same generated macros as the report. Typography is Be
Vietnam Pro with Source Code Pro for code; the palette is the Organic token set shared with the
console. Figures render on the deck's own canvas colour, so charts sit flush in the page.

**ViMRC Console.** A Streamlit application over the same evidence store, with five pages: extractive
QA, diagnostic matrix, error explorer, stress-test, and training/threshold views. Two design decisions
deserve credit:

- **The traceability invariant extends to the interface.** `data.py` reads `results/*.json` directly
  and hardcodes no result values; a missing value renders as `—`.
- **It refuses to fabricate.** `infer.py` uses the report's own inference class and the dev-selected
  τ, so an answer shown in the interface is the answer the results table scored. With no checkpoint
  present it raises an explicit error instead of displaying an invented answer.

### Accessibility audit — WCAG 2.1 AA

**Colour contrast: 21/21 pairs pass.**

| Surface | Element | Ratio | Required | Pass |
|---|---|---:|---:|---|
| Console | body text on canvas | 13.95 | 4.5 | ✅ |
| Console | lede / muted / kicker / label | 5.53 | 4.5 | ✅ |
| Console | muted on card | 6.02 | 4.5 | ✅ |
| Console | primary button text | 6.81 | 4.5 | ✅ |
| Console | tag: accent / ok / error / neutral | 9.46 / 6.46 / 6.81 / 8.12 | 4.5 | ✅ |
| Console | text on gold and prediction highlights | 11.34 / 10.97 | 4.5 | ✅ |
| Console | sidebar text on surface | 12.40 | 4.5 | ✅ |
| Deck | body text | 13.95 | 4.5 | ✅ |
| Deck | frame title on surface | 12.40 | 3.0 | ✅ |
| Deck | muted / source line | 5.53 | 4.5 | ✅ |
| Deck | block title | 9.46 | 4.5 | ✅ |
| Deck | PhoBERT / XLM-R emphasis | 3.03 / 3.14 | 3.0 | ✅ |

The console clears the normal-text threshold everywhere. The deck's two model-emphasis colours clear
only the large-text threshold, by 0.03 and 0.14 — adequate for projected slide text, but they are the
weakest pairs in the system.

**Other criteria.**

| Criterion | Finding |
|---|---|
| 1.1.1 Non-text content | No raster images in the console, so no missing alt text. The taxonomy bars are `<div>` segments carrying no accessible name or value — see A2. |
| 1.3.1 Info and relationships | The diagnostic matrix is a real `<table>` with `<thead>` and `<th>`. Nine section headings are `<div class='vm-h2'>` rather than `<h2>` — see A1. |
| 1.4.3 / 1.4.11 Contrast | Pass, as above. |
| 2.1.1 Keyboard | Navigation, system selection, example chips and threshold selection are Streamlit buttons and text inputs — natively focusable and operable. |
| 2.4.7 Focus visible | Explicit `:focus-visible` outlines for buttons and inputs, with `outline-offset`. |
| 3.3.2 Labels | Inputs carry visible widget labels styled through `[data-testid="stWidgetLabel"]`. |
| 4.1.2 Name, role, value | Buttons and inputs are native controls. The div-built bar charts expose no name or value — see A2. |
| Reduced motion | `prefers-reduced-motion` is honoured. |

---

## Threats to validity

- **Internal.** The causal account of instability → low abstention → apparent tie is well supported by
  the resume probes. The comparative account carries two named confounds — unequal hyperparameter
  budget and unequal convergence at a fixed epoch budget — which is why the conclusion is scoped to
  calibration.
- **External.** Wikipedia only, one dataset, base-size models only, Apple MPS only. Whether the
  instability occurs on CUDA is untested and declared as untested.
- **Construct.** "Reads better" is measured on the full answerable set with window configuration and
  run health held equal, which is the right construct; the both-answered slice, which conditions on a
  post-treatment variable and changes sign with run selection, is reported as unreliable rather than
  used as support. This is the correct call.
- **Statistical-conclusion.** Paired tests, confidence intervals and effect sizes are present and
  correct for question-sampling uncertainty. Training-run uncertainty rests on two seeds and is
  acknowledged as coarse; no claim is made that requires more precision than two seeds provide.

---

## Findings by severity

### No FATAL and no SERIOUS findings

The headline was attacked four ways — independent rescoring, reconstruction from raw window scores,
oracle-threshold substitution, and comparison at equal window configuration — and it held under all
four. The claims that the evidence cannot support are not made. This is stated plainly rather than
padded with manufactured objections.

### MAJOR

**A1 — Section headings are not headings (WCAG 1.3.1).** Nine `<div class='vm-h2'>` elements in
`demo/console/pages.py` carry heading styling without heading semantics, so screen-reader users get no
heading outline and cannot navigate the console by section. **Fix:** render them as `<h2>` and keep
the class.

**A2 — Bar charts expose no accessible name or value (WCAG 1.1.1, 4.1.2).** The taxonomy bars and
segment stacks are `<div>` elements; their quantities are conveyed only visually, and
`demo/console/pages.py` contains no `aria-*` or `role` attributes. **Fix:** add `role="img"` with an
`aria-label` summarising the distribution, or pair each chart with a visually hidden table of the same
numbers.

### MINOR

**M1 — README understates a segmentation-related data asymmetry.** The README states *"Dữ liệu sạch: 0
vi phạm trên 31.039 feature"* without noting that 1.66% of PhoBERT's training spans mismatch the gold
answer after normalisation against 0.50% for XLM-R. The report states both. The README is read first
and gives only the flattering half of a 3× asymmetry that runs against PhoBERT.

**M2 — `misaligned_labels.json` carries an unresolved review task,** is single-annotator with no
inter-annotator agreement, and underpins an RQ1 sub-claim.

**M3 — The per-model gradeability ceiling is never computed:** how many answerable questions have
their gold span outside *every* window at 256 tokens versus 384. `scan_features.py` reports 2,106 such
features but only at feature level. This is a per-model handicap that could differ between the two
window configurations and is currently unquantified. No GPU required.

**M4 — No lockfile and no single-command reproduction path.**

**M5 — The oracle-threshold robustness check is absent from the report.** The data to run it already
ships in `results/windows_*.json`; the result strengthens claim 1 at the cost of one paragraph.

**A3 — The deck's PhoBERT and XLM-R emphasis colours pass the large-text threshold by 0.03 and 0.14.**
Adequate for projection, but with no headroom; darker variants of the same hues would clear the
normal-text threshold.

---

## Bottom line

**Believe**
- All reported EM/F1, paired differences, confidence intervals and p-values. They reproduce.
- PhoBERT-base leads XLM-R-base by ≈6 EM in total score on this dataset with each model at its own
  dev-tuned threshold — robust to oracle thresholds and to equal window configuration.
- That the lead is abstention calibration, not reading ability. Two independent routes agree: the
  matched comparison (−0.04 to −0.08 EM, CI ±2) and the across-run spread (+2.45 EM against 2.9–3.3
  EM of training noise).
- That comparing SQuAD-2.0-style systems at a fixed threshold compares calibration rather than
  competence. This is the project's real methodological contribution.
- That the 250-question stress-test is unusable as an instrument, and the audit establishing it.
- That word segmentation is not the deciding factor here; the model-free ceiling argument bounds it.
- That the console shows the same answers the results table scored, and fails loudly when it cannot.

**Hold loosely**
- The instability mechanism. The symptom→trigger link is well evidenced; root cause, seed dependence
  and behaviour outside MPS are open, as the text states.
- The inference-latency advantage: real, but one device, single batch.

**Scope boundaries to respect when citing this work**
- The tuning budget is three runs for one family and one for the other. The total-score comparison is
  conditioned on that.
- Six of seven runs were still improving at their final epoch. No claim about either model at its best
  is available from this study.
- Training-run uncertainty rests on two seeds per family.

**What would extend the conclusions**
1. Train both families to convergence at five to six epochs.
2. Sweep XLM-R's learning rate over the same three values PhoBERT receives.
3. Three or more seeds per family at 256/96, reporting answerable-only EM with a seed-level interval.
4. Compute the per-model window-coverage ceiling — no GPU required, closes M3.
5. Promote the nine styled section headings to real `<h2>` elements and give the bar charts an
   accessible name — closes A1 and A2, both small markup changes.

---

## Recommendation

**Accept with minor revision.** The evidence base is sound and unusually auditable, and the claims are
stated at the strength the evidence carries — including the negative result that the advantage is not
reading ability, which is the harder and more valuable finding. The open items are two accessibility
defects in the console (A1, A2), both small markup changes, and a short list of documentation gaps
(M1–M5). The remaining limits on what the study can conclude are compute-bound and are already
declared in the text as scope boundaries rather than left for a reader to discover.
