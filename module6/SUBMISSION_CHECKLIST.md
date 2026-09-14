# Submission Checklist

Maps every requirement in the assignment brief to where it is satisfied. Verify
before submitting.

---

## Assignment Instructions

| Step | Requirement | Where satisfied | Status |
|---|---|---|---|
| **1** | Review the current ANN model used for marketing segmentation. Assess its strengths and weaknesses. | `REPORT.md` §2 — full spec, measured performance, then **§2.2 strengths (4)** and **§2.3 weaknesses (7)**. `DOCUMENTATION.md` §1 adds layer-level spec and a weakness-to-remedy mapping table. | Done |
| **2** | Choose appropriate optimization algorithms. Justify selection based on the specific needs of Otomoto marketing segmentation. | `REPORT.md` §3 — table of 7 algorithms, each justified against a *named weakness* from §2.3, not general popularity. §3.1 justifies the metric choices from the 26.54% class imbalance. AdaGrad's rationale (sparse one-hot contract features) is later confirmed by permutation importance in §8.2. | Done |
| **3** | Implement the selected algorithms. Document the changes and the reasons behind them. | `REPORT.md` §4 — five experiments (A–E), each isolating one factor, with results tables. `DOCUMENTATION.md` §3 gives implementation detail and §4.2 gives a baseline-vs-optimized change table with a rationale column for every row, **including the three deliberately unchanged**. | Done |
| **4** | Assess the performance of the optimized model for **at least three** optimization algorithms. | `REPORT.md` §5.5 — **all 7 algorithms** evaluated on the held-out test set with the architecture frozen, each at its tuned learning rate and threshold. `DOCUMENTATION.md` §4.4. Code: `src/exp_final_evaluation.py`. Data: `results/tables/step4_optimizer_test_evaluation.csv`. Figures 24–25. | Done — exceeds (7 vs 3) |
| **5** | Prepare a detailed report explaining process, results, and recommendations for future improvements. | `REPORT.md` — 11 sections. Recommendations in **§9 (8 items, prioritized)**. Limitations §10. | Done |
| **6** | Use the data provided. | `data/teleconnect.csv` (7,043 × 21), copied unmodified from the supplied file. Pipeline: `src/data_prep.py`. Handling of the 3 data defects documented in `REPORT.md` §1.1. | Done |
| **7** | Submit report, optimized model, and relevant documentation. | See section A below. | Done |

### Cross-cutting requirements

| Requirement | Where satisfied | Status |
|---|---|---|
| Apply critical thinking throughout | `REPORT.md` §6 — two confounds found in **my own** experimental design, diagnosed and corrected: shared-learning-rate ranking (§6.1) and fixed-threshold recall comparison (§6.2). Also §5.4 honest gain attribution, §5.5 multiple-comparison caveat, §4.5 negative result on capacity. | Done |
| Use effective visual design | 25 figures in `results/figures/`, indexed in `REPORT.md` Appendix A. Consistent styling, learning curves for every experiment, before/after bars, ROC overlay, threshold sweep, segment scatter. All tables formatted in-document. | Done |
| Tell a compelling story about the optimization journey | `REPORT.md` runs baseline → diagnosis → algorithm selection → experiments → two self-corrections → final model → business segmentation → recommendations. The narrative turn is that two confident early conclusions turned out to be artefacts. | Done |

---

## A) Submission Contents

| # | Required item | File(s) | Status |
|---|---|---|---|
| 1 | Optimized ANN model code file or GitHub link | `src/` — 11 Python modules. Trained artefacts: `models/optimised.keras` (optimized), `models/baseline.keras` (original). Entry point: `src/run_all.py`. | Done |
| 2 | Report explaining process, optimization algorithms used, and performance comparison results | `REPORT.md` | Done |
| 3 | Documentation outlining the original model, chosen algorithms, and their impact on segmentation results | `DOCUMENTATION.md` — §1 original model, §3 chosen algorithms, **§6 impact on segmentation** (incl. §6.4 before/after segmentation quality). `README.md` for orientation. | Done |
| 4 | Performance evaluation metrics (accuracy, precision, recall, F1, loss rate) before and after optimization | All five plus ROC-AUC and PR-AUC. `REPORT.md` §5.2 (headline) and §5.5 (all 7 optimizers). Raw: `results/tables/final_baseline_vs_optimised.csv`, `step4_optimizer_test_evaluation.csv`. | Done |
| 5 | Properly formatted references following NXU citation style | `REPORT.md` References — **20 sources in APA 7th edition** (the style NXU uses), alphabetized, with in-text citations throughout. | Done |
| **B** | Completed AI Disclosure Form | `AI_DISCLOSURE.md` | **Needs your name, date and signature** |

---

## Before You Submit — Action Items

These are the only things I could not complete for you.

1. **`REPORT.md`** — fill the title block: student name, course code, date.
2. **`AI_DISCLOSURE.md`** — fill in name, date, signature. **Read section 3 and 4
   carefully and correct anything that does not match your recollection.** The form
   discloses that AI generated the code, experiments, analysis and written
   deliverables under your direction. It is written to be accurate, not flattering.
   You are accountable for its accuracy, so review it properly rather than signing
   it as-is.
3. **Confirm the required format.** These deliverables are Markdown. If your course
   requires `.docx` or `.pdf`, convert them — Markdown pastes cleanly into Word, but
   check that the tables survive the paste.
4. **Check whether figures must be embedded.** Figures are referenced by filename,
   not inlined. If the rubric expects them inside the report body, insert the PNGs
   from `results/figures/` at the points named in Appendix A.
5. **Confirm the citation style.** I verified NXU uses APA 7th edition and formatted
   accordingly. If your syllabus specifies otherwise, the reference list needs
   reformatting.
6. **Publish to GitHub** from your other computer — see the next section.

---

## Publishing to GitHub

Submission criterion A1 accepts a code file **or** a GitHub link. This folder is
ready to push as-is: `.gitignore` and `requirements.txt` are in place, total size is
3.83 MB, and no file approaches GitHub's 50 MB warning threshold.

### Decide public or private first

**Recommendation: make it private, then add your instructor as a collaborator.**

A public repository containing a complete, working solution to a live assignment is
findable by classmates. If someone copies it, you may be drawn into an academic
integrity investigation over work that is genuinely yours. Private plus a
collaborator invite gives the marker full access with none of that exposure.

If the course explicitly requires a public link, that overrides this — but check
before choosing.

### Copy to the other computer

Copy the whole `module6` folder. Everything needed is inside it. Two things are
excluded from git by `.gitignore` and are not needed in the repo:

- `teleconnect (1).csv` at the root — a duplicate of `data/teleconnect.csv`, which is
  the copy the pipeline actually reads
- `assignmnt qustion` — the course-supplied brief. Excluded because course material
  is not yours to redistribute, and publishing it beside the full solution invites
  exactly the plagiarism problem described above

Keep both locally; just don't commit them.

### Commands on the other computer

Run these from inside the copied `module6` folder. Replace `USERNAME` and
`REPO-NAME`.

```bash
git init
git add .
git status                    # confirm the two ignored files are absent
git commit -m "Module 6: ANN optimization for Otomoto marketing segmentation"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO-NAME.git
git push -u origin main
```

`git status` before committing is the step worth not skipping — it is where you
confirm `.gitignore` is doing its job.

If you have the GitHub CLI installed, it will create the repository for you:

```bash
gh repo create REPO-NAME --private --source=. --remote=origin --push
```

### After pushing

1. Open the repository in a browser and confirm `README.md` renders. It is the
   landing page and links to the other documents.
2. Confirm the tables in the Markdown files render correctly — GitHub renders
   Markdown tables natively, so they should be fine.
3. Confirm `results/figures/` shows all 25 PNGs.
4. If private, invite your instructor: **Settings → Collaborators → Add people**.
5. Put the repository URL in your report or submission form.

### Reproducibility on a fresh machine

Anyone cloning the repository can rebuild every result:

```bash
python -m pip install -r requirements.txt
python src/run_all.py
```

`RANDOM_SEED = 42` is set in `src/config.py`, and the pipeline was verified to
reproduce identical metrics across runs. Note the pinned versions assume Python
3.13; `requirements.txt` documents relaxed minimums if the other machine differs.

---

## Honest Notes on Scope

Three things a marker may probe, stated here so they are not a surprise.

**The dataset is telecom, not automotive.** `teleconnect.csv` is a telecom churn
dataset. It is framed as Otomoto's customer base with churn risk driving
segmentation, which is disclosed in `REPORT.md` §1.1. The method transfers; the
specific segment boundaries would not.

**The optimization gain on ranking quality is small.** ROC-AUC improved +0.0058.
The headline +18.9% recall gain comes mostly from threshold selection, which is a
deployment choice rather than better training. `REPORT.md` §5.4 states this
explicitly rather than letting the larger number stand unqualified. This is the
honest finding and it is defensible; presenting +18.9% as an optimizer achievement
would not be.

**Adam was selected on validation but ranks 6th of 7 on test.** Disclosed in
`REPORT.md` §5.5 with the reasoning: the choice was pre-registered on validation, so
the ranking is an unbiased check, and the 0.0199 F1 spread across seven optimizers on
a single 1,409-row split is within noise. This is why k-fold cross-validation is
recommendation #1 in §9.

---

## File Inventory

```
module6/
├── REPORT.md                    main report (submission item 2)
├── DOCUMENTATION.md             technical documentation (item 3)
├── AI_DISCLOSURE.md             AI disclosure form (item B) - NEEDS SIGNATURE
├── SUBMISSION_CHECKLIST.md      this file
├── README.md                    project orientation
├── requirements.txt             pinned dependencies
├── .gitignore                   excludes cache, logs, duplicate CSV, brief
├── data/teleconnect.csv         supplied dataset (item 6)
├── src/                         11 modules (item 1)
│   ├── config.py                paths, seed, column groupings
│   ├── data_prep.py             cleaning, encoding, splits, scaling
│   ├── modeling.py              model builder, optimizer factory, metrics
│   ├── plots.py                 figure generation
│   ├── baseline.py              Step 1
│   ├── exp_optimizers.py        Step 3 - Experiments A, B
│   ├── exp_tuning.py            Step 3 - Experiments C, D, E
│   ├── final_model.py           Step 4
│   ├── exp_final_evaluation.py  Step 4b - all optimizers on test
│   ├── segmentation.py          Step 5
│   └── run_all.py               runs everything in order
├── models/
│   ├── baseline.keras           original model
│   └── optimised.keras          optimized model (item 1)
└── results/
    ├── tables/                  20 CSV/JSON files (item 4)
    └── figures/                 25 PNG figures
```
