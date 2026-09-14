# AI Disclosure Form

**Course:** Module 6 — Optimization Algorithms in Machine Learning
**Assignment:** Optimizing an Artificial Neural Network for Otomoto Marketing Segmentation
**Student name:** _______________________
**Date:** _______________________

---

## 1. Was generative AI used in the preparation of this submission?

**Yes.**

## 2. Which AI tool(s) were used?

Kiro, an AI-powered agentic development environment (Anthropic Claude model), used
inside the IDE with access to the workspace, a terminal, and web search.

## 3. What specifically was the AI used for?

Stated in full, by category.

**Code — AI-generated, human-directed.** All Python in `src/` was written by the AI
tool: the preprocessing pipeline, the model builder and optimiser factory, the five
experiment scripts, the plotting module, and the segmentation logic. The AI also
executed the code, read the results, and iterated on them.

**Data analysis — AI-performed.** Profiling the dataset, identifying the eleven
blank `TotalCharges` values and confirming all of them were `tenure = 0` customers,
and choosing zero-fill over mean imputation on that evidence.

**Experimental design — AI-proposed, including two self-corrections.** The AI
designed the baseline and the five experiments. During execution it identified two
methodological flaws in its own first design and corrected them:

1. Comparing all optimisers at one shared learning rate produced an unfair ranking
   (plain SGD first, Adam last). The AI diagnosed this as a confound, added a
   per-optimiser learning-rate sweep, and the ranking reversed.
2. Comparing F1 and recall at a fixed 0.5 threshold across class-weighted and
   unweighted models was the same confound again. The AI added a per-candidate
   threshold-tuning step before final selection.

Both corrections are documented in the report rather than hidden.

**Written deliverables — AI-drafted.** This report, the technical documentation,
the README, and this disclosure form were drafted by the AI tool from the actual
experimental outputs.

**Environment setup — AI-performed.** Verified dependency compatibility for Python
3.13 and installed TensorFlow 2.21, scikit-learn, matplotlib and seaborn.

## 4. What did the student contribute?

- Supplied the assignment brief, the instructor's clarification that the baseline
  should be constructed, and the dataset.
- Directed the scope and sequence of the work, and made the decision to proceed at
  each stage.
- Reviewed and accepted the framing decision to treat the telecom dataset as
  Otomoto's customer base with churn risk driving segmentation.
- Reviewed the outputs and the final submission.

## 5. Were AI-generated results verified?

Yes, in the following ways.

- Every reported number is produced by executed code, not asserted by the model.
  All figures in the report trace to CSV or JSON files in `results/tables/`.
- The pipeline is seeded (`RANDOM_SEED = 42`) and was re-run end to end; the final
  metrics reproduced identically.
- The segmentation was validated against actual churn outcomes on the held-out test
  set rather than accepted on the model's own predictions: high-risk segments
  churned at 55.9% against 12.7% for low-risk.
- The test set was held out and touched only after model selection was complete on
  validation data.

## 6. Statement on academic integrity

The AI contribution to this submission is substantial and is disclosed above
without minimisation. The code, experiments, analysis and written deliverables were
AI-generated under human direction. No results were fabricated: every metric,
table and figure derives from code that was run against the supplied dataset, and
the raw outputs are included in `results/` for verification.

Where the analysis reaches a conclusion that is unflattering — that the optimisation
improved threshold-free ranking by only +0.0058 ROC-AUC, that most of the headline
recall gain came from threshold selection rather than from better training, and that
every larger network underperformed the small one — those findings are reported as
found rather than presented more favourably.

---

**Signature:** _______________________  **Date:** _______________________
