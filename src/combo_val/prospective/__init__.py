"""Prospective validation infrastructure (v0.5).

Per Path B (post-v0.4 reviewer dialogue): the kit's Layer-3 ML combo
prediction has internal Pearson r ≈ 0.05 vs clinical CR. Before deciding
whether to keep / pivot / kill Layer-3, we run a prospective validation
study: at each enrolled patient, lock the kit's prediction immutably,
follow up on actual treatment + outcome at 6mo / 12mo, and at N=200 with
12-month follow-up run the formal Pearson + survival analysis.

Modules:
  - registry.py            : participating-site / PI / IRB enrollment
  - locked_prediction.py   : immutable hash-chained prediction store
  - outcome_capture.py     : 6/12/18mo follow-up form schema + reminders
  - analysis_trigger.py    : auto-runs external_validation when N reached
"""
