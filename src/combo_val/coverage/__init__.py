"""Multi-Target Coverage — set-cover combination optimizer.

Theory: Palmer & Sorger Cell 2017 (PMID 29245013) Independent Drug Action.
A combination's clinical benefit comes from covering multiple patient-
specific vulnerability targets, not from molecular drug-drug synergy.

Pipeline:
  patient (mut + RNA + clin) → patient_target_inference → active_targets {target_id: weight}
  drug pool → drug_target_matrix → coverage matrix [n_drugs × n_targets]
  active_targets + coverage_matrix + constraints → set_cover_solver → ranked combos

Modules:
  taxonomy.py             — load target_taxonomy_v2.yaml + drug coverage
  patient_targets.py      — infer active targets for a patient
  set_cover.py            — greedy + local-search optimizer (Bliss-IDA aggregation)
  report.py               — render top-N combos as markdown
"""
