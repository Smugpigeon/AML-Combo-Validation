# Virtual-cell gate review: patient_specific_combination

Research use only. This review does not authorize treatment selection.

## Strongest opposing case
Single-agent direction does not identify interaction, dose ratio, order, toxicity or causal synergy. A combination model can learn marginal potency and still fail on true interaction.

## Omitted or weak facts
- No prospective patient-level pair or triplet holdout is yet available.
- Ex vivo synergy is not equivalent to clinical response or tolerability.
- The current public cohort is too small for a clinical claim.

## Optimistic assumptions
- Single-drug embeddings transfer to combinations.
- Observed ex vivo interaction generalizes across dose and scheduling.
- Patient-specific gains exceed a strong drug/pair-mean baseline.

## Irreversible or opportunity cost
Prematurely naming a best combination can bias experimental allocation and clinical review toward an under-validated hypothesis.

## Worst consequence
An ineffective or toxic combination could be over-prioritized because its components were individually active.

## Strongest supporting evidence
No supporting unlock; active blockers: ['retrospective cell-identity replication gate did not pass', 'real single-drug viability-direction gate did not pass', 'combination data-readiness gate did not pass or was not supplied', 'held-out patient/pair validation has not beaten strong baselines'].

## Neutral verdict
Do not train or publish patient-specific combination predictions yet.

## Largest unknown
Out-of-patient generalization of interaction residuals beyond the stronger single-agent and pair-mean baselines.

## Evidence that would reverse the verdict
A grouped-by-patient external challenge showing calibrated interaction gains, followed by prospective organoid and safety validation.
