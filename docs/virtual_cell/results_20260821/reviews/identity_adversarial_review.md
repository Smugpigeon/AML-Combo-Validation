# Virtual-cell gate review: cell_identity

Research use only. This review does not authorize treatment selection.

## Strongest opposing case
ScType, CopyKAT and SCEVAN all derive evidence from the same scRNA-seq matrix. Agreement can reproduce a published annotation workflow but cannot establish genotype-linked malignant identity.

## Omitted or weak facts
- No same-cell DNA barcode or genotype-to-cell linkage is available.
- No CITE-seq or flow label is paired to each sequenced cell.
- Reported blast percentages are patient-level priors, not cell labels.

## Optimistic assumptions
- T-cell references are uncontaminated and representative.
- RNA-derived copy-number calls remain stable in low-CNA AML clones.
- Major virtual states are not dominated by doublets or stressed normal cells.

## Irreversible or opportunity cost
Training a response model on incorrectly labelled cells would propagate identity error into every downstream drug and combination score.

## Worst consequence
A normal or reactive state could be presented as a leukemic vulnerability.

## Strongest supporting evidence
The pinned published RNA ensemble was evaluated on 11652 cells across 3 selected patients.

## Neutral verdict
Stop before drug-response validation because identity reproduction failed.

## Largest unknown
Whether RNA-ensemble malignant calls agree with an orthogonal, same-cell genomic or immunophenotypic label.

## Evidence that would reverse the verdict
Prospectively barcoded scDNA+scRNA, genotype-linked TARGET-seq, or cell-matched CITE/flow evidence with predeclared concordance thresholds.
