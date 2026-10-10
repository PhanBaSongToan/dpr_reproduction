# Team plan and evidence checklist

## Required evidence per member

- Assigned GitHub issue.
- Feature branch named after the issue.
- At least one implementation file.
- At least one test or validation command.
- Focused commits with descriptive messages.
- Pull Request linked to the issue.

## Integration order

1. Member 1 merges dataset format and preprocessing.
2. Member 2 adds BM25 baseline and hard-negative export.
3. Member 3 adds dual encoders and embedding generation.
4. Member 4 adds training, in-batch negatives and checkpoint saving.
5. Member 5 adds vector indexing and evaluation.
6. All members review the final reproduction run.

## Definition of done

- `pytest -q` passes.
- One command reproduces the final experiment.
- Results table includes BM25, DPR and paper reference values.
- Any reduced dataset or changed hyperparameter is documented.
- README explains setup, data preparation, training and evaluation.
