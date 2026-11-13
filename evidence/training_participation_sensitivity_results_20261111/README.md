# Learning-rate sensitivity for DBpedia uniform initialization

The controlled deployment protocol required every trained row to change its FP32
bytes. The three DBpedia uniform-initialization runs at learning rate 10 left
914–1,207 of 34,050,161 rows bitwise unchanged and failed that gate, so the
primary selection for that family used learning rate 3 (test accuracy
94.83–94.87%). Zero initialization at learning rate 10 passed and reached 97.67%.

This audit, registered before it was run
([protocol](../training_participation_sensitivity_20261111.json)), asks whether
the gate distorted the learning-rate choice. It does not replace the primary
results.

1. Participation: every nonpadding row occurs in the stored training stream at
   least once per epoch over all ten epochs ([`participation.json`](participation.json)).
2. All nine uniform runs (learning rates 1, 3, 10 × seeds 3, 7, 11) were rescored
   on the 56,000 calibration documents. Learning rate 10 has the lowest mean
   calibration NLL and was frozen before any test document was read
   ([`selection.json`](selection.json)).
3. The three selected runs were scored on the 70,000 official test documents.

| Learning rate | Calibration accuracy | Test accuracy |
| ---: | ---: | ---: |
| 1 | 81.33–81.76% | – |
| 3 | 94.79–94.83% | – |
| 10 | 97.62–97.63% | 97.67–97.68% |

At matched learning rate, uniform and zero initialization reach the same test
accuracy (97.67%). The 2.8-point gap in the primary selection comes from the
every-row gate, not from the initialization.

Run `python evidence/training_participation_sensitivity_results_20261111/verify.py`
(needs NumPy) to recompute accuracy and NLL from the saved logits and check the
selection rule. `quality/` holds the per-run reports, logs and logits.
