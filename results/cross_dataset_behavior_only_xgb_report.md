# Cross-Dataset Generalization Report - `behavior_only` features

* Model: `xgb`  |  Feature mode: `behavior_only`  |  Datasets: cic-ids-2017, unsw-nb15, lab-captured
* Seeds: 7, 42, 99, 1337, 2024 (5 independent passes)
* Pairs: 9 ordered pairs (3 datasets, diagonal included)

## Headline

| Quantity | Value |
|---|---|
| Mean in-domain macro-F1 | 0.9549 |
| Mean cross-domain macro-F1 | 0.0381 |
| Mean F1 delta (in - cross) | **0.9168** |
| Cross-domain F1 retention | 4.0% |
| Mean test-label coverage | 87.5% |
| Cross-domain worse than in-domain | YES |
| Cross-domain pairs with a negative delta | 6 / 6 |

Values are means over seeds; `+/-` is the sample standard deviation across seeds.

## Full matrix

| Train | Test | In-domain | ACC | P | R | F1 | macro-F1 | In-domain ref | dF1 | Test cov. | Feat. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cic-ids-2017 | cic-ids-2017 | yes | 0.9587 | 0.9596 | 0.9587 | 0.9587 | 0.9587 +/-0.0071 | 0.9587 | ref | 100% | 16 |
| cic-ids-2017 | unsw-nb15 | no | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 +/-0.0000 | 0.9587 | +0.9587 | 83% | 16 |
| cic-ids-2017 | lab-captured | no | 0.1670 | 0.0279 | 0.1670 | 0.0478 | 0.0477 +/-0.0000 | 0.9587 | +0.9111 | 100% | 16 |
| unsw-nb15 | cic-ids-2017 | no | 0.2000 | 0.0400 | 0.2000 | 0.0667 | 0.0667 +/-0.0000 | 0.9320 | +0.8654 | 62% | 16 |
| unsw-nb15 | unsw-nb15 | yes | 0.9333 | 0.9351 | 0.9333 | 0.9320 | 0.9320 +/-0.0254 | 0.9320 | ref | 100% | 16 |
| unsw-nb15 | lab-captured | no | 0.1996 | 0.0398 | 0.1996 | 0.0664 | 0.0665 +/-0.0000 | 0.9320 | +0.8655 | 83% | 16 |
| lab-captured | cic-ids-2017 | no | 0.1667 | 0.0278 | 0.1667 | 0.0476 | 0.0476 +/-0.0000 | 0.9739 | +0.9262 | 75% | 16 |
| lab-captured | unsw-nb15 | no | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 +/-0.0000 | 0.9739 | +0.9739 | 83% | 16 |
| lab-captured | lab-captured | yes | 0.9739 | 0.9745 | 0.9739 | 0.9739 | 0.9739 +/-0.0013 | 0.9739 | ref | 100% | 16 |

## Per-dataset detail

| Train | Test | Shared labels | Test-only labels | Test rows evaluated | Note |
|---|---|---|---|---|---|
| cic-ids-2017 | cic-ids-2017 | Benign, Botnet, BruteForce, DDoS, DoS, Infiltration, PortScan, WebAttack | - | 320 / 320 | - |
| cic-ids-2017 | unsw-nb15 | Benign, Botnet, DoS, PortScan, WebAttack | Other | 1,000 / 1,200 | label subset excluded |
| cic-ids-2017 | lab-captured | Benign, Botnet, BruteForce, DoS, PortScan, WebAttack | - | 16,043 / 16,043 | - |
| unsw-nb15 | cic-ids-2017 | Benign, Botnet, DoS, PortScan, WebAttack | BruteForce, DDoS, Infiltration | 1,000 / 1,600 | label subset excluded |
| unsw-nb15 | unsw-nb15 | Benign, Botnet, DoS, Other, PortScan, WebAttack | - | 240 / 240 | - |
| unsw-nb15 | lab-captured | Benign, Botnet, DoS, PortScan, WebAttack | BruteForce | 13,364 / 16,043 | label subset excluded |
| lab-captured | cic-ids-2017 | Benign, Botnet, BruteForce, DoS, PortScan, WebAttack | DDoS, Infiltration | 1,200 / 1,600 | label subset excluded |
| lab-captured | unsw-nb15 | Benign, Botnet, DoS, PortScan, WebAttack | Other | 1,000 / 1,200 | label subset excluded |
| lab-captured | lab-captured | Benign, Botnet, BruteForce, DoS, PortScan, WebAttack | - | 3,209 / 3,209 | - |

## Interpretation

The mean macro-F1 falls from 0.9549 in-domain to 0.0381 cross-domain, a delta of 0.9168 (4.0% of in-domain F1 retained).

The in-domain reference for a pair `A->B` is the diagonal cell `A->A`: the same model family and the same feature space, evaluated on held-out data from the training domain. dF1 therefore isolates the cost of transfer from the quality of the in-domain fit.

Cells with `Test cov. < 100%` are scored only on labels the model was trained to emit; the excluded rows are listed above rather than silently dropped. Predictions that fall outside the trained label set are counted as errors.

## Notes

* Seeds: 42, 1337, 2024, 7, 99 (5 passes; every cell repeated per seed).
* `macro-F1` is the headline metric because the unified taxonomy is imbalanced; weighted F1 is retained in the CSV as `f1`.
* ΔF1 for a pair `A→B` is measured against the diagonal cell `A→A`, i.e. the same feature space and model family evaluated on held-out data from the training domain.
* Test rows whose label never appears in the training set are excluded from scoring and reported through `test_coverage`; predictions that fall outside the trained label set are counted as errors.
