# Cross-Dataset Generalization Report - `full` features

* Model: `xgb`  |  Feature mode: `full`  |  Datasets: cic-ids-2017, unsw-nb15, lab-simulated
* Seeds: 42 (1 independent passes)
* Pairs: 9 ordered pairs (3 datasets, diagonal included)

> [!NOTE]
> Scientific inference is limited strictly to the evaluated datasets (N=3 datasets, 9 directed pairs); pair observations share dataset marginals and are not independent.

## Headline

| Quantity | Value |
|---|---|
| Mean in-domain macro-F1 | 0.9890 |
| Mean cross-domain macro-F1 | 0.9269 |
| Mean majority-baseline macro-F1 | 0.0465 |
| Mean binary Benign-vs-Attack F1 | 0.9943 |
| Mean F1 delta (in - cross) | **0.0621** |
| Cross-domain F1 retention | 93.7% |
| Mean test-label coverage | 87.5% |
| Cross-domain worse than in-domain | YES |
| Cross-domain pairs with a negative delta | 4 / 6 |

Values are means over seeds; `+/-` is the sample standard deviation across seeds.

## Full matrix

| Train | Test | In-domain | Macro-F1 (Mean +/- Std) | Row 95% CI | Maj-Class Macro-F1 | In-domain Ref | dF1 | Shared Feat. | Test Cov. |
|---|---|---|---|---|---|---|---|---|---|
| cic-ids-2017 | cic-ids-2017 | yes | 0.9951 +/-0.0000 | [0.9911, 0.9980] | 0.0278 | 0.9951 | ref | 28 | 100% |
| cic-ids-2017 | unsw-nb15 | no | 0.9808 +/-0.0000 | [0.9770, 0.9842] | 0.0000 | 0.9951 | +0.0144 | 18 | 83% |
| cic-ids-2017 | lab-simulated | no | 0.7776 +/-0.0000 | [0.7721, 0.7835] | 0.0477 | 0.9951 | +0.2175 | 28 | 100% |
| unsw-nb15 | cic-ids-2017 | no | 0.9918 +/-0.0000 | [0.9893, 0.9944] | 0.0667 | 0.9880 | -0.0038 | 18 | 62% |
| unsw-nb15 | unsw-nb15 | yes | 0.9880 +/-0.0000 | [0.9813, 0.9936] | 0.0476 | 0.9880 | ref | 18 | 100% |
| unsw-nb15 | lab-simulated | no | 0.8562 +/-0.0000 | [0.8507, 0.8618] | 0.0668 | 0.9880 | +0.1318 | 18 | 83% |
| lab-simulated | cic-ids-2017 | no | 0.9926 +/-0.0000 | [0.9902, 0.9948] | 0.0476 | 0.9838 | -0.0088 | 28 | 75% |
| lab-simulated | unsw-nb15 | no | 0.9622 +/-0.0000 | [0.9565, 0.9675] | 0.0667 | 0.9838 | +0.0216 | 18 | 83% |
| lab-simulated | lab-simulated | yes | 0.9838 +/-0.0000 | [0.9793, 0.9878] | 0.0480 | 0.9838 | ref | 28 | 100% |

## Per-dataset detail

| Train | Test | Shared labels | Test-only labels | Test rows evaluated | Note |
|---|---|---|---|---|---|
| cic-ids-2017 | cic-ids-2017 | Benign, Botnet, BruteForce, DDoS, DoS, Infiltration, PortScan, WebAttack | - | 1,440 / 1,440 | - |
| cic-ids-2017 | unsw-nb15 | Benign, Botnet, DoS, PortScan, WebAttack | Other | 4,500 / 5,400 | label subset excluded |
| cic-ids-2017 | lab-simulated | Benign, Botnet, BruteForce, DoS, PortScan, WebAttack | - | 16,043 / 16,043 | - |
| unsw-nb15 | cic-ids-2017 | Benign, Botnet, DoS, PortScan, WebAttack | BruteForce, DDoS, Infiltration | 4,500 / 7,200 | label subset excluded |
| unsw-nb15 | unsw-nb15 | Benign, Botnet, DoS, Other, PortScan, WebAttack | - | 1,080 / 1,080 | - |
| unsw-nb15 | lab-simulated | Benign, Botnet, DoS, PortScan, WebAttack | BruteForce | 13,364 / 16,043 | label subset excluded |
| lab-simulated | cic-ids-2017 | Benign, Botnet, BruteForce, DoS, PortScan, WebAttack | DDoS, Infiltration | 5,400 / 7,200 | label subset excluded |
| lab-simulated | unsw-nb15 | Benign, Botnet, DoS, PortScan, WebAttack | Other | 4,500 / 5,400 | label subset excluded |
| lab-simulated | lab-simulated | Benign, Botnet, BruteForce, DoS, PortScan, WebAttack | - | 3,209 / 3,209 | - |

## Binary Benign-vs-Attack Performance

| Train | Test | In-domain | Binary F1 | Recall | FPR | AUROC | MCC | Majority Binary F1 |
|---|---|---|---|---|---|---|---|---|
| cic-ids-2017 | cic-ids-2017 | yes | 0.9976 | 0.9984 | 0.0222 | n/a | 0.9809 | 0.9333 |
| cic-ids-2017 | unsw-nb15 | no | 0.9891 | 0.9944 | 0.0656 | n/a | 0.9446 | 0.8889 |
| cic-ids-2017 | lab-simulated | no | 0.9938 | 0.9966 | 0.0450 | n/a | 0.9626 | 0.9086 |
| unsw-nb15 | cic-ids-2017 | no | 0.9951 | 0.9944 | 0.0167 | n/a | 0.9758 | 0.8889 |
| unsw-nb15 | unsw-nb15 | yes | 0.9945 | 0.9967 | 0.0389 | n/a | 0.9665 | 0.9091 |
| unsw-nb15 | lab-simulated | no | 0.9939 | 0.9942 | 0.0253 | n/a | 0.9697 | 0.8882 |
| lab-simulated | cic-ids-2017 | no | 0.9971 | 0.9967 | 0.0122 | n/a | 0.9827 | 0.9091 |
| lab-simulated | unsw-nb15 | no | 0.9921 | 0.9925 | 0.0333 | n/a | 0.9604 | 0.8889 |
| lab-simulated | lab-simulated | yes | 0.9951 | 0.9944 | 0.0204 | n/a | 0.9711 | 0.9085 |

## Feature Harmonization & Drop Detail

| Train | Test | Shared Features | Train Features Dropped | Test Features Dropped | Dropped Names |
|---|---|---|---|---|---|
| cic-ids-2017 | cic-ids-2017 | 28 | 0 | 0 | - |
| cic-ids-2017 | unsw-nb15 | 18 | 10 | 0 | bwd_iat_mean, bwd_packet_length_std, fin_flag_count, flow_iat_std, fwd_iat_mean, fwd_packet_length_std, packet_length_mean, packet_length_std, psh_flag_count, rst_flag_count |
| cic-ids-2017 | lab-simulated | 28 | 0 | 0 | - |
| unsw-nb15 | cic-ids-2017 | 18 | 0 | 10 | bwd_iat_mean, bwd_packet_length_std, fin_flag_count, flow_iat_std, fwd_iat_mean, fwd_packet_length_std, packet_length_mean, packet_length_std, psh_flag_count, rst_flag_count |
| unsw-nb15 | unsw-nb15 | 18 | 0 | 0 | - |
| unsw-nb15 | lab-simulated | 18 | 0 | 10 | bwd_iat_mean, bwd_packet_length_std, fin_flag_count, flow_iat_std, fwd_iat_mean, fwd_packet_length_std, packet_length_mean, packet_length_std, psh_flag_count, rst_flag_count |
| lab-simulated | cic-ids-2017 | 28 | 0 | 0 | - |
| lab-simulated | unsw-nb15 | 18 | 10 | 0 | bwd_iat_mean, bwd_packet_length_std, fin_flag_count, flow_iat_std, fwd_iat_mean, fwd_packet_length_std, packet_length_mean, packet_length_std, psh_flag_count, rst_flag_count |
| lab-simulated | lab-simulated | 28 | 0 | 0 | - |

## Interpretation

The mean macro-F1 falls from 0.9890 in-domain to 0.9269 cross-domain, a delta of 0.0621 (93.7% of in-domain F1 retained).

The in-domain reference for a pair `A->B` is the diagonal cell `A->A`: the same model family and the same feature space, evaluated on held-out data from the training domain. dF1 isolates the cost of transfer from the quality of the in-domain fit.

Multi-class macro-F1 is evaluated strictly over shared labels with `labels=shared_labels`.
Evaluation uses the strict feature intersection between train and test domains, with zero dummy imputation.

## Notes

* Seeds: 42 (1 passes; every cell repeated per seed).
* `macro-F1` is the headline metric because the unified taxonomy is imbalanced; weighted F1 is retained in the CSV as `f1`.
* ΔF1 for a pair `A→B` is measured against the diagonal cell `A→A`, i.e. the same feature space and model family evaluated on held-out data from the training domain.
* Test rows whose label never appears in the training set are excluded from scoring and reported through `test_coverage`; predictions that fall outside the trained label set are counted as errors.
