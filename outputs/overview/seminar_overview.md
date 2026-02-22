# Seminar Results Overview

## Mean ± Std Across Runs/Folds

| method | n_runs | CC_mean | KL_mean | EMD_mean | CC_std | KL_std | EMD_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AugSal | 4 | 0.000582 | 0.160675 | 2.276735 | 0.000014 | 0.002720 | 0.069215 |
| AugSalStrong | 4 | 0.000519 | 0.152449 | 2.227858 | 0.000013 | 0.002305 | 0.055648 |
| Baseline | 1 | 0.000706 | 0.301527 | 3.830145 | 0.000000 | 0.000000 | 0.000000 |
| Finetuned | 4 | 0.000492 | 0.147212 | 2.159278 | 0.000009 | 0.001881 | 0.035065 |
| MiaMix | 4 | 0.000461 | 0.253672 | 3.204473 | 0.000012 | 0.003469 | 0.015034 |

## Delta vs Baseline (positive is better)

| method | CC_delta_vs_baseline | KL_delta_vs_baseline | EMD_delta_vs_baseline | CC_pct_vs_baseline | KL_pct_vs_baseline | EMD_pct_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| AugSal | -0.000124 | 0.140852 | 1.553410 | -17.584412 | 46.712791 | 40.557479 |
| AugSalStrong | -0.000187 | 0.149078 | 1.602286 | -26.523114 | 49.441001 | 41.833571 |
| Baseline | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Finetuned | -0.000215 | 0.154315 | 1.670867 | -30.408226 | 51.177848 | 43.624125 |
| MiaMix | -0.000246 | 0.047856 | 0.625672 | -34.781755 | 15.871055 | 16.335457 |
