# SCache refactor measurements

Measured on 2026-10-24 with Julia 1.13.0, aarch64 macOS, OpenBLAS 0.3.30, and one BLAS thread. The baseline is an untouched source snapshot of commit `6eb1618`; both versions used the same dependency environment and benchmark driver. All cases have 300 observations. Times are median milliseconds after warm-up. Each benchmark process ran separately; these are local synthetic measurements, not guarantees for other hardware or datasets.

## 1 Julia thread

Each cell is **original → refactored**, in milliseconds. Configuration columns list predictors / retained rank / active columns, including the intercept.

| Operation | 10,000 / 100 / 11 | 10,000 / 300 / 41 | 100,000 / 100 / 11 | 100,000 / 300 / 41 |
|---|---:|---:|---:|---:|
| Response refresh | 1.094 → 0.471 | 2.563 → 0.747 | 12.978 → 5.073 | 27.542 → 7.581 |
| Covariance + response refresh | 1.097 → 0.690 | 2.556 → 1.291 | 13.376 → 7.066 | 27.276 → 12.658 |
| Unchanged-model update | 1.368 → 0.430 | 7.960 → 1.568 | 13.889 → 4.400 | 81.403 → 15.820 |
| Add predictor + update | 2.131 → 1.035 | 9.097 → 2.459 | 22.035 → 12.392 | 96.942 → 34.147 |
| Remove predictor + update | 1.382 → 0.477 | 8.036 → 1.648 | 14.168 → 5.583 | 83.588 → 17.360 |
| 25 proposals | 0.107 → 0.059 | 1.043 → 0.649 | 0.109 → 0.059 | 1.074 → 0.671 |
| Full auxiliary transition | 2.897 → 2.270 | 13.323 → 10.970 | 27.647 → 20.320 | 110.924 → 87.613 |

## 4 Julia threads

Each cell is **original → refactored**, in milliseconds. Configuration columns list predictors / retained rank / active columns, including the intercept.

| Operation | 10,000 / 100 / 11 | 10,000 / 300 / 41 | 100,000 / 100 / 11 | 100,000 / 300 / 41 |
|---|---:|---:|---:|---:|
| Response refresh | 0.789 → 0.232 | 1.668 → 0.405 | 7.132 → 2.698 | 17.284 → 4.104 |
| Covariance + response refresh | 0.797 → 0.312 | 1.642 → 0.712 | 8.545 → 4.125 | 18.487 → 7.924 |
| Unchanged-model update | 0.491 → 0.170 | 2.331 → 0.474 | 5.498 → 1.307 | 23.342 → 4.277 |
| Add predictor + update | 1.007 → 0.473 | 2.835 → 1.041 | 8.511 → 5.906 | 35.850 → 11.488 |
| Remove predictor + update | 0.504 → 0.218 | 2.436 → 0.531 | 5.438 → 2.265 | 32.687 → 5.296 |
| 25 proposals | 0.135 → 0.063 | 1.024 → 0.676 | 0.115 → 0.062 | 1.052 → 0.679 |
| Full auxiliary transition | 2.034 → 1.119 | 6.966 → 5.583 | 12.008 → 8.086 | 42.047 → 30.014 |

## Allocation and storage tradeoff

Representative allocated bytes with four Julia threads:

| Operation | 10,000 predictors, rank 100, 11 active | 100,000 predictors, rank 300, 41 active |
|---|---:|---:|
| Response refresh | 1,856 → 0 | 5,248 → 0 |
| Covariance + response refresh | 1,856 → 96 | 5,248 → 96 |
| Unchanged-model update | 992,128 → 400 | 33,883,536 → 400 |
| Add predictor + update | 2,160,480 → 85,536 | 69,237,328 → 832,944 |
| Remove predictor + update | 1,990,816 → 85,520 | 67,631,696 → 832,944 |
| 25 proposals | 445,200 → 0 | 4,247,200 → 0 |
| Full auxiliary transition | 2,032,880 → 527,584 | 41,492,416 → 2,717,200 |

With one Julia thread, warmed cache refreshes, unchanged-model updates and proposal evaluations measured zero allocated bytes. The remaining flip allocation is predominantly predictor-index bookkeeping, proportional to predictor count; it is not an active-by-predictor matrix allocation. Auxiliary transitions still allocate coefficient-draw and latent-response arrays, which are outside this first refactor.

Lower recurring allocation comes with retained covariance-adjusted active products. Their logical size is `a × (p + 1)` floating-point values; backing arrays reserve geometrically growing model sizes and worker scratch is bounded by `allocated_model_size × 256 × workers`. No full predictor-by-predictor Gram matrix is retained. Growth in allocated model size itself allocates and is excluded from warmed measurements.

The gain is largest for repeated model updates and removals. Complete auxiliary transitions benefit less because covariance changes still require rebuilding active products and coefficient draws retain their existing implementation. Timings were collected before the final failure-propagation guard and introspection checks; those do not change numerical work or matrix allocation scaling.

