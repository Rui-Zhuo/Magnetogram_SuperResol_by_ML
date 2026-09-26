# Metric definitions

Let `P` be a prediction and `G` the SP reference on the same pixel grid, both in Gauss. All sample metrics are calculated before averaging across samples.

| Metric | Definition |
|---|---|
| MAE | `mean(abs(P-G))` |
| RMSE | `sqrt(mean((P-G)^2))` |
| normalized MAE/RMSE | Divide the sample error by `std(G)` with population variance |
| Pearson CC | `sum((P-mean(P))*(G-mean(G))) / (sqrt(sum(Pc^2)*sum(Gc^2)) + 1e-8)` |
| R² | `1 - sum((G-P)^2)/(sum((G-mean(G))^2) + 1e-8)` |
| SSIM | scikit-image default 7×7 window with `data_range=max(G)-min(G)` |
| historical PSNR | `20*log10(max(G)/RMSE)`; this is intentionally not peak-to-peak PSNR |
| net flux ratio | `sum(P)/sum(G)` on the same grid |
| unsigned flux ratio | `sum(abs(P))/sum(abs(G))` on the same grid |

SSIM and normalized errors are undefined for a constant reference. PSNR is infinite for zero error and undefined when a nonzero-error reference maximum is non-positive. Zero reference flux produces an undefined ratio. Non-finite arrays are rejected rather than masked piecemeal, since masking changes SSIM neighbourhoods. Summary JSON uses null where a finite mean cannot be formed and records the count of finite values. Infinite perfect-prediction PSNR values remain visible in the per-sample CSV but are not folded into finite-value summary means.

For agreement scores, the statistics implementation first averages flux ratios within a population and then computes `1 - abs(mean_ratio - 1)`. This is different from averaging each sample's `1 - abs(ratio - 1)`. Scores can be negative and must not be silently clipped. Ratios become unstable when signed reference flux is near zero; report their finite counts and inspect the underlying sample CSV.

Projected pixel area uses 0.5 arcsec HMI sampling and the research conversion `7e8/960` metres per arcsecond. The target area is the HMI projected area divided by `(1.56*1.68)`. This is an image-plane approximation, not a foreshortening-corrected surface area or radial-field conversion. Flux in Webers requires `B_G * 1e-4 * area_m2`. The old code used `1e-5`; `magnetosr/metrics.py` uses the correct SI conversion. On a common grid this constant cancels from the published flux ratios.

AR-like labels use `percentile(abs(HMI), 99) >= 600 G`. They describe patch magnetic activity, not catalogue membership. The angle bins are based on `sin(theta)` with left-inclusive boundaries: [0,0.5), [0.5,0.8), [0.8,1]. Metadata are copied from the saved test classification, not recomputed from predictions. Group summaries never substitute the full dataset for an empty subgroup.

The release compares each model output with a bicubic floor. Historical empirical calibration relations are retained only in the research backup; no new physical unit interpretation of their intercepts is assumed here. Additional baseline predictions can be evaluated by placing arrays under the `pred` key with the same sample filenames.
