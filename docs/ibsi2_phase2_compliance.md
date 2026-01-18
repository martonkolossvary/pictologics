# IBSI 2 Phase 2 Compliance Report

> [!NOTE]
> Only 3D methods (Config B) are implemented.
> Uses `RadiomicsPipeline` with `filter` step for all filtering.

## Preprocessing (Config B)

1. **Resample**: Tricubic spline to 1×1×1 mm
2. **Round**: Intensities to nearest integer
3. **Re-segment**: Mask to [-1000, 400] HU

## Summary

- **Total Tests**: 9
- **Passed**: 9
- **Failed**: 0

## Results

| ID | Filter | Status | Features | Notes |
|:---|:-------|:-------|:---------|:------|
| 1.B | None | ✅ | 18/18 | Baseline (no filter) |
| 2.B | Mean | ✅ | 18/18 | 3D, support=5 |
| 3.B | LoG | ✅ | 18/18 | σ=1.5mm, truncate=4σ |
| 4.B | Laws | ✅ | 18/18 | L5E5E5, rot-inv max, energy δ=7 |
| 5.B | Gabor | ✅ | 18/18 | σ=5mm, λ=2mm, γ=1.5, rot-inv avg |
| 6.B | Daubechies 3 | ✅ | 18/18 | LLH level 1, rot-inv avg |
| 7.B | Daubechies 3 | ✅ | 18/18 | HHH level 2, rot-inv avg |
| 8.B | Simoncelli | ✅ | 17/17 | B map level 1 |
| 9.B | Simoncelli | ✅ | 18/18 | B map level 2 |