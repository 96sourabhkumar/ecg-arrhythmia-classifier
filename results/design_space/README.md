# results/design_space/

Ten out-of-context builds from the hardware design-space study (all routed at 50 MHz on the
XC7Z020), each with `utilization.rpt`, `timing.rpt` and `power.rpt`. The original build script is
the original synthesis script and RTL of these earlier builds are not part of this repository.

| Folder | Top module (archive) | What it is | LUT | FF | DSP | WNS (ns) |
|---|---|---|---|---|---|---|
| `A_frontend` | `frontend_top` | QRS front end with its own multipliers | 1,934 | 2,714 | 1 | +6.69 |
| `B_cnn` | `cnn_top` | first CNN (128-sample window) | 2,035 | 382 | 4 | +0.30 |
| `AB_frontend_cnn_unshared` | `naive_ab_top` | A and B side by side | 3,140 | 3,059 | 8 | +0.28 |
| `ABN_frontend_cnn_unshared_narrow` | `naive_ab_narrow_top` | A + B with measured (narrow) bit widths | 2,261 | 1,820 | 2 | +5.09 |
| `BN_cnn_narrow` | `cnn_narrow_top` | B with narrow widths | 1,646 | 343 | 1 | +4.85 |
| `C_frontend_cnn_shared_multiplier` | `shared_top` | A + B on one shared multiplier | 2,280 | 1,914 | 1 | +7.69 |
| `FA_frontend_arithmetic` | `frontend_arith_top` | front-end arithmetic only | 1,172 | 2,676 | 4 | +10.21 |
| `FN_frontend_narrow` | `frontend_narrow_top` | front end with narrow widths | 615 | 1,477 | 1 | +11.20 |
| `RR_rr_feature_engine` | `rr_engine` | RR feature engine alone | 452 | 491 | 0 | +8.64 |
| `SQ_squarer` | `squarer_only_top` | squarer alone | 49 | 35 | 4 | +12.15 |

The released design descends from build C (shared multiplier) with the narrow front end.
