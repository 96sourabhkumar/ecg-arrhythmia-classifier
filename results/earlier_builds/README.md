# results/earlier_builds/

Vivado reports of the two designs that ran on the board before the released v7 build, kept because the comparison figures in `paper_figures/` (resources, timing, power, detector-in-the-loop accuracy) are drawn from them.

| File | Contents |
|---|---|
| `build_c_board_top_util.rpt`, `_timing.rpt`, `_power.rpt` | Build C: first board design (QRS detector + first CNN on one shared multiplier) |
| `board_top_v6_util.rpt`, `_timing.rpt`, `_power.rpt` | v6 board design (256-sample CNN + RR branch; v7 reuses this hardware) |
| `detector_loop_v6_summary.txt` | v6 accuracy with the detector in the loop, before the v7 peak correction |
