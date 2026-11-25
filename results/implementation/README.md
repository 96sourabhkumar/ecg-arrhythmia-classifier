# results/implementation/

Vivado 2025.1 reports of the released design (`system_wrapper`: Zynq PS7 + AXI interconnect +
`ecg_accel_top`), device xc7z020clg400-1, 50 MHz.

| File | Key numbers |
|---|---|
| `utilization.rpt` | 6,146 LUTs (11.55 %), 4,560 FFs (4.29 %), 4.5 BRAM tiles (3.21 %), 2 DSP48E1 (0.91 %) |
| `utilization_hierarchical.rpt` | Per-module breakdown (instance names `u_cnn`, `u_fe`, `u_th`, ...) |
| `timing_summary.rpt` | WNS +5.478 ns, WHS +0.024 ns, all constraints met |
| `power.rpt` | 1.418 W total on chip: PS7 1.256 W, accelerator 0.025 W, AXI 0.002 W, static 0.135 W; vectorless, confidence Medium |
| `synthesis_utilization.rpt`, `synthesis_utilization_hierarchical.rpt` | Post-synthesis versions |
| `vivado_build.log` | Log of the build that produced the released bitstream |

Power is a tool estimate without switching activity from simulation; it was not measured on the board.
