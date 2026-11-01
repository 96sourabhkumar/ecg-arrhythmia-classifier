# results/board_run/

`board_test_results_2026-10-09.txt` is the unedited notebook output of the board test that produced
the headline result: one JSON block per record (peak, RR and class agreement with the golden model,
accuracy, mean CNN latency, wall time, PASS), followed by the summary:

```
records 22   all PASS (bit-exact vs golden): True   beats classified on FPGA: 49027
ON-BOARD accuracy, minutes 5-30 (patient-adaptive test set): 39613/40816 = 97.05%
on-board accuracy, whole records (includes training minutes): 47530/48794 = 97.41%
```

It was produced with the development versions of the scripts (`board_test_v7.py`, driver
`v7-fix2-2026-10-09`); the code in `software/` is the
same code under release names.
