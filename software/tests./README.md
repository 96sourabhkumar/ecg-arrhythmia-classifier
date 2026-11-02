# software/tests/

`mock_board_test.py` runs the real board runtime (`pynq_driver.py` through `board_test.py`) against
a software model of the programmable logic: streamed samples, R-peaks that become available 27
samples after they occur (the hardware latency), the beat and RR registers, and a CNN computed by
`int8_model.py`. It checks that the PS code reproduces the golden test vectors bit-exactly.

```bash
cd software
python tests/mock_board_test.py 113 202 228     # selected records
python tests/mock_board_test.py                 # every record in pynq/test_vectors/
```
Result for this release: records 113, 202 and 228 PASS with the same accuracy as on the board.
