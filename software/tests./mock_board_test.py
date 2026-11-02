"""
mock_board_test.py -- runs the real PYNQ runtime (pynq_driver.py + board_test.py) against a
software model of the PL, so the PS code path can be checked without a board.

The fake PL implements the register map of hardware/rtl/top/axi_lite_regs.v:
  * SAMPLE_IN counts streamed samples; a detected R-peak becomes poppable DELAY samples after
    it (the hardware detector confirms peaks about 27 samples late). Peak positions are the
    golden hardware-detector peaks stored in the test vector, i.e. the PL detector is assumed
    correct (that is checked separately by the RTL testbenches and on the board).
  * START classifies the written beat window + RR_PS word with int8_model.py (the bit-exact
    reference of the CNN), latches RR_DBG, and sets the sticky cnn_done bit.
PASS here means: post-processing, RR features, window extraction, quantisation and the
register protocol in the renamed PS code reproduce the golden expectations bit-exactly.

    python software/tests/mock_board_test.py            # every record in pynq/test_vectors
    python software/tests/mock_board_test.py 113 202
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SW = os.path.join(HERE, '..'); sys.path.insert(0, SW)
import numpy as np
import pynq_mmio as M
import int8_model as G
import board_test as T

DELAY = 27
Q, MQ, gmi, gsh = G.load_weights(); CLASSIFY = G.make_classifier(Q, MQ, gmi, gsh); G.verify_against_gold(CLASSIFY)


class FakePL(T.D.EcgAccel):
    def __init__(self):
        self.mmio = object(); self.hw = []; self._reset()

    def _reset(self):
        self.n = 0; self.pk_i = 0; self.done = False; self.cls = 0; self.beat = [0] * 256; self.baddr = 0
        self.rr_ps = 0; self.cfg = 0; self.rr_dbg = 0; self.ring = [0] * 1024; self.xaddr = 0

    def write(self, addr, val):
        val &= 0xFFFFFFFF
        if addr == M.A_CTRL:
            if val & M.CTRL_SOFT_RESET: self._reset()
            if val & M.CTRL_CNN_START:
                b = np.array([(v - 256 if v > 127 else v) for v in self.beat], np.int32).reshape(1, 1, 256)
                w = self.rr_ps
                r = np.array([((w >> s) & 0xFF) for s in (24, 16, 8, 0)], np.int32)
                r = np.where(r > 127, r - 256, r).reshape(1, 4)
                self.cls = int(CLASSIFY(b, r)[0]); self.rr_dbg = w; self.done = True
        elif addr == M.A_SAMPLE_IN:
            v = val & 0xFFF; self.ring[self.n % 1024] = v; self.n += 1
        elif addr == M.A_BEAT_ADDR: self.baddr = val & 0xFF
        elif addr == M.A_BEAT_DATA: self.beat[self.baddr] = val & 0xFF; self.baddr = (self.baddr + 1) & 0xFF
        elif addr == M.A_XBUF_ADDR: self.xaddr = val & 1023
        elif addr == 0x38: self.rr_ps = val
        elif addr == 0x3C: self.cfg = val & 1

    def read(self, addr):
        if addr == M.A_STATUS:
            st = (1 if self.done else 0) | (2 if self.pk_i < len(self.hw) and self.hw[self.pk_i] + DELAY <= self.n else 0)
            self.done = False; return st
        if addr == M.A_PEAK_POP:
            p = int(self.hw[self.pk_i]); self.pk_i += 1; return p
        if addr == M.A_CLASS_OUT: return self.cls
        if addr == M.A_XBUF_DATA: return self.ring[self.xaddr]
        if addr == 0x28: return self.rr_dbg
        if addr == 0x3C: return self.cfg
        return 0


def main(records):
    paths = sorted(p for p in __import__('glob').glob(os.path.join(T.BT, 'record_*.npz'))
                   if not records or T._rec_id(p) in records)
    ok = True
    for p in paths:
        board = FakePL(); board.hw = list(map(int, np.load(p)['hw_peaks']))
        res = T.run_record(p, board, realtime=False); ok &= res['PASS']
    print('MOCK BOARD: ALL PASS' if ok and paths else 'MOCK BOARD: FAILED'); return ok


if __name__ == '__main__':
    sys.exit(0 if main({int(a) for a in sys.argv[1:]}) else 1)

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
