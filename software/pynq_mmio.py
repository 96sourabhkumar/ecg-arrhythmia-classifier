"""
pynq_mmio.py -- register-level access to the ECG accelerator IP (ecg_accel_top) on the PYNQ-Z2,
plus the beat-window normaliser/quantiser the PS applies before writing a beat to the PL.

REGISTER MAP (AXI4-Lite, base 0x43C10000; source of truth: hardware/rtl/top/axi_lite_regs.v)
  0x00 CTRL        W   bit0 soft_reset, bit1 cnn_start_req (pulse), bit2 flush
  0x04 STATUS      R   bit0 cnn_done, bit1 peak_available, bit2 buf_overrun,
                        bit3 peak_dropped (sticky), bit4 start_pending, bit5 rr_dropped (sticky)
  0x08 PEAK_POP    R   pop-on-read: index of the oldest detected R-peak
  0x0C CLASS_OUT   R   {29'b0, class[2:0]}   (0=N 1=S 2=V 3=F 4=Q)
  0x10 XBUF_ADDR   W   sample ring-buffer read address (0..1023)
  0x14 XBUF_DATA   R   ring-buffer data at XBUF_ADDR
  0x18 BEAT_ADDR   W   beat-window write address (0..255)
  0x1C BEAT_DATA   W   writes one INT8 beat sample at BEAT_ADDR, auto-increments
  0x20 SAMPLE_IN   W   pushes one 12-bit 180 Hz ECG sample into the QRS front end
  0x24 DERIV_SAT   R   derivative saturation counter (debug)
  0x28 RR_DBG      R   RR word latched for the last classification (debug / verification)
  0x38 RR_PS       W/R RR feature word written by the PS {rr0, rr1, rr2, rr3}
  0x3C CFG         W/R bit0 = 1: CNN uses RR_PS (PS-computed RR); 0: hardware rr_feature_engine
The high-level runtime (peak post-processing, RR features, classification loop) is in
pynq_driver.py.
"""
import os
import sys
import time
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---------------------------------------------------------------------------
# Register map -- must match hardware/rtl/top/axi_lite_regs.v exactly
# ---------------------------------------------------------------------------
BASE_ADDR        = 0x43C10000     # fpga/vivado/create_project.tcl AXI_BASEADDR

A_CTRL           = 0x00
A_STATUS         = 0x04
A_PEAK_POP       = 0x08
A_CLASS_OUT      = 0x0C
A_XBUF_ADDR      = 0x10
A_XBUF_DATA      = 0x14
A_BEAT_ADDR      = 0x18
A_BEAT_DATA      = 0x1C
A_SAMPLE_IN      = 0x20
A_DERIV_SAT_CNT  = 0x24

CTRL_SOFT_RESET  = 1 << 0
CTRL_CNN_START   = 1 << 1
CTRL_FLUSH       = 1 << 2

ST_CNN_DONE      = 1 << 0
ST_PEAK_AVAIL    = 1 << 1
ST_BUF_OVERRUN   = 1 << 2
ST_PEAK_DROPPED  = 1 << 3
ST_START_PENDING = 1 << 4
ST_RR_DROPPED    = 1 << 5

RING_DEPTH       = 1024            # sample_ring_buffer.v AW=10
SAMPLE_PERIOD_S  = 1.0 / 180.0      # true 180 Hz real-time pacing -- do not
                                     # remove this sleep to "go faster"

# Beat window: 256 samples at 180 Hz, 90 before the R-peak and 166 after. (config.py's
# BEAT_PRE/BEAT_POST = 45/83 describe the earlier 128-sample design and are not used here.)
WIN_PRE  = 90
WIN_POST = 166
WIN_LEN       = WIN_PRE + WIN_POST   # 256


def _s12(v):
    """Sign-extend a 12-bit field."""
    v &= 0xFFF
    return v - 0x1000 if v & 0x800 else v


class EcgAccelMMIO:
    """Thin MMIO wrapper. Uses pynq.MMIO if available (real hardware);
    falls back to a NotImplementedError stub off-hardware so this file can
    still be imported (e.g. for the beat-quantization self-test at the
    bottom) without pynq installed."""

    def __init__(self, base_addr=BASE_ADDR, length=0x1000):
        try:
            from pynq import MMIO
            self.mmio = MMIO(base_addr, length)
        except ImportError:
            self.mmio = None

    def _require_mmio(self):
        if self.mmio is None:
            raise RuntimeError(
                "pynq.MMIO not available -- this call requires running on "
                "actual PYNQ-Z2 hardware with the pynq package installed. "
                "(Unit-testable parts of this file, like the quantizer, "
                "work without it -- see __main__.)")

    def write(self, addr, val):
        self._require_mmio()
        self.mmio.write(addr, val & 0xFFFFFFFF)

    def read(self, addr):
        self._require_mmio()
        return self.mmio.read(addr)

    def soft_reset(self):
        self.write(A_CTRL, CTRL_SOFT_RESET)
        self.write(A_CTRL, 0)

    def status(self):
        return self.read(A_STATUS)

    def peak_available(self):
        return bool(self.status() & ST_PEAK_AVAIL)

    def pop_peak(self):
        return self.read(A_PEAK_POP) & 0xFFFFF   # IDX_W = 20 bits

    def read_xbuf(self, addr):
        """addr must be 0..1023 (10-bit ring-buffer address)."""
        self.write(A_XBUF_ADDR, addr & (RING_DEPTH - 1))
        return _s12(self.read(A_XBUF_DATA))

    def push_sample(self, sample_12b_signed):
        self.write(A_SAMPLE_IN, sample_12b_signed & 0xFFF)

    def write_beat_window(self, beat_int8):
        """beat_int8: iterable of 256 ints in [-128, 127] -- WAS 128 in
        Build C's driver."""
        assert len(beat_int8) == WIN_LEN, \
            f"the accelerator expects a {WIN_LEN}-sample beat window, got {len(beat_int8)}"
        self.write(A_BEAT_ADDR, 0)
        for v in beat_int8:
            self.write(A_BEAT_DATA, int(v) & 0xFF)   # auto-increments addr

    def start_cnn(self):
        self.write(A_CTRL, CTRL_CNN_START)
        self.write(A_CTRL, 0)

    def wait_cnn_done(self, timeout_s=1.0, poll_interval_s=0.0005):
        t0 = time.monotonic()
        while True:
            st = self.status()
            if st & ST_CNN_DONE:
                return True
            if time.monotonic() - t0 > timeout_s:
                return False
            time.sleep(poll_interval_s)

    def class_out(self):
        return self.read(A_CLASS_OUT) & 0x7


# ---------------------------------------------------------------------------
# Beat-window normalize + INT8 quantize.
#
# Byte-for-byte the SAME formula int8_model.py already validated
# (self-checked 64/64 against tb_cnn_accelerator.v's own golden vectors first). Do
# NOT "simplify" this into a single division -- the two-step round-to-
# int16-then-round-to-int8 order matches what the golden model and RTL
# comparisons were actually built against.
# ---------------------------------------------------------------------------
def normalize_quantize_beat(x180_window, s_act):
    """x180_window: 256 raw decimated-sample values (int, 12-bit range).
    s_act: calibrated input scale, S['x'] in hardware/weights/quant_params.json.
    It is fixed by the trained/calibrated model and must match what
    cnn_accelerator.v's weights were quantized against."""
    w = np.asarray(x180_window, dtype=np.float64)
    w = w - w.mean()
    s = w.std()
    if s > 1e-6:
        w = w / s
    z16 = np.clip(np.round(w * 256), -32768, 32767).astype(np.int16)
    beat_int8 = np.clip(np.round((z16.astype(np.float64) / 256.0) / s_act),
                         -127, 127).astype(np.int64)
    return beat_int8


def load_input_scale():
    q_path = os.path.join(HERE, "..", "hardware", "weights", "quant_params.json")
    with open(q_path) as f:
        q = json.load(f)
    return float(q["S"]["x"])
