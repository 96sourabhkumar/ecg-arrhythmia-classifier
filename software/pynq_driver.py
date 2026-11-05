"""
pynq_driver.py -- PS runtime for ecg_accel_top: HW QRS detector -> PS peak post-processing
-> PS RR features -> HW CNN. Verified on the PYNQ-Z2 on 2026-10-09 (49,027 beats bit-exact).

What the PS adds on top of the register access in pynq_mmio.py:
  * HW peaks are cleaned by peak_postprocessing.PostProcessor (refine / dedupe /
    T-wave reject / searchback). See that file for the measured rationale.
  * RR features are computed on the PS with the canonical integer spec
    (qrs/rr_features.py::rr_int8, the same spec rr_feature_engine.v implements) on the
    CLEANED peak train, written to RR_PS (0x38); CFG (0x3C) bit0 = 1 selects it.
  * Beat k is classified once it is final (peak k+2 exists), so every beat's
    window AND RR are exactly what the offline evaluation used.
  * The beat window is taken from the samples the PS itself streamed (identical
    to the PL ring buffer -- proven 1063/1063 beats on hardware in an earlier board test).
    read_from_ring=True reads it back from the PL ring instead, as a check.

Registers used beyond the basic map in pynq_mmio.py:
  0x38 RR_PS  W/R  {rr0[31:24], rr1, rr2, rr3[7:0]}
  0x3C CFG    W/R  bit0 rr_src_ps
"""
import os, sys, time, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pynq_mmio import (EcgAccelMMIO, normalize_quantize_beat, WIN_PRE,
                            WIN_POST, WIN_LEN, RING_DEPTH, SAMPLE_PERIOD_S)
from peak_postprocessing import PostProcessor, W as REF_W
# FIX 2026-10-09 (board run, record 113): a HW peak q may only be post-processed once
# every sample the post-processor can touch exists: refine() reads up to q+24 and then
# slope() reads up to (refined p)+7 <= q+31. The HW reports peaks ~27 samples late, so
# the old condition (i >= q+24) let slope() run on a truncated window. Wait for q+32.
LOOKAHEAD = REF_W + 8
DRIVER_VERSION = '1.0.0'
from qrs.rr_features import rr_int8, constants

A_RR_PS, A_CFG = 0x38, 0x3C
Q_JSON = os.path.join(HERE, '..', 'hardware', 'weights', 'quant_params.json')


class EcgAccel(EcgAccelMMIO):
    def set_rr_source_ps(self, on=True):
        self.write(A_CFG, 1 if on else 0)

    def write_rr(self, rr4):
        v = [int(t) & 0xFF for t in rr4]
        self.write(A_RR_PS, (v[0] << 24) | (v[1] << 16) | (v[2] << 8) | v[3])


def load_q(path=Q_JSON):
    q = json.load(open(path))
    K, CLIP = constants(q['S']['rr'])
    return float(q['S']['x']), K, CLIP


def classify_beat(board, x, pk, k, s_act, K, CLIP, read_from_ring=False):
    p = pk[k]; lo = p - WIN_PRE
    if read_from_ring:
        win = [board.read_xbuf((lo + t) % RING_DEPTH) for t in range(WIN_LEN)]
    else:
        win = x[lo:lo + WIN_LEN]
    beat = normalize_quantize_beat(win, s_act)
    rr = rr_int8(pk, k, K, CLIP)
    board.write_beat_window(beat)
    board.write_rr(rr)
    t0 = time.perf_counter(); board.start_cnn()
    # FIX 2026-10-09 (board run, record 202): one result arrived after the 1 s timeout
    # (an OS/Jupyter stall -- mean CNN latency is 12.7 ms), so the beat was dropped.
    # Wait longer, and if it really never completes, re-issue the start once.
    if not board.wait_cnn_done(timeout_s=5.0):
        board.write_rr(rr); board.start_cnn()
        if not board.wait_cnn_done(timeout_s=5.0):
            return None, rr, None
    return board.class_out(), rr, time.perf_counter() - t0


def run(x180, board=None, realtime=True, read_from_ring=False, q_path=Q_JSON):
    """Stream x180 (decimated samples) and classify every beat.
    Returns dict(peaks=[accepted peaks], beats=[(k, peak, class, rr, latency_s)])."""
    board = board or EcgAccel()
    s_act, K, CLIP = load_q(q_path)
    board.soft_reset(); board.set_rr_source_ps(True)
    pp = PostProcessor(); hwq = []; hw_all = []; done = 0; out = []
    x = []; n = len(x180); t_next = time.monotonic()
    for i in range(n):
        if realtime:
            now = time.monotonic()
            if now < t_next: time.sleep(t_next - now)
            t_next += SAMPLE_PERIOD_S
        board.push_sample(int(x180[i])); x.append(int(x180[i]))
        while board.peak_available():
            q = board.pop_peak(); hwq.append(q); hw_all.append(q)
        while hwq and i >= hwq[0] + LOOKAHEAD:
            pp.push(hwq.pop(0), x)
        pk = pp.pk
        # beat k is final when peak k+2 exists; window needs samples to p+165
        while len(pk) >= done + 4 and i >= pk[done + 1] + WIN_POST - 1:
            k = done + 1; done = k                    # beat k final: peak k+2 exists
            if pk[k] - WIN_PRE < 0:
                continue                              # window starts before record
            cls, rr, lat = classify_beat(board, x, pk, k, s_act, K, CLIP, read_from_ring)
            out.append((k, pk[k], cls, rr.tolist(), lat))
    # end of stream: remaining HW peaks, then every beat that has a next peak
    for q in hwq:
        pp.push(q, x)                                 # refine() clips at the record end
    pk = pp.pk
    for k in range(done + 1, len(pk) - 1):
        if pk[k] - WIN_PRE >= 0 and pk[k] + WIN_POST <= len(x):
            cls, rr, lat = classify_beat(board, x, pk, k, s_act, K, CLIP, read_from_ring)
            out.append((k, pk[k], cls, rr.tolist(), lat))
    return dict(hw_peaks=hw_all, peaks=list(pk), beats=out)
