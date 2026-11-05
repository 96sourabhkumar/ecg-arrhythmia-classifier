"""
peak_postprocessing.py -- ARM-side (PS) beat post-processing for ecg_accel_top.  2026-10-08

WHY (measured, see docs/accuracy_analysis.txt):
  The hardware Pan-Tompkins detector reports each beat at the maximum of |bandpass|
  within +-8 samples of (MWI peak - GROUP_DELAY). On MIT-BIH that lands on the R,
  S or Q wave depending on morphology: detected-minus-annotated offset std 10.8
  samples (60 ms), bimodal. It also double-counts tall T-waves (rec 117: 3002
  detections for 1535 beats) and stops detecting for up to 22 s after its
  threshold over-adapts (rec 228). Misplaced windows + corrupted RR features cost
  the deployed system 95.6% -> 86.6%.

WHAT (each step uses only the decimated x180 samples the PS itself streams):
  1. refine   p' = argmax |x[p-24..p+24] - median(x[p-24..p+24])|
              -> offset std 10.8 -> 1.8 samples, 97% of beats within +-2.
  2. dedupe   detections < 36 samples (200 ms refractory) apart after refining
              are the same QRS; keep the steeper one.
  3. T-reject drop a detection whose max |x[n+1]-x[n-1]| (+-6 samples) is below
              KT x median of the last 8 accepted beats' slopes (T-waves are slow).
  4. searchback if the gap since the last beat exceeds SBF x median(last 8 RR),
              insert the steepest point in the gap (if slope >= KSB x ref);
              repeat up to MAXFILL times.
  KT/KSB/SBF/MAXFILL were selected on DS1 ONLY (FP+FN on the 22 training
  records); DS2 was evaluated once with the frozen values.

ONLINE CONTRACT: process HW peak q once samples up to q+W have been streamed.
Peak k is FINAL once peak k+2 exists (a later detection can still replace k+1
via dedupe, which changes beat k's post-RR). Beat k (k>=1) is classified when
final; its RR features come from qrs/rr_features.py::rr_int8 on the accepted list,
which reads peaks[0..k+1] only, so online == offline bit-for-bit.
"""
import numpy as np

W, REFR = 24, 36
KT, KSB, SBF, MAXFILL, NRR = 0.2, 0.5, 1.5, 10, 8


def refine(x, p, w=W):
    a = max(0, p - w)
    s = np.asarray(x[a:p + w + 1], dtype=np.float64)
    return a + int(np.argmax(np.abs(s - np.median(s))))


def slope(x, p, h=6):
    a = max(1, p - h); b = min(len(x) - 1, p + h + 1)
    s = np.asarray(x[a:b + 1], dtype=np.int64)
    return int(np.max(np.abs(s[2:] - s[:-2]))) if len(s) > 2 else 0


class PostProcessor:
    """Feed hardware peaks in order with push(q, x). x = all samples streamed so
    far (must include q+W). Accepted peaks accumulate in self.pk."""

    def __init__(self, kt=KT, ksb=KSB, sbf=SBF, maxfill=MAXFILL, use_t=True, use_sb=True):
        self.kt, self.ksb, self.sbf, self.maxfill = kt, ksb, sbf, maxfill
        self.use_t, self.use_sb = use_t, use_sb
        self.pk, self.sl = [], []

    def push(self, q, x):
        pk, sl = self.pk, self.sl
        p = refine(x, int(q)); s = slope(x, p)
        if pk and p - pk[-1] < REFR:                       # same QRS
            if s > sl[-1]:
                pk[-1] = p; sl[-1] = s
            return
        ref = float(np.median(sl[-8:])) if len(sl) >= 3 else None
        if self.use_t and ref is not None and s < self.kt * ref:
            return                                          # T-wave / noise
        if self.use_sb and ref is not None and len(pk) >= 3:
            rr = float(np.percentile(np.diff(pk[-(NRR + 1):]), 50))
            for _ in range(self.maxfill):
                if p - pk[-1] <= self.sbf * rr:
                    break
                a = pk[-1] + REFR; b = min(p - REFR, pk[-1] + int(1.5 * rr))
                if b <= a:
                    break
                xa = np.asarray(x[a - 1:b + 1], dtype=np.int64)    # centred diff at n = a..b-1
                d = np.abs(xa[2:] - xa[:-2])
                c = refine(x, a + int(np.argmax(d))); sc = slope(x, c)
                if sc < self.ksb * ref or c - pk[-1] < REFR or p - c < REFR:
                    break
                pk.append(c); sl.append(sc)
        pk.append(p); sl.append(s)


def postprocess(x, hw_peaks, **kw):
    """Offline convenience: run the online processor over a whole record."""
    pp = PostProcessor(**kw)
    for q in hw_peaks:
        pp.push(int(q), x)
    return np.asarray(pp.pk, dtype=np.int64)
