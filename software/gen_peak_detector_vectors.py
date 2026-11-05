"""
gen_peak_detector_vectors.py -- vectors for tb_peak_detector.v

Dumps the MWI and bandpass streams for one record plus the EXACT peak list the
Python golden model produces, so the RTL can be checked beat-for-beat rather
than by counting.

WHY A WHOLE RECORD AND NOT A SYNTHETIC STIMULUS
-----------------------------------------------
peak_detector is adaptive: SPKI and NPKI depend on the entire history. A short
synthetic burst would never exercise the adaptation, the T-wave rejection, or
the refractory logic in the states they actually reach on real data. Record 100
is used for the same reason the other front-end vectors use it, and record 108
(severe baseline wander) is offered as the hard case.

WHAT THIS CANNOT PROVE
----------------------
Matching peaks on one record does not prove the FSM is correct on all 44. It
proves the FSM matches the golden model on this one. The golden model is the
thing already validated against the literature; this test verifies the
translation to hardware, nothing more.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mitbih import load_record
from qrs.pan_tompkins import pan_tompkins_fixed, detect_peaks_fixed

VEC = os.path.join(HERE, '..', 'hardware', 'sim', 'vectors')
RECORD = int(sys.argv[1]) if len(sys.argv) > 1 else 100
NSAMP  = int(sys.argv[2]) if len(sys.argv) > 2 else 20000


def wr(path, arr, nybbles):
    with open(path, 'w') as f:
        for v in arr:
            f.write(f"{int(v) & ((1 << (4*nybbles)) - 1):0{nybbles}x}\n")


def main():
    print(f"record {RECORD}, first {NSAMP} samples of the 180 Hz stream")
    adc, _, _ = load_record(RECORD)
    _, _, st = pan_tompkins_fixed(adc, return_stages=True)

    mwi = np.asarray(st['mwi'],  dtype=np.int64)[:NSAMP]
    bp  = np.asarray(st['hp'],   dtype=np.int64)[:NSAMP]

    # Re-run the detector on exactly the truncated streams the RTL will see,
    # so the expected peaks correspond to this stimulus and not to the full
    # record. Truncating the input but keeping whole-record peaks would make
    # the test pass or fail for the wrong reason.
    peaks, _ = detect_peaks_fixed(mwi, bp)
    peaks = np.asarray(peaks, dtype=np.int64)
    peaks = peaks[peaks < NSAMP]

    print(f"  mwi range   {mwi.min():,} .. {mwi.max():,}")
    print(f"  bp  range   {bp.min():,} .. {bp.max():,}")
    print(f"  |bp| max    {np.abs(bp).max():,}   (needs {int(np.abs(bp).max()).bit_length()} bits unsigned)")
    print(f"  peaks found {len(peaks)}")
    if len(peaks) > 1:
        rr = np.diff(peaks)
        print(f"  RR interval {rr.min()} .. {rr.max()} samples "
              f"({60*180/rr.max():.0f} .. {60*180/rr.min():.0f} bpm)")

    assert len(peaks) > 20, "too few peaks -- stimulus is not exercising the FSM"
    assert mwi.min() >= 0, "MWI must be non-negative"

    os.makedirs(VEC, exist_ok=True)
    wr(os.path.join(VEC, 'peak_detector_mwi_input.mem'),   mwi,  9)   # 33 bits -> 9 nybbles
    wr(os.path.join(VEC, 'peak_detector_bandpass_input.mem'),    bp,   6)   # 21 bits -> 6 nybbles
    wr(os.path.join(VEC, 'peak_detector_expected_peaks.mem'), peaks, 5)  # 20 bits -> 5 nybbles
    with open(os.path.join(VEC, 'peak_detector_meta.txt'), 'w') as f:
        f.write(f"{NSAMP}\n{len(peaks)}\n")

    # The learn-phase constants the RTL must reproduce, printed so a mismatch
    # can be localised to initialisation rather than to the detection loop.
    learn = min(360, len(mwi))
    smax = int(mwi[:learn].max())
    smean = int(mwi[:learn].sum()) // learn
    print(f"\n  LEARN PHASE (first {learn} samples) -- the RTL must match these:")
    print(f"    sum          {int(mwi[:learn].sum()):,}")
    print(f"    max          {smax:,}   -> SPKI = {smax >> 1:,}")
    print(f"    mean (floor) {smean:,}   -> NPKI = {smean >> 1:,}")
    print(f"    NOTE: Python uses int(np.mean(...)). For these magnitudes the")
    print(f"          float64 quotient is exact enough that int() == floor(),")
    print(f"          so integer division in RTL is equivalent. Verified by")
    print(f"          this test passing, not assumed.")
    npmean = int(np.mean(mwi[:learn]))
    assert npmean == smean, (
        f"int(np.mean)={npmean} != floor division {smean}. The RTL divider "
        f"cannot match np.mean here -- revisit the divider spec.")
    print(f"    int(np.mean) == floor(sum/{learn})  CONFIRMED")

    print(f"\n  wrote 4 files to {os.path.abspath(VEC)}")


if __name__ == '__main__':
    main()
