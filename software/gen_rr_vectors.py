"""
gen_rr_vectors.py -- vectors for tb_rr_feature_engine.v

Feeds rr_feature_engine.v a real peak train from record 100 and dumps the four INT8
features the golden model (qrs/rr_features.py) produces for each beat.

WHAT THE RTL CAN AND CANNOT EMIT
--------------------------------
post_i needs peaks[i+1], so beat i is emitted when peak i+1 arrives. The RTL
also needs peaks[i-1] for pre_i, so the first beat it can emit is **i = 1**.
Beat 0 is never emitted, which is correct: its golden features are degenerate
(pre = 0, local = 1, glob = 1) and mitbih would discard that beat anyway
if its 128-sample window fell off the record start.

The expected file therefore starts at beat 1, and the testbench compares in
that order.

WHY A REAL PEAK TRAIN AND NOT SYNTHETIC INTERVALS
-------------------------------------------------
`local` is a mean over a sliding 8-interval window and `glob` a mean over all
intervals so far, so both depend on the entire history. Uniform synthetic
intervals would make local == glob == pre for every beat, and f0/f1/f2 would all
collapse to the same constant -- a test that cannot distinguish a correct engine
from one that computes a single ratio and copies it four times.

Real ECG has RR variation, so the four features genuinely differ. The generator
ASSERTS that they do.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mitbih import load_record
from qrs.pan_tompkins import pan_tompkins_fixed
from qrs.rr_features import rr_int8_stream, constants

VEC = os.path.join(HERE, '..', 'hardware', 'sim', 'vectors')
HEX = os.path.join(HERE, '..', 'hardware', 'weights')
RECORD = int(sys.argv[1]) if len(sys.argv) > 1 else 100
NREAL  = int(sys.argv[2]) if len(sys.argv) > 2 else 24   # real-beat prefix


def wr(path, arr, nyb):
    with open(path, 'w') as f:
        for v in np.asarray(arr).flatten():
            f.write(f"{int(v) & ((1 << (4*nyb)) - 1):0{nyb}x}\n")


def main():
    s_rr = None
    for line in open(os.path.join(HEX, 'rr_input_scales.txt')):
        if line.startswith('#') or not line.strip():
            continue
        k_, v_ = line.split()
        if k_ == 'rr':
            s_rr = float(v_)
    assert s_rr is not None, "input_scales.txt has no 'rr' entry -- run quantize_v2.py"
    K, CLIP = constants(s_rr)
    print(f"record {RECORD}")
    print(f"  s_rr    {s_rr:.6e}")
    print(f"  K       {K:,}      RR_CLIP {CLIP}")
    print(f"  these MUST match rr_feature_engine.v's parameters exactly")

    adc, _, _ = load_record(RECORD)
    peaks, _ = pan_tompkins_fixed(adc)
    peaks = np.asarray(peaks, dtype=np.int64)

    # ---------------------------------------------------------------------
    # STIMULUS DESIGN -- verified against the golden model before use
    #
    # Real ECG alone is NOT sufficient. Record 100 is healthy sinus rhythm:
    # RR sits in 114..179 samples, every ratio lands near 1.0, and at
    # s_rr = 0.2836 all four features quantise into {3,4,5} with f3 CONSTANT.
    # An engine emitting a fixed 4 on all four outputs would pass.
    #
    # f3 = glob/FS is the hard one: `glob` is a mean over ALL intervals so
    # far, so it only moves if the record is DOMINATED by extremes. A short
    # stress tail appended to 200 real beats barely shifts it -- that is why
    # the first attempt still had f3 constant.
    #
    # So: a SHORT real prefix (realistic warm-up) then long homogeneous runs
    # that drag `glob` across its range, then rapid alternation to decouple
    # `local` from `glob`.
    #
    #   glob=40 -> f3=1     glob=130 -> f3=3     glob=240 -> f3=5
    #   glob=90 -> f3=2     glob=180 -> f3=4     glob=300 -> f3=6
    # ---------------------------------------------------------------------
    peaks = peaks[:NREAL]
    rr_stress = np.array(
        [300]*70 +                       # drag glob UP
        [45]*70 +                        # drag glob DOWN
        [300]*40 + [45]*40 +             # again, wider swing
        [40,300,40,300,                  # alternation: local != glob
         60,240,60,240,
         120,130,110,125,
         300,40,200,80], dtype=np.int64)
    tail = peaks[-1] + np.cumsum(rr_stress)
    peaks = np.concatenate([peaks, tail])
    n = len(peaks)
    print(f"  real prefix  {NREAL} peaks")
    print(f"  stress tail  {len(rr_stress)} intervals, RR {rr_stress.min()}..{rr_stress.max()} "
          f"({60*180/rr_stress.max():.0f}..{60*180/rr_stress.min():.0f} bpm)")
    print(f"\n  peaks used   {n}")
    rr = np.diff(peaks)
    print(f"  RR interval  {rr.min()} .. {rr.max()} samples "
          f"({60*180/rr.max():.0f} .. {60*180/rr.min():.0f} bpm)")

    feats = rr_int8_stream(peaks, K, CLIP)          # (n,4)

    # the RTL emits beats 1 .. n-2  (beat 0 has no pre, beat n-1 has no post)
    lo, hi = 1, n - 1
    exp = feats[lo:hi]
    print(f"  beats emitted by the RTL: {lo} .. {hi-1}  ({len(exp)} beats)")
    print(f"  feature ranges  f0[{exp[:,0].min()},{exp[:,0].max()}] "
          f"f1[{exp[:,1].min()},{exp[:,1].max()}] "
          f"f2[{exp[:,2].min()},{exp[:,2].max()}] "
          f"f3[{exp[:,3].min()},{exp[:,3].max()}]")

    # A test where every feature is the same number cannot distinguish a correct
    # engine from one that computes one ratio and copies it four times.
    distinct = len(set(map(tuple, exp.tolist())))
    print(f"  distinct feature vectors: {distinct} of {len(exp)}")
    nclip = int((np.abs(exp) == CLIP).sum())
    print(f"  values at the RR_CLIP boundary ({CLIP}): {nclip}   "
          f"{'OK - saturation exercised' if nclip else 'NOT exercised'}")

    # A stimulus where the features barely move cannot tell a correct engine
    # from one that computes a single ratio and copies it four times.
    for j in range(4):
        assert exp[:, j].min() != exp[:, j].max(), (
            f"feature f{j} is CONSTANT across every beat. An engine that "
            f"ignored f{j} entirely would pass. Widen the stress tail.")
    assert distinct >= 12, (
        f"only {distinct} distinct feature vectors -- too uniform to "
        f"discriminate. Widen the stress tail.")
    assert nclip > 0, (
        "no value reaches RR_CLIP, so the saturation path is never taken. "
        "That is the branch most likely to be wrong.")
    print("  OK -- all four features vary, and saturation is exercised")

    os.makedirs(VEC, exist_ok=True)
    wr(os.path.join(VEC, 'rr_engine_peak_input.mem'), peaks, 5)      # 20-bit indices
    wr(os.path.join(VEC, 'rr_engine_expected.mem'),   exp,   2)      # INT8, 4 per beat
    with open(os.path.join(VEC, 'rr_engine_meta.txt'), 'w') as f:
        f.write(f"{n}\n{len(exp)}\n{K}\n{CLIP}\n")

    print(f"\n  wrote rr_engine_peak_input.mem ({n}), rr_engine_expected.mem ({len(exp)}x4), rr_engine_meta.txt")
    print(f"  to {os.path.abspath(VEC)}")


if __name__ == '__main__':
    main()
