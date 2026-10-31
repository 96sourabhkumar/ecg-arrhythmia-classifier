"""
rr_features.py -- CANONICAL integer specification for rr_feature_engine.v.

This is the golden model. rr_feature_engine.v must reproduce rr_int8() bit-exactly.

WHY THIS EXISTS SEPARATELY FROM mitbih.rr_features()
------------------------------------------------------------
mitbih.rr_features() computes four FLOAT ratios and feeds them to
training. Hardware has no float, so the RTL cannot be bit-exact against it --
which would have made rr_feature_engine.v the ONLY non-bit-exact module in the project.

Rather than change the training pipeline (which would put INT8 values in
DS*_RR.npy, break train.py's clip(RR,-5,5), and force a retrain), the hardware
gets its own integer specification. The cost of that choice was MEASURED, not
assumed, on the corrected quantisation path:

    archive/results/rr_int8_study.json      49,692 DS2 beats
      identical INT8 vectors        92.17%
      largest disagreement          1 INT8 unit
      accuracy      81.57% -> 81.60%      (+0.03 pp)
      macro-F1      0.4721 -> 0.4719      (-0.0002)

-0.0002 is an order of magnitude below the -0.0211 that INT8 quantisation
itself costs, so the integer spec is adopted.

    NOTE: an earlier version of this study reported "+0.0003, adopt" while both
    arms ran the BROKEN quantisation path (13.83% accuracy). That result was
    void. These numbers are from the corrected path.

THE ARITHMETIC, AND WHY IT IS SHAPED THIS WAY
---------------------------------------------
Producing a ratio and then quantising it needs TWO roundings and cannot be made
bit-exact. Instead the INT8 value is produced directly with ONE division:

    rr_int = clip( ((v * K) / d + 2^(SH-1)) >> SH , -RR_CLIP, +RR_CLIP )

    K       = round(2^SH / s_rr)     compile-time constant
    RR_CLIP = round(5 / s_rr)        replicates train.py's clip(RR, -5, 5)

RR_CLIP IS NOT OPTIONAL. train.py clips the RATIO to +/-5 before the model ever
sees it. Omitting that clip produced a 109-INT8-unit disagreement on post/local
at record boundaries where `local` collapses: the float path gives 206 -> 5 ->
18, an unclipped integer path gives 726 -> 127.

NO MULTIPLIER. K is a constant, so `v * K` is a constant-coefficient multiply
that synthesises to shift-and-add. Every division is by a runtime value and
uses a sequential restoring divider. rr_feature_engine.v must contribute ZERO DSPs --
the front end owns exactly one multiplier (the squarer), and that is what the
whole sharing result rests on.

LATENCY. `post` looks one beat into the FUTURE, so features for beat i cannot
be emitted until peak i+1 arrives -- roughly 800 ms. Inherent to the feature
definition, not to the hardware. frontend_top.v must be built around it.
"""
import numpy as np

SH = 16                     # fixed-point shift


def constants(s_rr):
    """(K, RR_CLIP) for a given RR quantisation scale."""
    return int(round((1 << SH) / s_rr)), int(round(5.0 / s_rr))


def _scaled(v, d, K, clip):
    """((v*K)/d + 2^(SH-1)) >> SH, clipped. Floor division, v,d >= 0."""
    if d <= 0:
        d = 1
    clip = min(clip, 127)   # FIX 2026-10-08: INT8 saturation, matches rr_feature_engine.v fin() (v6 clip=339 > 127 used to wrap negative)
    r =((int(v) * K) // int(d) + (1 << (SH - 1))) >> SH
    return max(-clip, min(clip, r))


def rr_int8(peaks, i, K, clip, fs=180):
    """
    Four INT8 RR features for beat i. `peaks` are 180 Hz sample indices.

    Mirrors mitbih.rr_features() exactly in its choice of windows:
        pre   = peaks[i]   - peaks[i-1]
        post  = peaks[i+1] - peaks[i]
        local = mean of the last min(i,8) RR intervals ending at i
        glob  = mean of ALL i RR intervals so far
    with the float means replaced by FLOOR integer division.
    """
    peaks = np.asarray(peaks, dtype=np.int64)
    n = len(peaks)
    pre  = int(peaks[i] - peaks[i-1]) if i > 0     else 0
    post = int(peaks[i+1] - peaks[i]) if i < n-1   else 0

    lo = max(0, i - 8)
    if i - lo >= 1:
        d = np.diff(peaks[lo:i+1])
        local = int(d.sum()) // len(d)          # FLOOR, not float mean
    else:
        local = pre if pre else 1
    if i >= 1:
        d = np.diff(peaks[:i+1])
        glob = int(d.sum()) // len(d)
    else:
        glob = pre if pre else 1
    local = local if local else 1
    glob  = glob  if glob  else 1

    return np.array([_scaled(pre,   local, K, clip),
                     _scaled(post,  local, K, clip),
                     _scaled(local, glob,  K, clip),
                     _scaled(glob,  fs,    K, clip)], dtype=np.int64)


def rr_int8_stream(peaks, K, clip, fs=180):
    """All beats. Returns (n,4) INT8."""
    return np.stack([rr_int8(peaks, i, K, clip, fs)
                     for i in range(len(peaks))]).astype(np.int64)
