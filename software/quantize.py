"""
quantize.py -- INT8 export of a trained ECGNet for the FPGA accelerator.

Follows the scheme first proven in archive/python/quantize_v2.py exactly:
    act_int[0]   = clip(round(z / s_x), -127, 127)
    w_int[l]     = clip(round(w[l] / s_w[l]), -127, 127)
    b_int[l]     = round(b[l] / (s_in[l] * s_w[l]))         <- accumulator scale
    acc[l]       = conv(act_int, w_int) + b_int             <- INT32
    act_int[l+1] = clip(round(acc * M[l]), 0, 127)          <- ReLU folded in
    M[l]         = (s_in[l] * s_w[l]) / s_out[l]
Activation scales are calibrated on DS1 ONLY (a high percentile, not the max,
so a single outlier beat cannot compress the scale for every other beat).
DS2 is never used for calibration -- that would leak the test set into the
quantisation constants.

WHAT IS NEW vs the first (Build B) quantiser
-----------------------------
1. BATCHNORM FOLDING. The network trains with BN after every conv and after
   rr1. BN is a pure affine map at inference, so it folds into the preceding
   layer's weights and bias:
        W' = W * gamma / sqrt(var + eps)
        b' = (b - mean) * gamma / sqrt(var + eps) + beta
   After folding, the arithmetic the RTL must implement is IDENTICAL in form
   to the old design (conv -> bias -> ReLU -> pool). BN therefore costs ZERO
   hardware -- it only changed how the weights were learned.

2. THE RR BRANCH. RR is no longer concatenated raw; it goes through
   rr1(4->16) -> BN -> ReLU -> rr2(16->16) -> ReLU. Both are quantised as
   ordinary INT8 fully-connected layers.

3. THE CONCATENATION SUBTLETY (unchanged in spirit, new in detail).
   feat = concat(GAP(32), rr_out(16)) feeds fc1, so both halves MUST share one
   scale or fc1 sees mismatched units. rr2's output scale is forced equal to
   the GAP output scale for exactly this reason.

4. 256-SAMPLE WINDOW. Beat buffer is 256 deep (still a power of two, so the
   RTL address counter stays a wrap-around counter -- 8-bit now, was 7-bit).

Run:  python quantize.py          (inputs/outputs can be overridden, see QMODEL/QOUT/QCALIB/QTEST)
Writes to hardware/weights/:
    quant_params.json               all scales / multipliers / shifts (also read by the PYNQ driver)
    cnn_weights.mem, cnn_biases.mem hex weight + bias memory images for $readmemh
    cnn_params.vh                   per-layer offsets, requant multipliers and shifts (RTL include)
    golden_beats.mem, golden_rr_words.mem, golden_classes.mem
                                    64 test beats + the INT8 class the RTL must produce
    int8_report.txt                 INT8-vs-float accuracy on the report split
"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C

PROJ = os.path.join(HERE, '..')
DATA = os.path.join(PROJ, 'data', 'processed')
# Inputs are parameterised through environment variables:
#   QMODEL  checkpoint            (default results/models/ecgnet_patient_adaptive.pt)
#   QOUT    output directory      (default hardware/weights)
#   QCALIB  calibration npz       (default data/beats/PA_train.npz -- training data only)
#   QTEST   report npz            (default data/beats/PA_test.npz)
MODELS = os.path.join(PROJ, 'results', 'models'); D7 = os.path.join(PROJ, 'data', 'beats')
MODEL = os.environ.get('QMODEL', os.path.join(MODELS, 'ecgnet_patient_adaptive.pt'))
OUT = os.environ.get('QOUT', os.path.join(PROJ, 'hardware', 'weights')); os.makedirs(OUT, exist_ok=True)

WIN, N_RR = 256, 4
CH = (8, 16, 32, 32)
RR_HIDDEN, FC_HIDDEN = 16, 40
CALIB_PCT = 99.9          # percentile for activation calibration


# ---------------------------------------------------------------------------
def load256(split):
    f = os.environ.get('QCALIB', os.path.join(D7, 'PA_train.npz')) if split == 'DS1' else os.environ.get('QTEST', os.path.join(D7, 'PA_test.npz'))
    d = np.load(f); return d['X'].astype(np.float32), d['R'].astype(np.float32), d['y'].astype(np.int64)
def _old_load256(split):
    X = np.load(os.path.join(DATA, f'{split}_X256.npy')).astype(np.float32) / 256.0
    R = np.clip(np.load(os.path.join(DATA, f'{split}_RR256.npy')).astype(np.float32), -5, 5)
    y = np.load(os.path.join(DATA, f'{split}_y256.npy')).astype(np.int64)
    return X, R[:, :N_RR], y


def fold_bn(W, b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm into the preceding conv/linear. Returns (W', b')."""
    scale = bn_w / np.sqrt(bn_var + eps)
    if W.ndim == 3:                      # conv: (out, in, k)
        W2 = W * scale[:, None, None]
    else:                                # linear: (out, in)
        W2 = W * scale[:, None]
    b2 = (b - bn_mean) * scale + bn_b
    return W2, b2


def qscale(x, bits=8):
    """Symmetric per-tensor scale so that max|x| maps to 127."""
    m = float(np.max(np.abs(x)))
    return (m / 127.0) if m > 0 else 1.0


def qint8(x, s):
    return np.clip(np.round(x / s), -127, 127).astype(np.int32)


# ---------------------------------------------------------------------------
#  float reference (BN already folded) -- used to calibrate and to cross-check
# ---------------------------------------------------------------------------
def conv1d(x, W, b, pad):
    """x (C_in, L) -> (C_out, L). Same-padding, stride 1."""
    Cin, L = x.shape
    Cout, _, K = W.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    out = np.empty((Cout, L), dtype=np.float64)
    for o in range(Cout):
        acc = np.zeros(L)
        for i in range(Cin):
            for k in range(K):
                acc += W[o, i, k] * xp[i, k:k + L]
        out[o] = acc + b[o]
    return out


def maxpool2(x):
    C, L = x.shape
    return x[:, :L // 2 * 2].reshape(C, L // 2, 2).max(axis=2)


def forward_float(x, rr, P):
    a = x
    for l in range(4):
        a = conv1d(a, P[f'c{l+1}_W'], P[f'c{l+1}_b'], 2 if l < 2 else 1)
        a = maxpool2(np.maximum(a, 0))
    gap = a.mean(axis=1)                                    # (32,)
    r = np.maximum(P['rr1_W'] @ rr + P['rr1_b'], 0)
    r = np.maximum(P['rr2_W'] @ r + P['rr2_b'], 0)
    h = np.concatenate([gap, r])
    h = np.maximum(P['fc1_W'] @ h + P['fc1_b'], 0)
    return P['fc2_W'] @ h + P['fc2_b']


# ---------------------------------------------------------------------------
def main():
    if not os.path.exists(MODEL):
        print(f"FAIL: {MODEL} not found."); sys.exit(1)
    sd = torch.load(MODEL, map_location='cpu')
    g = lambda k: sd[k].numpy().astype(np.float64)

    # ---- fold BN into every conv and into rr1 --------------------------
    P = {}
    for l in range(1, 5):
        W, b = g(f'c{l}.weight'), g(f'c{l}.bias')
        W, b = fold_bn(W, b, g(f'b{l}.weight'), g(f'b{l}.bias'),
                       g(f'b{l}.running_mean'), g(f'b{l}.running_var'))
        P[f'c{l}_W'], P[f'c{l}_b'] = W, b
    W, b = fold_bn(g('rr1.weight'), g('rr1.bias'), g('rrb.weight'), g('rrb.bias'),
                   g('rrb.running_mean'), g('rrb.running_var'))
    P['rr1_W'], P['rr1_b'] = W, b
    for k in ('rr2', 'fc1', 'fc2'):
        P[f'{k}_W'], P[f'{k}_b'] = g(f'{k}.weight'), g(f'{k}.bias')
    print("BatchNorm folded into conv1-4 and rr1 (zero hardware cost).")

    X1, R1, y1 = load256('DS1')
    X2, R2, y2 = load256('DS2')

    # ---- calibrate activation scales on DS1 ONLY -----------------------
    idx = np.random.RandomState(0).choice(len(y1), 1500, replace=False)
    acts = {k: [] for k in ['x', 'a1', 'a2', 'a3', 'a4', 'gap', 'r1', 'r2', 'h1']}
    for i in idx:
        a = X1[i]
        acts['x'].append(np.abs(a).max())
        for l in range(4):
            a = conv1d(a, P[f'c{l+1}_W'], P[f'c{l+1}_b'], 2 if l < 2 else 1)
            a = maxpool2(np.maximum(a, 0))
            acts[f'a{l+1}'].append(np.abs(a).max())
        acts['gap'].append(np.abs(a.mean(axis=1)).max())
        r = np.maximum(P['rr1_W'] @ R1[i] + P['rr1_b'], 0); acts['r1'].append(np.abs(r).max())
        r = np.maximum(P['rr2_W'] @ r + P['rr2_b'], 0);     acts['r2'].append(np.abs(r).max())
        gapv = a.mean(axis=1)
        h = np.maximum(P['fc1_W'] @ np.concatenate([gapv, r]) + P['fc1_b'], 0)
        acts['h1'].append(np.abs(h).max())
    S = {k: float(np.percentile(v, CALIB_PCT)) / 127.0 for k, v in acts.items()}

    # CRITICAL: GAP and the RR branch output are concatenated into fc1, so they
    # MUST share one scale or fc1 sees mismatched units (this was fault 3 in
    # the original quantize.py). Force rr2's output scale to the GAP scale.
    S['r2'] = S['gap']
    print("activation scales (DS1-calibrated, p%.1f):" % CALIB_PCT)
    for k, v in S.items():
        print(f"   s_{k:4s} = {v:.6e}")

    # ---- quantise weights ----------------------------------------------
    Q, SW = {}, {}
    for k in ['c1', 'c2', 'c3', 'c4', 'rr1', 'rr2', 'fc1', 'fc2']:
        SW[k] = qscale(P[f'{k}_W'])
        Q[f'{k}_W'] = qint8(P[f'{k}_W'], SW[k])

    # RR input needs its OWN scale. Setting it to 1.0 rounds every RR feature
    # (they all sit near 1.0 after normalisation) to the same integer and
    # destroys the timing signal the S class depends on -- measured: Se(S)
    # fell to 7.7% and INT8 accuracy to 90.86%. Calibrate it like any other
    # activation, on DS1 only.
    S['rr'] = float(np.percentile(np.abs(R1), CALIB_PCT)) / 127.0
    print(f"   s_rr   = {S['rr']:.6e}   (RR input scale)")

    s_in = {'c1': S['x'], 'c2': S['a1'], 'c3': S['a2'], 'c4': S['a3'],
            'rr1': S['rr'], 'rr2': S['r1'], 'fc1': S['gap'], 'fc2': S['h1']}
    s_out = {'c1': S['a1'], 'c2': S['a2'], 'c3': S['a3'], 'c4': S['a4'],
             'rr1': S['r1'], 'rr2': S['r2'], 'fc1': S['h1'], 'fc2': 1.0}
    for k in Q_KEYS:
        Q[f'{k}_b'] = np.round(P[f'{k}_b'] / (s_in[k] * SW[k])).astype(np.int32)

    M = {k: (s_in[k] * SW[k]) / s_out[k] for k in Q_KEYS if k != 'fc2'}
    # fixed-point form of M: M ~= M_int / 2^shift, M_int in [2^14, 2^15)
    MQ = {}
    for k, m in M.items():
        sh = 0
        while m * (2 ** sh) < 16384 and sh < 31:
            sh += 1
        MQ[k] = (int(round(m * (2 ** sh))), sh)
    print("\nrequant multipliers  M = (s_in*s_w)/s_out  ->  (M_int, shift)")
    for k in MQ:
        print(f"   {k:4s} M={M[k]:.6e}  ->  {MQ[k]}")

    # GAP rescale constants, computed HERE (not later) so forward_int8() below
    # can use the exact same fixed-point (gmi, gsh) the RTL uses -- see the
    # bug note at the top of the GAP block in forward_int8().
    gm = S['a4'] / S['gap']
    gsh = 0
    while gm * (2 ** gsh) < 16384 and gsh < 31:
        gsh += 1
    gmi = int(round(gm * (2 ** gsh)))

    json.dump({'S': S, 'SW': SW, 'M': {k: list(v) for k, v in MQ.items()},
               'win': WIN, 'n_rr': N_RR, 'ch': list(CH),
               'rr_hidden': RR_HIDDEN, 'fc_hidden': FC_HIDDEN},
              open(os.path.join(OUT, 'quant_params.json'), 'w'), indent=1)
    print(f"\nwrote {os.path.join(OUT, 'quant_params.json')}")

    # ---- INT8 inference, EXACTLY as the RTL will do it -----------------
    def conv_int(a, W, bq, pad, Mint, Msh):
        """a (B,Cin,L) int32 -> (B,Cout,L/2) int32, ReLU+pool+requant folded."""
        B, Cin, L = a.shape
        Cout, _, K = W.shape
        ap = np.pad(a, ((0, 0), (0, 0), (pad, pad)))
        acc = np.zeros((B, Cout, L), dtype=np.int64)
        for k in range(K):
            # (B,Cin,L) x (Cout,Cin) -> (B,Cout,L)
            acc += np.einsum('bil,oi->bol', ap[:, :, k:k + L], W[:, :, k],
                             optimize=True).astype(np.int64)
        acc += bq[None, :, None]
        # requant + ReLU (saturating, never wrapping -- config.SATURATE).
        # UPDATED 2026-09-02 (again): the conv path's RTL used to be
        # requant_relu_pool_shared.v, reused verbatim from Build B/C, which
        # floors (no rounding bias) -- that forced this function to floor
        # too (measured: 93.86% INT8 accuracy, see docs/development_log.md 2.5.6
        # bug 6). The conv path now instead uses requant_relu_pool.v, a
        # dedicated module that rounds (does NOT touch the shared file, so the
        # Build A/B/C/ABN comparison is still unaffected). This function
        # rounds again to match. If cnn_accelerator.v's u_rq instantiation ever
        # reverts to requant_relu_pool_shared.v, this must revert too, or the
        # golden vectors stop being bit-exact with the RTL.
        out = np.clip((acc * Mint + (1 << (Msh - 1))) >> Msh, 0, 127)
        return out[:, :, :L // 2 * 2].reshape(B, Cout, L // 2, 2).max(axis=3)

    def fc_int(a, W, bq, Mint=None, Msh=None, relu=True):
        acc = a.astype(np.int64) @ W.T.astype(np.int64) + bq[None, :]
        if Mint is None:
            return acc
        out = (acc * Mint + (1 << (Msh - 1))) >> Msh
        return np.clip(out, 0, 127) if relu else out

    def forward_int8(X, R, bs=2048):
        preds = []
        for i in range(0, len(X), bs):
            xb, rb = X[i:i + bs], R[i:i + bs]
            a = np.clip(np.round(xb / S['x']), -127, 127).astype(np.int32)
            for l, k in enumerate(['c1', 'c2', 'c3', 'c4']):
                a = conv_int(a, Q[f'{k}_W'], Q[f'{k}_b'], 2 if l < 2 else 1, *MQ[k])
            # GAP: BUG FIXED 2026-09-02. This used to be a.mean(axis=2) --
            # exact floating-point division -- while the RTL computes the mean
            # as an integer sum right-shifted by 4 (>>>4, i.e. FLOOR division,
            # not round-to-nearest) and only THEN applies the fixed-point
            # gap_mult/gap_sh rescale. Those are two different roundings; the
            # float version does not reproduce what the hardware computes, so
            # this array was never actually a valid "golden" reference for
            # the GAP stage despite this file's own header claiming bit-exact
            # integer arithmetic throughout. Replaced with the identical
            # integer sequence cnn_accelerator.v performs: sum -> >>4 (arithmetic,
            # matches Verilog's >>>) -> *gmi -> +round-bias -> >>gsh -> clip.
            gap_sum  = a.sum(axis=2, dtype=np.int64)         # (B,32) int
            gap_mean = gap_sum >> 4                          # arithmetic shift, matches RTL >>>4
            round_bias = (1 << (gsh - 1)) if gsh > 0 else 0
            gap_q = np.clip((gap_mean * gmi + round_bias) >> gsh, -127, 127).astype(np.int32)
            r = np.clip(np.round(rb / S['rr']), -127, 127).astype(np.int32)
            r = fc_int(r, Q['rr1_W'], Q['rr1_b'], *MQ['rr1'])
            r = fc_int(r, Q['rr2_W'], Q['rr2_b'], *MQ['rr2'])
            h = np.concatenate([gap_q, r], axis=1)
            h = fc_int(h, Q['fc1_W'], Q['fc1_b'], *MQ['fc1'])
            logits = fc_int(h, Q['fc2_W'], Q['fc2_b'])
            preds.append(logits.argmax(axis=1))
        return np.concatenate(preds)

    print("\nrunning INT8 inference on DS2 (this is what the RTL will compute)...")
    p_int = forward_int8(X2, R2)

    def stats(y, p):
        cm = np.zeros((5, 5), int)
        for t, q in zip(y, p):
            cm[t, q] += 1
        se, f1 = [], []
        for i in range(5):
            TP = cm[i, i]; FN = cm[i].sum() - TP; FP = cm[:, i].sum() - TP
            s = TP / (TP + FN) if TP + FN else 0.0
            pp = TP / (TP + FP) if TP + FP else 0.0
            se.append(s); f1.append(2 * s * pp / (s + pp) if s + pp else 0.0)
        return np.trace(cm) / cm.sum(), float(np.mean(f1[:4])), se, cm

    acc, mac, se, cm = stats(y2, p_int)
    lines = ["INT8 (hardware-exact), float RR rounded -- see evaluate.py for integer-RR bit-exact numbers",
             "=" * 52,
             f"  INT8 accuracy   : {100*acc:.2f}%",
             f"  INT8 macro-F1   : {mac:.4f}",
             "  (float reference: see train_*.json for this model)",
             "",
             f"  Se(N/S/V/F) = {100*se[0]:.2f} / {100*se[1]:.2f} / "
             f"{100*se[2]:.2f} / {100*se[3]:.2f}",
             "", "confusion (rows=TRUE):"]
    for i, c in enumerate(C.AAMI_CLASSES):
        lines.append(f"  {c}  " + " ".join(f"{v:7d}" for v in cm[i]))
    txt = "\n".join(lines)
    print("\n" + txt)
    open(os.path.join(OUT, 'int8_report.txt'), 'w').write(txt + "\n")

    # ---- weight/bias memory images for $readmemh ------------------------
    with open(os.path.join(OUT, 'cnn_weights.mem'), 'w') as fw:
        for k in Q_KEYS:
            for v in Q[f'{k}_W'].reshape(-1):
                fw.write(f"{int(v) & 0xFF:02x}\n")
    # Biases are emitted at the RTL's ACCW width (24 bits = 6 hex digits), NOT
    # 32. $readmemh into a 24-bit reg silently truncates a wider literal, which
    # happens to be correct for two's complement but hides real overflow.
    # Assert the values genuinely fit instead of relying on that.
    ACCW_RTL = 24
    lim = 1 << (ACCW_RTL - 1)
    for k in Q_KEYS:
        bmax = int(np.abs(Q[f'{k}_b']).max()) if Q[f'{k}_b'].size else 0
        assert bmax < lim, (f"bias for {k} needs more than {ACCW_RTL} bits "
                            f"({bmax} >= {lim}) -- widen ACCW in the RTL")
    with open(os.path.join(OUT, 'cnn_biases.mem'), 'w') as fb:
        for k in Q_KEYS:
            for v in Q[f'{k}_b'].reshape(-1):
                fb.write(f"{int(v) & ((1 << ACCW_RTL) - 1):06x}\n")
    nw = sum(Q[f'{k}_W'].size for k in Q_KEYS)
    nb = sum(Q[f'{k}_b'].size for k in Q_KEYS)
    print(f"\nwrote cnn_weights.mem ({nw} INT8 weights) and cnn_biases.mem ({nb} INT32 biases)")

    # ---- emit the RTL constants so they CANNOT drift from the model -------
    # A wrong constant in cnn_accelerator.v does not fail loudly; it silently
    # mis-scales a layer. Generating them here means "retrain -> rerun this"
    # is the only supported path.
    woff, boff = {}, {}
    aw, ab = 0, 0
    for k in Q_KEYS:
        woff[k], boff[k] = aw, ab
        aw += Q[f'{k}_W'].size
        ab += Q[f'{k}_b'].size
    gm = S['a4'] / S['gap']
    gsh = 0
    while gm * (2 ** gsh) < 16384 and gsh < 31:
        gsh += 1
    gmi = int(round(gm * (2 ** gsh)))

    v = ["// AUTO-GENERATED by quantize.py -- DO NOT EDIT BY HAND.",
         "// Regenerate after ANY retrain: `python quantize.py`.",
         "// Included by cnn_accelerator.v / tb_cnn_accelerator.v.",
         ""]
    for i, k in enumerate(['c1', 'c2', 'c3', 'c4']):
        v.append(f"cfg_woff[{i}]={woff[k]}; cfg_boff[{i}]={boff[k]};"
                 f" cfg_mult[{i}]=16'd{MQ[k][0]}; cfg_sh[{i}]=6'd{MQ[k][1]};")
    for i, k in enumerate(['rr1', 'rr2', 'fc1', 'fc2']):
        mi, ms = MQ.get(k, (0, 0))
        v.append(f"fcs_woff[{i}]={woff[k]}; fcs_boff[{i}]={boff[k]};"
                 f" fcs_mult[{i}]=16'd{mi}; fcs_sh[{i}]=6'd{ms};")
    v.append(f"gap_mult=16'd{gmi}; gap_sh=6'd{gsh};   // s_a4/s_gap = {gm:.6f}")
    open(os.path.join(OUT, 'cnn_params.vh'), 'w').write("\n".join(v) + "\n")
    print(f"wrote cnn_params.vh (RTL constants -- include this, do not retype them)")

    # ---- golden vectors for the RTL testbench ----------------------------
    # First 64 DS2 beats: quantised input, RR, and the INT8 class the RTL must
    # reproduce EXACTLY. Any mismatch is an RTL bug, not a rounding difference,
    # because forward_int8() above is bit-identical integer arithmetic.
    # Sample ACROSS classes. The first 64 DS2 beats are 63 N + 1 S (record 100
    # is almost entirely normal), which would let a broken RTL that always
    # outputs N score 63/64 and look correct.
    NV = 64
    rng = np.random.RandomState(0)
    sel = []
    for cls in range(5):
        idx_c = np.where(y2 == cls)[0]
        if len(idx_c):
            sel.extend(rng.choice(idx_c, min(len(idx_c), NV // 4), replace=False))
    sel = np.array(sorted(sel))[:NV]
    NV = len(sel)
    xq = np.clip(np.round(X2[sel] / S['x']), -127, 127).astype(int)
    rq = np.clip(np.round(R2[sel] / S['rr']), -127, 127).astype(int)
    p_int = p_int[sel]
    y2 = y2[sel]
    with open(os.path.join(OUT, 'golden_beats.mem'), 'w') as f:
        for b in range(NV):
            for s in xq[b, 0]:
                f.write(f"{int(s) & 0xFF:02x}\n")
    with open(os.path.join(OUT, 'golden_rr_words.mem'), 'w') as f:
        for b in range(NV):
            for s in rq[b]:
                f.write(f"{int(s) & 0xFF:02x}\n")
    with open(os.path.join(OUT, 'golden_classes.mem'), 'w') as f:
        for c in p_int[:NV]:
            f.write(f"{int(c)}\n")
    print(f"wrote golden vectors for {NV} beats "
          f"(golden_beats.mem / golden_rr_words.mem / golden_classes.mem)")
    print(f"  expected class histogram: {np.bincount(p_int[:NV], minlength=5).tolist()}")


Q_KEYS = ['c1', 'c2', 'c3', 'c4', 'rr1', 'rr2', 'fc1', 'fc2']

if __name__ == '__main__':
    main()
