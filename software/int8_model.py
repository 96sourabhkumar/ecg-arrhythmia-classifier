"""
int8_model.py -- bit-exact INT8 reference model of the FPGA CNN (hardware/rtl/cnn/).

load_weights() reads the three files quantize.py writes (cnn_params.vh, cnn_weights.mem,
cnn_biases.mem) from QDIR -- the same files the RTL loads -- and make_classifier() returns
forward_int8(X, R), which reproduces the accelerator's integer arithmetic exactly:
conv -> +bias -> (acc * M_int + 2^(sh-1)) >> sh -> clip[0,127] -> max-pool(2), GAP via a
fixed-point multiplier, RR branch and FC layers, argmax.  verify_against_gold() re-classifies
the 64 golden beats used by the RTL testbenches and must report 64/64 before anything else
is trusted.

    import int8_model as M
    M.QDIR = 'hardware/weights'            # default
    Q, MQ, gmi, gsh = M.load_weights(); classify = M.make_classifier(Q, MQ, gmi, gsh)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = os.path.join(HERE, '..', 'hardware', 'weights')

WIN = 256
BEAT_PRE, BEAT_POST = 90, 166           # confirmed 2026-09-02 against
                                         # DS1_X256.npy[0] bit-exactly -- see
                                         # docs/development_log.md board-level TB entry
assert BEAT_PRE + BEAT_POST == WIN

# per-layer (cin, cout, k, ln, pad) at FULL resolution (ln = conv output
# length before pooling = input length, same-padding stride-1). Matches
# quantize.py's CH=(8,16,32,32) / pad = 2 if l<2 else 1, with WIN=256.
LAYERS = [(1, 8, 5, 256, 2), (8, 16, 5, 128, 2), (16, 32, 3, 64, 1), (32, 32, 3, 32, 1)]
CONV_KEYS = ['c1', 'c2', 'c3', 'c4']
FC_KEYS   = ['rr1', 'rr2', 'fc1', 'fc2']
ALL_KEYS  = CONV_KEYS + FC_KEYS
FC_DIMS = {'rr1': (4, 16), 'rr2': (16, 16), 'fc1': (48, 40), 'fc2': (40, 5)}  # (nin, nout)


# ---------------------------------------------------------------------------
def parse_consts_vh(path):
    """Pull woff/boff/mult/shift straight from the file cnn_accelerator.v
    itself includes -- see module header for why this beats recomputing
    from quant_params.json's floats."""
    txt = open(path).read()
    import re
    cfg = {}
    for i, k in enumerate(CONV_KEYS):
        woff = int(re.search(rf"cfg_woff\[{i}\]=(\d+)", txt).group(1))
        boff = int(re.search(rf"cfg_boff\[{i}\]=(\d+)", txt).group(1))
        mult = int(re.search(rf"cfg_mult\[{i}\]=16'd(\d+)", txt).group(1))
        sh   = int(re.search(rf"cfg_sh\[{i}\]=6'd(\d+)", txt).group(1))
        cfg[k] = dict(woff=woff, boff=boff, mult=mult, sh=sh)
    for i, k in enumerate(FC_KEYS):
        woff = int(re.search(rf"fcs_woff\[{i}\]=(\d+)", txt).group(1))
        boff = int(re.search(rf"fcs_boff\[{i}\]=(\d+)", txt).group(1))
        mult = int(re.search(rf"fcs_mult\[{i}\]=16'd(\d+)", txt).group(1))
        sh   = int(re.search(rf"fcs_sh\[{i}\]=6'd(\d+)", txt).group(1))
        cfg[k] = dict(woff=woff, boff=boff, mult=mult, sh=sh)
    gmi = int(re.search(r"gap_mult=16'd(\d+)", txt).group(1))
    gsh = int(re.search(r"gap_sh=6'd(\d+)", txt).group(1))
    return cfg, gmi, gsh


def read_hex_flat(path, bits, signed):
    lim = 1 << (bits - 1)
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16)
            if signed and v >= lim:
                v -= (1 << bits)
            out.append(v)
    return np.array(out, dtype=np.int64)


def load_weights():
    cfg, gmi, gsh = parse_consts_vh(os.path.join(QDIR, 'cnn_params.vh'))
    w_flat = read_hex_flat(os.path.join(QDIR, 'cnn_weights.mem'), 8, signed=True)
    b_flat = read_hex_flat(os.path.join(QDIR, 'cnn_biases.mem'), 24, signed=True)

    Q = {}
    for cin, cout, k, ln, pad in LAYERS:
        pass  # dims consumed below via zip with CONV_KEYS
    for i, key in enumerate(CONV_KEYS):
        cin, cout, k, ln, pad = LAYERS[i]
        n = cout * cin * k
        woff = cfg[key]['woff']
        Q[f'{key}_W'] = w_flat[woff:woff + n].reshape(cout, cin, k)
        boff = cfg[key]['boff']
        Q[f'{key}_b'] = b_flat[boff:boff + cout]
    for key in FC_KEYS:
        nin, nout = FC_DIMS[key]
        n = nout * nin
        woff = cfg[key]['woff']
        Q[f'{key}_W'] = w_flat[woff:woff + n].reshape(nout, nin)
        boff = cfg[key]['boff']
        Q[f'{key}_b'] = b_flat[boff:boff + nout]

    MQ = {k: (cfg[k]['mult'], cfg[k]['sh']) for k in CONV_KEYS + ['rr1', 'rr2', 'fc1']}
    return Q, MQ, gmi, gsh


# ---------------------------------------------------------------------------
# Forward pass -- copied verbatim in form from quantize.py's forward_int8()
# (the function already proven bit-exact against the RTL by tb_cnn_accelerator.v).
def conv_int(a, W, bq, pad, Mint, Msh):
    B, Cin, L = a.shape
    Cout, _, K = W.shape
    ap = np.pad(a, ((0, 0), (0, 0), (pad, pad)))
    acc = np.zeros((B, Cout, L), dtype=np.int64)
    for k in range(K):
        acc += np.einsum('bil,oi->bol', ap[:, :, k:k + L], W[:, :, k],
                         optimize=True).astype(np.int64)
    acc += bq[None, :, None]
    out = np.clip((acc * Mint + (1 << (Msh - 1))) >> Msh, 0, 127)
    return out[:, :, :L // 2 * 2].reshape(B, Cout, L // 2, 2).max(axis=3)


def fc_int(a, W, bq, Mint=None, Msh=None, relu=True):
    acc = a.astype(np.int64) @ W.T.astype(np.int64) + bq[None, :]
    if Mint is None:
        return acc
    out = (acc * Mint + (1 << (Msh - 1))) >> Msh
    return np.clip(out, 0, 127) if relu else out


def make_classifier(Q, MQ, gmi, gsh):
    def forward_int8(X, R):
        """X: (B,1,256) int32 already-quantised INT8 beat window.
        R: (B,4) int32 already-quantised INT8 RR features.
        Returns (B,) predicted class 0..4."""
        a = X
        for l, k in enumerate(CONV_KEYS):
            a = conv_int(a, Q[f'{k}_W'], Q[f'{k}_b'], 2 if l < 2 else 1, *MQ[k])
        gap_sum  = a.sum(axis=2, dtype=np.int64)
        gap_mean = gap_sum >> 4
        round_bias = (1 << (gsh - 1)) if gsh > 0 else 0
        gap_q = np.clip((gap_mean * gmi + round_bias) >> gsh, -127, 127).astype(np.int32)
        r = R
        r = fc_int(r, Q['rr1_W'], Q['rr1_b'], *MQ['rr1'])
        r = fc_int(r, Q['rr2_W'], Q['rr2_b'], *MQ['rr2'])
        h = np.concatenate([gap_q, r], axis=1)
        h = fc_int(h, Q['fc1_W'], Q['fc1_b'], *MQ['fc1'])
        logits = fc_int(h, Q['fc2_W'], Q['fc2_b'])
        return logits.argmax(axis=1)
    return forward_int8


def verify_against_gold(classify):
    """Reload the SAME 64 beats tb_cnn_accelerator.v already verified bit-exact
    (golden_beats.mem/golden_rr_words.mem/golden_classes.mem) and re-classify them
    with THIS script's freshly-loaded weights. Any mismatch here means this
    script's weight loading / forward pass has a bug independent of anything
    detector-related -- must be 64/64 before trusting anything below."""
    gb = read_hex_flat(os.path.join(QDIR, 'golden_beats.mem'), 8, signed=True)
    gr = read_hex_flat(os.path.join(QDIR, 'golden_rr_words.mem'), 8, signed=True)
    gc = np.array([int(x.strip()) for x in
                   open(os.path.join(QDIR, 'golden_classes.mem')) if x.strip()])
    n = len(gc)
    X = gb.reshape(n, 1, WIN).astype(np.int32)
    R = gr.reshape(n, 4).astype(np.int32)
    pred = classify(X, R)
    ok = int((pred == gc).sum())
    print(f"  self-check vs golden_classes.mem: {ok}/{n} match "
          f"({'PASS' if ok == n else '*** FAIL -- DO NOT TRUST BELOW ***'})")
    if ok != n:
        bad = np.where(pred != gc)[0]
        print(f"    mismatches at indices: {bad[:10].tolist()}")
        sys.exit(1)
