"""
mitbih.py — one command, from nothing to training-ready arrays.

    python mitbih.py --download      # first time only (~100 MB)
    python mitbih.py                 # build beats from local files

Produces in  data/processed/ :
    DS1_X.npy   (n_beats, 128)   int16   beat windows, 180 Hz
                                 (AUGMENTED: each true beat contributes 1
                                 exact-center copy + AUG_COPIES jittered/
                                 noisy copies -- see AUG_* constants above)
    DS1_RR.npy  (n_beats, 4)     float32 RR features
    DS1_y.npy   (n_beats,)       int8    AAMI class index 0..4
    DS2_*.npy   same, for the held-out test patients
                                 (NOT augmented -- exact annotated peak only,
                                 unchanged from the original protocol, so the
                                 inter-patient accuracy number stays honest)
    qrs_report.csv               per-record Se/PPV of the QRS detector

Every step prints PASS/FAIL. If anything says FAIL, stop and read it.
"""
import argparse, os, sys, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from qrs.pan_tompkins import pan_tompkins_fixed, FS

HERE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(HERE, '..', 'data', 'mitdb')
OUT  = os.path.join(HERE, '..', 'data', 'processed')
ALL_RECORDS = sorted(C.DS1 + C.DS2 + C.EXCLUDED_RECORDS)

# ---------------------------------------------------------------------------
# POSITION-JITTER / NOISE AUGMENTATION  [2026-09-02 addition]
#
# Measured on real hardware (see docs/development_log.md 2026-09-01 session):
# the QRS detector's peak position differs from the ground-truth annotation
# by std ~5.3 samples (easy record 100) up to ~11.5 samples (noisy record
# 108), at 180 Hz. build()'s original beat extraction ALWAYS windows at the
# exact annotated peak (offset 0) -- the CNN therefore never sees a
# realistically mis-centered beat during training, and true end-to-end
# (detector+CNN) accuracy collapses on noisy records (108: 1.10%, 203:
# 31.21%) even though annotation-centered inter-patient accuracy is 82.57%.
#
# Fix: augment DS1 (TRAINING beats only) with extra copies windowed at a
# randomly jittered offset (matching the measured detector-jitter range)
# plus light additive noise. DS2 (test) is NEVER touched -- augmenting the
# test set would hide the real inter-patient accuracy number, not fix
# anything.
#
# This is a design choice, not a proven-optimal one: jitter is drawn
# uniformly rather than matched to each record's true empirical jitter
# distribution (which is record-dependent and only measurable via the
# actual RTL detector, i.e. after synthesis). Report actual measured
# before/after numbers, do not assume this closes the gap.
# ---------------------------------------------------------------------------
AUG_JITTER_MAX  = 10     # samples, +/- (matches measured detector jitter range)
AUG_NOISE_STD   = 0.08   # fraction of window std, added post-normalization
AUG_COPIES      = 2      # extra jittered+noisy copies per beat (on top of the
                          # original, exactly-centered, noise-free copy)
AUG_RNG_SEED    = 1234


# --------------------------------------------------------------------------
def download():
    """Fetch all 48 records from PhysioNet. ~100 MB, a few minutes."""
    import urllib.request
    os.makedirs(RAW, exist_ok=True)
    base = 'https://physionet.org/files/mitdb/1.0.0/'
    for i, r in enumerate(ALL_RECORDS, 1):
        for ext in ('hea', 'dat', 'atr'):
            f = f'{r}.{ext}'
            p = os.path.join(RAW, f)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                continue
            for attempt in range(5):
                try:
                    urllib.request.urlretrieve(base + f, p)
                    break
                except Exception as e:
                    if attempt == 4:
                        print(f"  FAIL {f}: {e}")
        print(f"  [{i:2d}/48] record {r}", flush=True)
    print("download complete")


# --------------------------------------------------------------------------
def load_record(rec):
    """Read one record. Returns (adc_int_360Hz, peak_samples, symbols)."""
    import wfdb
    p = os.path.join(RAW, str(rec))
    sig = wfdb.rdrecord(p)
    ann = wfdb.rdann(p, 'atr')
    # Select MLII BY NAME, never by index.
    # Records 102, 104 and 114 do not have MLII on channel 0. Hardcoding
    # ch=0 cost record 114 a measured Se of 67.91% instead of 99.36%.
    ch = sig.sig_name.index('MLII') if 'MLII' in sig.sig_name else 0
    adc = sig.p_signal[:, ch]
    adc = np.round(adc * sig.adc_gain[ch]).astype(np.int64)
    return adc, ann.sample, np.array(ann.symbol)


def match_peaks(det, ann, tol):
    """Greedy one-to-one match. Returns TP, FP, FN."""
    used = np.zeros(len(det), bool)
    TP = 0
    for a in ann:
        if len(det) == 0:
            break
        d = np.abs(det - a).astype(float)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            TP += 1
            used[j] = True
    return TP, len(det) - TP, len(ann) - TP


def rr_features(peaks, i):
    """4 RR features for beat i, normalised by the local rhythm."""
    pre  = peaks[i]   - peaks[i-1] if i > 0             else 0
    post = peaks[i+1] - peaks[i]   if i < len(peaks)-1  else 0
    lo = max(0, i - 8)
    local = np.diff(peaks[lo:i+1]).mean() if i - lo >= 1 else (pre or 1)
    glob  = np.diff(peaks[:i+1]).mean()   if i >= 1      else (pre or 1)
    local = local or 1.0
    glob  = glob or 1.0
    return np.array([pre/local, post/local, local/glob, glob/FS],
                    dtype=np.float32)


# --------------------------------------------------------------------------
def build(split_name, records, augment=False):
    rng = np.random.RandomState(AUG_RNG_SEED)
    X, RR, Y, rows = [], [], [], []
    for rec in records:
        try:
            adc, ann_s, ann_sym = load_record(rec)
        except Exception as e:
            print(f"  record {rec}: LOAD FAIL ({e})")
            continue

        # Take the decimated signal FROM the detector itself, so the beats the
        # CNN sees are bit-identical to what the hardware front-end produces.
        # Previously this used adc[::2] (naive sample dropping = aliasing)
        # while the detector halfband-filtered — two different signals.
        det, _, stages = pan_tompkins_fixed(adc, return_stages=True)
        x180 = stages['dec']

        beat = np.array([s in C.SYMBOL_TO_AAMI for s in ann_sym])
        ann180 = (ann_s[beat] // 2).astype(np.int64)
        labels = np.array([C.SYMBOL_TO_AAMI[s] for s in ann_sym[beat]])

        TP, FP, FN = match_peaks(det, ann180, int(0.150 * FS))
        se  = 100*TP/(TP+FN) if TP+FN else 0
        ppv = 100*TP/(TP+FP) if TP+FP else 0
        rows.append([rec, len(ann180), len(det), TP, FP, FN,
                     round(se, 2), round(ppv, 2)])
        print(f"  {rec}: beats={len(ann180):5d} Se={se:6.2f}% PPV={ppv:6.2f}%",
              flush=True)

        # --- extract beats at the ANNOTATED peaks -------------------------
        # Training on annotated peaks isolates classifier performance from
        # detector performance. The detector is evaluated separately above,
        # and end-to-end (detector + classifier) is measured on hardware.
        for i, p in enumerate(ann180):
            n_copies = 1 + (AUG_COPIES if augment else 0)
            rr_i = rr_features(ann180, i)
            for copy_idx in range(n_copies):
                # copy 0 is always the original, exactly-centered, noise-free
                # window -- identical to the pre-augmentation behaviour.
                offset = 0
                if copy_idx > 0:
                    offset = int(rng.randint(-AUG_JITTER_MAX, AUG_JITTER_MAX + 1))
                a, b = p + offset - C.BEAT_PRE, p + offset + C.BEAT_POST
                if a < 0 or b > len(x180):
                    continue
                w = x180[a:b].astype(np.float64)
                w = w - w.mean()
                s = w.std()
                w = w / s if s > 1e-6 else w
                if copy_idx > 0:
                    w = w + rng.normal(0.0, AUG_NOISE_STD, size=w.shape)
                X.append(np.clip(np.round(w * 256), -32768, 32767).astype(np.int16))
                RR.append(rr_i)
                Y.append(C.CLASS_INDEX[labels[i]])

    X  = np.stack(X).astype(np.int16)
    RR = np.stack(RR).astype(np.float32)
    Y  = np.array(Y, dtype=np.int8)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, f'{split_name}_X.npy'),  X)
    np.save(os.path.join(OUT, f'{split_name}_RR.npy'), RR)
    np.save(os.path.join(OUT, f'{split_name}_y.npy'),  Y)

    counts = {c: int((Y == i).sum()) for i, c in enumerate(C.AAMI_CLASSES)}
    print(f"  {split_name}: {len(Y)} beats  {counts}")
    return rows, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--download', action='store_true')
    ap.add_argument('--augment', action='store_true',
                     help='augment DS1 with jitter+noise copies -- NOT recommended, '
                          'see docs/development_log.md 2.5.1 (made macro-F1 worse in testing)')
    a = ap.parse_args()
    if a.download:
        download()

    missing = [r for r in ALL_RECORDS
               if not os.path.exists(os.path.join(RAW, f'{r}.dat'))]
    if missing:
        print(f"FAIL: {len(missing)} records missing: {missing[:8]}...")
        print("Run:  python mitbih.py --download")
        sys.exit(1)

    # augment=False is the DEPLOYED default. A 2026-09-02 ablation (see
    # docs/development_log.md 2.5.1) found jitter/noise augmentation, at every
    # magnitude tested, made macro-F1 WORSE than no augmentation -- so the
    # standard pipeline must NOT silently switch to it. Pass --augment to
    # opt into the (currently not-recommended) augmented DS1 explicitly.
    print("\n=== DS1 (training patients) ===" +
          (" [position-jitter + noise augmented]" if a.augment else ""))
    r1, c1 = build('DS1', C.DS1, augment=a.augment)
    print("\n=== DS2 (held-out test patients) === [UNCHANGED -- exact annotated peak, no aug]")
    r2, c2 = build('DS2', C.DS2, augment=False)

    with open(os.path.join(OUT, 'qrs_report.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['record','annotated','detected','TP','FP','FN','Se%','PPV%'])
        w.writerows(r1 + r2)

    allr = r1 + r2
    TP = sum(r[3] for r in allr); FP = sum(r[4] for r in allr)
    FN = sum(r[5] for r in allr)
    print(f"\n=== QRS DETECTOR, ALL 44 RECORDS ===")
    print(f"  TP={TP} FP={FP} FN={FN}")
    print(f"  Se  = {100*TP/(TP+FN):.2f}%   [Pan&Tompkins 1985 reported 99.3%]")
    print(f"  PPV = {100*TP/(TP+FP):.2f}%")
    print(f"\nPASS — arrays written to {OUT}")
    print("Next:  python train.py")


if __name__ == '__main__':
    main()
