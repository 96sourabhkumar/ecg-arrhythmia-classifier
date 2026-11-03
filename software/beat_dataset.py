"""
beat_dataset.py -- beat datasets at the positions the deployed system produces.

peaks  = peak_postprocessing.postprocess(x180, pan_tompkins_fixed peaks)   (HW detector golden
         model + ARM post-processing, exactly what pynq_driver does online)
window = x180[p-90 : p+166] at the post-processed peak, z-normalised (as mitbih.py)
RR     = mitbih.rr_features on the post-processed peak train (float, for training)
label  = AAMI class of the annotation matched within 150 ms; unmatched detections
         (false positives) have no label and are not scored.
Beats 0 and P-1 of each record are dropped (no pre/post RR), as in hardware.

Writes data/beats/:
  DS1.npz          all DS1 beats (inter-patient training)
  DS2.npz          all DS2 beats (inter-patient test)
  PA_train.npz     DS1 + DS2 minutes 0-4   (patient-adaptive training; AAMI EC57 local data)
  PA_val.npz       DS2 minute 4-5          (patient-adaptive model selection ONLY)
  PA_test.npz      DS2 minutes 5-30        (patient-adaptive test, scored once)
  PA_train4.npz    PA_train with the DS2 minute 0-4 beats repeated 4x (selected config)
Each npz: X (n,1,256) float32, R (n,4) float32, y (n,) int, rec (n,), pos (n,), j (n,)
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import config as C
from qrs.pan_tompkins import pan_tompkins_fixed, FS
from mitbih import load_record, rr_features
from beat_matching import match
from peak_postprocessing import postprocess
OUT = os.path.join(HERE, '..', 'data', 'beats')
PRE, POST = 90, 166
MIN = 60 * FS


def window(x, p):
    a, b = p - PRE, p + POST
    if a < 0 or b > len(x):
        return None
    w = x[a:b].astype(np.float64); w -= w.mean(); s = w.std(); w = w / s if s > 1e-6 else w
    return (np.clip(np.round(w * 256), -32768, 32767) / 256.0).astype(np.float32)


def record_beats(rec):
    adc, ann_s, ann_sym = load_record(rec)
    det, _, st = pan_tompkins_fixed(adc, return_stages=True)
    x = np.asarray(st['dec'], np.int64)
    pk = postprocess(x, np.asarray(det, np.int64))
    beat = np.array([s in C.SYMBOL_TO_AAMI for s in ann_sym])
    ann = (ann_s[beat] // 2).astype(np.int64)
    lab = np.array([C.CLASS_INDEX[C.SYMBOL_TO_AAMI[s]] for s in ann_sym[beat]])
    out = []
    for j, ai in match(pk, ann, int(0.150 * FS)):
        if j < 1 or j >= len(pk) - 1:
            continue
        w = window(x, int(pk[j]))
        if w is None:
            continue
        out.append((w, rr_features(pk, j), lab[ai], rec, int(pk[j]), j))
    return out


def pack(rows):
    X = np.stack([r[0] for r in rows])[:, None, :]
    R = np.clip(np.stack([r[1] for r in rows]).astype(np.float32), -5, 5)
    return dict(X=X, R=R, y=np.array([r[2] for r in rows]), rec=np.array([r[3] for r in rows]),
                pos=np.array([r[4] for r in rows]), j=np.array([r[5] for r in rows]))


def save(name, d):
    np.savez(os.path.join(OUT, name), **d)
    print(f'  {name}: {len(d["y"])} beats {np.bincount(d["y"], minlength=5).tolist()}', flush=True)


def cat(a, b, rec_offset=0):
    return {k: np.concatenate([a[k], b[k] + (rec_offset if k == 'rec' else 0)]) for k in a}


def sub(d, m):
    return {k: v[m] for k, v in d.items()}


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    ds1 = pack([b for r in C.DS1 for b in record_beats(r)]); save('DS1.npz', ds1)
    ds2 = pack([b for r in C.DS2 for b in record_beats(r)]); save('DS2.npz', ds2)
    loc = sub(ds2, ds2['pos'] < 4 * MIN)
    save('PA_train.npz', cat(ds1, loc, 10000))          # rec+10000 marks DS2 local beats
    save('PA_val.npz', sub(ds2, (ds2['pos'] >= 4 * MIN) & (ds2['pos'] < 5 * MIN)))
    save('PA_test.npz', sub(ds2, ds2['pos'] >= 5 * MIN))
    save('PA_train4.npz', cat(ds1, cat(cat(cat(loc, loc), loc), loc), 10000))
