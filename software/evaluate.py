"""
evaluate.py -- bit-exact INT8 evaluation of the complete system on MIT-BIH DS2.

For every DS2 record: HW detector golden model (pan_tompkins_fixed, bit-exact with RTL)
-> peak_postprocessing.postprocess -> 256-sample window, INT8-quantised exactly as
pynq_driver.py does -> RR from qrs/rr_features.py::rr_int8 (integer spec) on the cleaned peak
train -> golden INT8 classifier (int8_model.make_classifier, same integer
arithmetic as cnn_accelerator.v, self-checked against the 64 golden vectors).

  python evaluate.py hardware/weights PA_test                        # deployed patient-adaptive model, minutes 5-30
  python evaluate.py results/models/inter_patient_int8 DS2         # inter-patient model, all of DS2

Prints accuracy on detected+labelled beats AND the strict figure in which every
annotated beat the detector missed counts as an error. Writes results/evaluation/<qdir name>_<split>.json
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
qdir, split = os.path.abspath(sys.argv[1]), sys.argv[2]; sys.argv = sys.argv[:1]
import config as C
import int8_model as G
from qrs.pan_tompkins import pan_tompkins_fixed, FS
from qrs.rr_features import rr_int8_stream, constants
from mitbih import load_record
from beat_matching import match
from peak_postprocessing import postprocess
from classification_metrics import metrics
G.QDIR = qdir
Q, MQ, gmi, gsh = G.load_weights(); classify = G.make_classifier(Q, MQ, gmi, gsh)
if os.path.exists(os.path.join(G.QDIR, 'golden_classes.mem')): G.verify_against_gold(classify)
q = json.load(open(os.path.join(qdir, 'quant_params.json'))); s_act = q['S']['x']; K, CLIP = constants(q['S']['rr'])
T0 = 5 * 60 * FS if split == 'PA_test' else 0
Y, P, per, truth = [], [], {}, 0
for rec in C.DS2:
    adc, ann_s, ann_sym = load_record(rec)
    det, _, st = pan_tompkins_fixed(adc, return_stages=True); x = np.asarray(st['dec'], np.int64)
    pk = postprocess(x, np.asarray(det, np.int64)); rr = rr_int8_stream(pk, K, CLIP)
    beat = np.array([s in C.SYMBOL_TO_AAMI for s in ann_sym]); ann = (ann_s[beat] // 2).astype(np.int64)
    lab = np.array([C.CLASS_INDEX[C.SYMBOL_TO_AAMI[s]] for s in ann_sym[beat]])
    a_ = ann[1:-1]; a_ = a_[(a_ >= 90) & (a_ + 166 <= len(x)) & (a_ >= T0)]; truth += len(a_)
    XI, RI, L = [], [], []
    for j, ai in match(pk, ann, int(0.150 * FS)):
        p = int(pk[j])
        if j < 1 or j >= len(pk) - 1 or p - 90 < 0 or p + 166 > len(x) or p < T0:
            continue
        w = x[p - 90:p + 166].astype(np.float64); w -= w.mean(); sd = w.std(); w = w / sd if sd > 1e-6 else w
        z = np.clip(np.round(w * 256), -32768, 32767)
        XI.append(np.clip(np.round((z / 256.0) / s_act), -127, 127).astype(np.int32))
        RI.append(rr[j].astype(np.int32)); L.append(lab[ai])
    pr = classify(np.stack(XI).reshape(-1, 1, 256), np.stack(RI)); L = np.array(L)
    per[rec] = dict(n=len(L), acc=float((pr == L).mean())); Y.append(L); P.append(pr)
    print(f'  {rec}: {len(L)} beats  acc {100*per[rec]["acc"]:.2f}%', flush=True)
Y, P = np.concatenate(Y), np.concatenate(P)
EVAL_DIR = os.path.join(HERE, '..', 'results', 'evaluation'); os.makedirs(EVAL_DIR, exist_ok=True)
acc, mf1, cm, f1 = metrics(Y, P); corr = int((Y == P).sum())
se = [100 * cm[i, i] / cm[i].sum() if cm[i].sum() else 0 for i in range(5)]
print(f'\n{qdir} on {split}: INT8 bit-exact accuracy {100*acc:.2f}%  macro-F1(N,S,V,F) {mf1:.4f}  n={len(Y)}')
print(f'  Se N/S/V/F = {se[0]:.2f}/{se[1]:.2f}/{se[2]:.2f}/{se[3]:.2f}')
print(f'  strict (detector-missed beats counted wrong): {corr}/{truth} = {100*corr/truth:.2f}%')
print('confusion (rows true N,S,V,F,Q):\n', cm)
json.dump(dict(split=split, acc=acc, mf1=mf1, n=len(Y), correct=corr, n_truth=truth, strict=corr / truth,
               se=se, f1=list(map(float, f1)), cm=cm.tolist(), per_record=per),
          open(os.path.join(EVAL_DIR, f'{os.path.basename(qdir)}_{split}.json'), 'w'), indent=1)
