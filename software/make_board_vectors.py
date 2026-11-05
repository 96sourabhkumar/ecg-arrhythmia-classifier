"""
make_board_vectors.py -- OFFLINE (needs wfdb). Golden expectations for board_test.py.
For each record writes pynq/test_vectors/record_<R>.npz:
  x180      decimated stream (exactly what the driver pushes)
  hw_peaks  golden HW detector peaks (pan_tompkins_fixed, bit-exact with RTL)
  peaks     golden post-processed peaks (peak_postprocessing.py)
  beat_k, rr_word, cls_gold, label   per classified beat (label -1 = unmatched detection)
Default: all 22 DS2 records (the set used for the reported board run).
  python make_board_vectors.py [rec ...]
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
args = [int(a) for a in sys.argv[1:]]; sys.argv = sys.argv[:1]
import config as C
import int8_model as G
from qrs.pan_tompkins import pan_tompkins_fixed, FS
from qrs.rr_features import rr_int8_stream, constants
from mitbih import load_record
from beat_matching import match
from peak_postprocessing import postprocess
from pynq_mmio import normalize_quantize_beat
QDIR = os.path.join(HERE, '..', 'hardware', 'weights'); G.QDIR = QDIR
args = args or list(C.DS2)
Q, MQ, gmi, gsh = G.load_weights(); classify = G.make_classifier(Q, MQ, gmi, gsh); G.verify_against_gold(classify)
q = json.load(open(os.path.join(QDIR, 'quant_params.json'))); s_act = q['S']['x']; K, CLIP = constants(q['S']['rr'])
OUT = os.path.join(HERE, '..', 'pynq', 'test_vectors'); os.makedirs(OUT, exist_ok=True)
for rec in args:
    adc, ann_s, ann_sym = load_record(rec)
    det, _, st = pan_tompkins_fixed(adc, return_stages=True); x = np.asarray(st['dec'], np.int64)
    det = np.asarray(det, np.int64); pk = postprocess(x, det); rr = rr_int8_stream(pk, K, CLIP)
    beat = np.array([s in C.SYMBOL_TO_AAMI for s in ann_sym]); ann = (ann_s[beat] // 2).astype(np.int64)
    lab = np.array([C.CLASS_INDEX[C.SYMBOL_TO_AAMI[s]] for s in ann_sym[beat]])
    lmap = {j: int(lab[ai]) for j, ai in match(pk, ann, int(0.150 * FS))}
    ks, words, cls, labels = [], [], [], []
    for k in range(1, len(pk) - 1):
        p = int(pk[k])
        if p - 90 < 0 or p + 166 > len(x): continue
        b = normalize_quantize_beat(x[p - 90:p + 166], s_act).astype(np.int32)
        r = rr[k].astype(np.int32)
        ks.append(k); words.append(int(((r[0] & 255) << 24) | ((r[1] & 255) << 16) | ((r[2] & 255) << 8) | (r[3] & 255)))
        cls.append(int(classify(b.reshape(1, 1, 256), r.reshape(1, 4))[0])); labels.append(lmap.get(k, -1))
    np.savez(os.path.join(OUT, f'record_{rec}.npz'), x180=x, hw_peaks=det, peaks=pk, beat_k=np.array(ks),
             rr_word=np.array(words, np.int64), cls_gold=np.array(cls), label=np.array(labels))
    L = np.array(labels); C_ = np.array(cls); m = L >= 0
    print(f'rec {rec}: {len(det)} HW peaks -> {len(pk)} cleaned, {len(ks)} beats, golden acc vs labels '
          f'{100*(C_[m]==L[m]).mean():.2f}% on {m.sum()} labelled')
