"""
board_test.py -- ON-BOARD test of the complete system (PYNQ-Z2, ecg_accel bitstream).
Upload the package built by pynq/make_pynq_package.py, load pynq/overlay/ecg_accel.bit, then:
    import board_test as T; T.run_all()
Checks, per record:  HW peaks == golden HW peaks; cleaned peaks == golden; per beat
RR word (read back from A_RR_DBG) == golden and class == golden bit-exact; accuracy vs
labels; CNN latency. Writes pynq/results/board_test_results.json
"""
import os, re, sys, glob, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pynq_driver as D
A_RR_DBG = 0x28
TEST_VERSION = '1.0.0'
# Refuse to run with a stale driver (the 2026-10-09 re-run of record 113 reproduced the OLD
# driver's result digit-for-digit -> the fixed file had not been loaded).
if getattr(D, 'DRIVER_VERSION', None) != TEST_VERSION or getattr(D, 'LOOKAHEAD', 0) != 32:
    raise RuntimeError(f'STALE DRIVER loaded from {D.__file__}: DRIVER_VERSION='
                       f'{getattr(D, "DRIVER_VERSION", None)}. Re-upload the package '
                       f'and restart the kernel (Kernel -> Restart).')
print(f'board_test {TEST_VERSION}: driver {D.__file__} LOOKAHEAD={D.LOOKAHEAD} OK')
BT = os.path.join(HERE, '..', 'pynq', 'test_vectors')
RES = os.path.join(HERE, '..', 'pynq', 'results')


def _rec_id(path):
    return int(re.search(r'(\d+)\.npz$', path).group(1))


Checked = D.EcgAccel


def run_record(path, board, realtime=True):
    g = np.load(path); rec = str(_rec_id(path))
    words = []                       # RR word latched by the PL, one per classified beat
    orig = D.classify_beat
    def cb(board_, *a, **k):
        cls, rr, lat = orig(board_, *a, **k)
        words.append(board_.read(A_RR_DBG) & 0xFFFFFFFF if cls is not None else None)
        return cls, rr, lat
    D.classify_beat = cb
    try:
        t0 = time.time(); r = D.run(g['x180'], board=board, realtime=realtime)
    finally:
        D.classify_beat = orig
    hw_ok = list(map(int, r['hw_peaks'])) == list(map(int, g['hw_peaks']))
    pk_ok = list(map(int, r['peaks'])) == list(map(int, g['peaks']))
    gk = {int(k): i for i, k in enumerate(g['beat_k'])}
    n = cls_ok = rr_ok = 0; lat = []; acc_n = acc_c = 0; t_n = t_c = 0
    for (k, p, cls, rr, l), w in zip(r['beats'], words):
        i = gk.get(int(k))
        if i is None: continue
        n += 1; cls_ok += int(cls == int(g['cls_gold'][i])); rr_ok += int(w is not None and w == int(g['rr_word'][i]))
        if l is not None: lat.append(l)
        if g['label'][i] >= 0:
            acc_n += 1; acc_c += int(cls == int(g['label'][i]))
            if p >= 5 * 60 * 180:          # minutes 5-30: the patient-adaptive TEST region
                t_n += 1; t_c += int(cls == int(g['label'][i]))
    res = dict(record=rec, hw_peaks_match=hw_ok, cleaned_peaks_match=pk_ok, beats=n,
               beats_expected=int(len(g['beat_k'])), class_vs_golden=f'{cls_ok}/{n}',
               rr_vs_golden=f'{rr_ok}/{n}', acc_vs_labels=acc_c / max(acc_n, 1), labelled=acc_n,
               acc_vs_labels_min5_30=t_c / max(t_n, 1), labelled_min5_30=t_n,
               cnn_latency_ms_mean=1e3 * float(np.mean(lat)) if lat else None, wall_s=time.time() - t0)
    res['PASS'] = bool(hw_ok and pk_ok and n == len(g['beat_k']) and cls_ok == n and rr_ok == n)
    print(json.dumps(res, indent=1)); return res


def run_all(realtime=True, records=None):
    paths = sorted(glob.glob(os.path.join(BT, 'record_*.npz')))
    if records: paths = [p for p in paths if _rec_id(p) in set(records)]
    board = Checked(); out = [run_record(p, board, realtime) for p in paths]
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, 'board_test_results.json' if not records else 'board_test_quick.json'), 'w'), indent=1)
    print('ALL PASS' if all(o['PASS'] for o in out) else 'SOME FAILED'); return out
