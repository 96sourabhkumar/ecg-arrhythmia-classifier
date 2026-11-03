"""board_summary.py -- combine pynq/results/board_test_results.json into the on-board totals.
    import board_summary as S; S.summary()
"""
import os, json
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pynq', 'results')

def summary(path=os.path.join(RES, 'board_test_results.json')):
    R = json.load(open(path))
    c = n = c_all = n_all = beats = 0; ok = True
    for r in R:
        c += round(r['acc_vs_labels_min5_30'] * r['labelled_min5_30']); n += r['labelled_min5_30']
        c_all += round(r['acc_vs_labels'] * r['labelled']); n_all += r['labelled']
        beats += r['beats']; ok &= r['PASS']
        print(f"{r['record']:>4}  PASS={r['PASS']}  min5-30 acc {100*r['acc_vs_labels_min5_30']:6.2f}% on {r['labelled_min5_30']}")
    print(f'\nrecords {len(R)}   all PASS (bit-exact vs golden): {ok}   beats classified on FPGA: {beats}')
    print(f'ON-BOARD accuracy, minutes 5-30 (patient-adaptive test set): {c}/{n} = {100*c/n:.2f}%')
    print(f'on-board accuracy, whole records (includes training minutes): {c_all}/{n_all} = {100*c_all/n_all:.2f}%')
