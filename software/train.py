"""
train.py -- train ECGNet (8,101 parameters) on the beat datasets written by beat_dataset.py.

Selection protocol: the epoch is chosen on --val ONLY; --test is scored once, at the end.
Selected configuration (chosen among 5 configs by PA_val accuracy only, see
docs/accuracy_analysis.txt):
  python train.py --train PA_train4 --val PA_val --test PA_test \
         --epochs 20 --wpow 0.5 --wcap 10 --seed 0 --tag patient_adaptive
"""
import argparse, json, copy, os, sys
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ecgnet import ECGNet, metrics, predict
D = os.path.join(HERE, '..', 'data', 'beats')
OUT = os.path.join(HERE, '..', 'results', 'models')

ap = argparse.ArgumentParser()
ap.add_argument('--train', default='PA_train4'); ap.add_argument('--val', default='PA_val')
ap.add_argument('--test', default='PA_test'); ap.add_argument('--epochs', type=int, default=20)
ap.add_argument('--wpow', type=float, default=0.5); ap.add_argument('--wcap', type=float, default=10)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--threads', type=int, default=1); ap.add_argument('--tag', default='patient_adaptive')
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
L = lambda n: (lambda d: (d['X'], d['R'], d['y'].astype(np.int64)))(np.load(os.path.join(D, n + '.npz')))
X, R, y = L(a.train); Xv, Rv, yv = L(a.val)
cnt = np.bincount(y, minlength=5).astype(float); cnt[cnt == 0] = 1
w = np.clip((cnt.sum() / (5 * cnt)) ** a.wpow, 0, a.wcap); print('class weights', np.round(w, 2), flush=True)
m = ECGNet()
crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
Xt, Rt, yt = torch.tensor(X), torch.tensor(R), torch.tensor(y); n = len(yt); bs = 256
predict(m, Xv, Rv)            # (kept so the RNG/BN state sequence matches the original run)
best = (-1, None, -1); hist = []
for ep in range(a.epochs):
    m.train(); perm = torch.randperm(n)
    for i in range(0, n, bs):
        idx = perm[i:i + bs]; opt.zero_grad()
        loss = crit(m(Xt[idx], Rt[idx]), yt[idx]); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    sch.step()
    acc, mf1, _, _ = metrics(yv, predict(m, Xv, Rv)); hist.append((ep, acc, mf1))
    print(f' ep{ep} val acc {100*acc:.2f} mF1 {mf1:.4f}', flush=True)
    if acc > best[0]: best = (acc, copy.deepcopy(m.state_dict()), ep)
m.load_state_dict(best[1]); print(f'selected epoch {best[2]} (val acc {100*best[0]:.2f}%)')
os.makedirs(OUT, exist_ok=True); torch.save(m.state_dict(), os.path.join(OUT, f'ecgnet_{a.tag}.pt'))
Xs, Rs, ys = L(a.test); acc, mf1, cm, f1 = metrics(ys, predict(m, Xs, Rs))
print(f'TEST (scored once) float acc {100*acc:.2f}%  macro-F1 {mf1:.4f}\n{cm}')
json.dump(dict(vars(a), val_acc=best[0], epoch=best[2], hist=hist, test_acc=acc, test_mf1=mf1, cm=cm.tolist()),
          open(os.path.join(OUT, f'training_{a.tag}.json'), 'w'), indent=1)
