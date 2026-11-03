"""
ecgnet.py -- the 1D-CNN classifier (ECGNet, 8,101 parameters) and two helpers.

  ECGNet   four Conv1d+BatchNorm+ReLU+MaxPool(2) blocks (8/16/32/32 channels, kernels 5/5/3/3),
           global average pooling, an RR branch (4 -> 16 -> 16), and FC 48 -> 40 -> 5.
           Input: (B, 1, 256) normalised beat window and (B, 4) RR features. Output: 5 logits
           (AAMI N, S, V, F, Q). Parameter names (c1, b1, ..., fc2) are what quantize.py expects.
  metrics  (re-exported from classification_metrics.py) accuracy, macro-F1, confusion matrix.
  predict  batched argmax inference.
The architecture is fixed by the hardware (hardware/rtl/cnn/cnn_accelerator.v); do not change it
without regenerating the RTL constants.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from classification_metrics import metrics   # re-exported for train.py


class ECGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(1, 8, 5, padding=2);   self.b1 = nn.BatchNorm1d(8)
        self.c2 = nn.Conv1d(8, 16, 5, padding=2);  self.b2 = nn.BatchNorm1d(16)
        self.c3 = nn.Conv1d(16, 32, 3, padding=1); self.b3 = nn.BatchNorm1d(32)
        self.c4 = nn.Conv1d(32, 32, 3, padding=1); self.b4 = nn.BatchNorm1d(32)
        self.rr1 = nn.Linear(4, 16);  self.rrb = nn.BatchNorm1d(16)
        self.rr2 = nn.Linear(16, 16)
        self.fc1 = nn.Linear(48, 40); self.fc2 = nn.Linear(40, 5)

    def forward(self, x, rr):
        for c, b in ((self.c1, self.b1), (self.c2, self.b2),
                     (self.c3, self.b3), (self.c4, self.b4)):
            x = F.max_pool1d(F.relu(b(c(x))), 2)
        g = x.mean(dim=2)
        r = F.relu(self.rrb(self.rr1(rr)))
        r = F.relu(self.rr2(r))
        return self.fc2(F.relu(self.fc1(torch.cat([g, r], 1))))


def predict(m, X, R, bs=8192):
    m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o.append(m(torch.tensor(X[i:i+bs]), torch.tensor(R[i:i+bs])).argmax(1).numpy())
    return np.concatenate(o)
