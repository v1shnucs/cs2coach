"""Train the JEPA world model, then fit probes on frozen target latents."""

import json
import pickle
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from cs2coach import config
from cs2coach.models.jepa.data import StateBank, split_matches
from cs2coach.models.jepa.model import JEPA, device

CKPT = config.MODELS_DIR / "jepa.pt"
PROBE = config.MODELS_DIR / "jepa_probes.pkl"


def _t(x, dev):
    return torch.from_numpy(x).to(dev)


@torch.no_grad()
def embed_bank(model: JEPA, bank: StateBank, dev, bs: int = 2048) -> np.ndarray:
    model.eval()
    out = [model.embed(_t(bank.tokens[i:i + bs], dev)).cpu().numpy() for i in range(0, len(bank.tokens), bs)]
    return np.concatenate(out) if out else np.zeros((0, model.d), np.float32)


def train(bank: StateBank, steps: int = 3000, batch: int = 256, lr: float = 3e-4, d: int = 256,
          seed: int = 0, log_every: int = 200) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    tr, te = split_matches(bank)
    model = JEPA(d=d).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)

    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        s0, a, dt, s1 = sample = tr.sample_pairs(batch, rng)
        loss, parts = model.loss(*(_t(x, dev) for x in sample))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        model.update_target()
        if step % log_every == 0 or step == steps:
            entry = dict(step=step, loss=loss.item(), **parts, sec=round(time.time() - t0, 1))
            if len(te.tokens):
                model.eval()
                with torch.no_grad():
                    vs = te.sample_pairs(512, np.random.default_rng(123))
                    entry["val_pred"] = model.loss(*(_t(x, dev) for x in vs))[1]["pred"]
            history.append(entry)
            print(json.dumps(entry))

    config.ensure_dirs()
    torch.save({"state": model.state_dict(), "d": d}, CKPT)

    probe_metrics = fit_probes(model, tr, te, dev)
    return dict(params=n_params, history=history, probes=probe_metrics, device=str(dev))


def fit_probes(model: JEPA, tr: StateBank, te: StateBank, dev) -> dict:
    ztr = embed_bank(model, tr, dev)
    win = LogisticRegression(max_iter=2000, C=0.5)
    win.fit(ztr, tr.ct_win)
    PROBE.write_bytes(pickle.dumps({"win": win}))
    metrics = {}
    if len(te.tokens):
        zte = embed_bank(model, te, dev)
        p = np.clip(win.predict_proba(zte)[:, 1], 1e-4, 1 - 1e-4)
        metrics = dict(win_brier=brier_score_loss(te.ct_win, p), win_logloss=log_loss(te.ct_win, p, labels=[0, 1]),
                       latent_std=float(zte.std(0).mean()))
    return metrics


def load(dev=None) -> tuple[JEPA, dict]:
    dev = dev or device()
    ck = torch.load(CKPT, map_location=dev)
    model = JEPA(d=ck["d"]).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, pickle.loads(PROBE.read_bytes())
