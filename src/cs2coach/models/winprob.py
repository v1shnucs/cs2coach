"""Baseline tick-level round win probability (LightGBM + Platt calibration).

Also computes win-probability-added (WPA) for kills: the change in CT win
probability between the snapshot before and after each kill.
"""

import pickle
import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from cs2coach import config
from cs2coach.features.snapshot import FEATURES

MODEL_PATH = config.MODELS_DIR / "winprob.pkl"


@dataclass
class WinProbModel:
    booster: lgb.Booster
    calibrator: LogisticRegression

    def predict(self, snap: pl.DataFrame) -> np.ndarray:
        raw = self.booster.predict(snap.select(FEATURES).to_numpy(), raw_score=True)
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

    def save(self) -> None:
        config.ensure_dirs()
        MODEL_PATH.write_bytes(pickle.dumps(self))

    @staticmethod
    def load() -> "WinProbModel":
        return pickle.loads(MODEL_PATH.read_bytes())


def split_by_match(snap: pl.DataFrame, test_frac: float = 0.2, seed: int = 0):
    matches = snap["match_id"].unique().sort().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(matches)
    n_test = max(1, int(len(matches) * test_frac)) if len(matches) > 1 else 0
    test_ids = set(matches[:n_test])
    is_test = snap["match_id"].is_in(list(test_ids))
    return snap.filter(~is_test), snap.filter(is_test)


def alive_baseline(snap: pl.DataFrame, train: pl.DataFrame) -> np.ndarray:
    """Empirical CT win rate per (ct_alive, t_alive) from train: the bar to beat."""
    table = train.group_by("ct_alive", "t_alive").agg(p=pl.col("ct_win").mean())
    prior = train["ct_win"].mean()
    return snap.join(table, on=["ct_alive", "t_alive"], how="left")["p"].fill_null(prior).to_numpy()


def train(snap: pl.DataFrame) -> tuple[WinProbModel, dict]:
    tr, te = split_by_match(snap)
    # Hold out a calibration slice of train matches so calibration is not fit on training predictions.
    # Platt scaling (not isotonic) keeps probabilities smooth, which WPA deltas depend on.
    fit, cal = split_by_match(tr, test_frac=0.2, seed=1)
    if cal.is_empty():
        fit, cal = tr, tr

    booster = lgb.train(
        dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
             feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1),
        lgb.Dataset(fit.select(FEATURES).to_numpy(), fit["ct_win"].to_numpy(), feature_name=FEATURES),
        num_boost_round=400,
    )
    platt = LogisticRegression()
    platt.fit(booster.predict(cal.select(FEATURES).to_numpy(), raw_score=True).reshape(-1, 1), cal["ct_win"].to_numpy())
    model = WinProbModel(booster, platt)

    metrics = {"n_train_rows": tr.height, "n_test_rows": te.height}
    if not te.is_empty():
        y = te["ct_win"].to_numpy()
        p = np.clip(model.predict(te), 1e-4, 1 - 1e-4)
        b = np.clip(alive_baseline(te, tr), 1e-4, 1 - 1e-4)
        metrics.update(
            brier=brier_score_loss(y, p), logloss=log_loss(y, p, labels=[0, 1]),
            baseline_brier=brier_score_loss(y, b), baseline_logloss=log_loss(y, b, labels=[0, 1]),
        )
    return model, metrics


def kill_wpa(model: WinProbModel, snap: pl.DataFrame, kills: pl.DataFrame) -> pl.DataFrame:
    """Attach CT win probability before/after each kill and WPA for the killer's side."""
    snap = snap.with_columns(wp_ct=pl.Series(model.predict(snap))).sort("match_id", "round_num", "tick")
    s = snap.select("match_id", "round_num", "tick", "wp_ct")
    k = kills.sort("match_id", "round_num", "tick")
    # Before = last snapshot strictly before the kill (the one at the kill tick may already show the death).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        before = k.join_asof(s.rename({"tick": "s_tick"}), left_on="tick", right_on="s_tick",
                             by=["match_id", "round_num"], strategy="backward",
                             allow_exact_matches=False).rename({"wp_ct": "wp_before"})
        after = k.join_asof(s.rename({"tick": "s_tick"}), left_on="tick", right_on="s_tick",
                            by=["match_id", "round_num"], strategy="forward")
    out = before.drop("s_tick").with_columns(wp_after=after["wp_ct"])
    delta = pl.col("wp_after") - pl.col("wp_before")
    return out.with_columns(
        wpa=pl.when(pl.col("attacker_side") == "CT").then(delta).otherwise(-delta)
    )
