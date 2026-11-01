"""Flattened round states and (s_t, a, Δ, s_t+Δ) sampling for JEPA training."""

from dataclasses import dataclass

import numpy as np

from cs2coach.features.state_tensor import RoundStates, full_action
from cs2coach.models.jepa.model import DELTAS


@dataclass
class StateBank:
    rounds: list[RoundStates]
    tokens: np.ndarray        # (N, N_TOKENS, TOKEN_DIM)
    round_idx: np.ndarray     # (N,) index into rounds
    local_idx: np.ndarray     # (N,) sample index within its round
    ct_win: np.ndarray        # (N,)
    match_ids: np.ndarray     # (N,)

    @staticmethod
    def build(rounds: list[RoundStates]) -> "StateBank":
        toks, ri, li, win, mids = [], [], [], [], []
        for k, rs in enumerate(rounds):
            n = len(rs.ticks)
            toks.append(rs.tokens)
            ri.append(np.full(n, k))
            li.append(np.arange(n))
            win.append(np.full(n, rs.ct_win))
            mids.append(np.full(n, rs.match_id, dtype=object))
        return StateBank(rounds, np.concatenate(toks), np.concatenate(ri), np.concatenate(li),
                         np.concatenate(win), np.concatenate(mids))

    def subset(self, mask: np.ndarray) -> "StateBank":
        return StateBank(self.rounds, self.tokens[mask], self.round_idx[mask], self.local_idx[mask],
                         self.ct_win[mask], self.match_ids[mask])

    def sample_pairs(self, n: int, rng: np.random.Generator):
        """Random transitions. Returns s0, actions, delta index, s1 as numpy arrays."""
        s0 = np.empty((n,) + self.tokens.shape[1:], np.float32)
        s1 = np.empty_like(s0)
        acts = np.empty((n, 28), np.float32)
        dts = np.empty(n, np.int64)
        filled = 0
        while filled < n:
            g = rng.integers(len(self.tokens))
            d_idx = rng.integers(len(DELTAS))
            rs = self.rounds[self.round_idx[g]]
            i = self.local_idx[g]
            j = i + DELTAS[d_idx]
            if j >= len(rs.ticks):
                j = len(rs.ticks) - 1
                if j <= i:
                    continue
                # Clamp to round end and use the nearest delta bucket at or below the true gap.
                d_idx = max(k for k, d in enumerate(DELTAS) if d <= max(j - i, DELTAS[0]))
            s0[filled], s1[filled] = rs.tokens[i], rs.tokens[j]
            acts[filled] = full_action(rs, i, j)
            dts[filled] = d_idx
            filled += 1
        return s0, acts, dts, s1


def split_matches(bank: StateBank, test_frac: float = 0.2, seed: int = 0) -> tuple[StateBank, StateBank]:
    mids = np.array(sorted(set(bank.match_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(mids)
    n_test = max(1, int(len(mids) * test_frac)) if len(mids) > 1 else 0
    test = np.isin(bank.match_ids, mids[:n_test])
    return bank.subset(~test), bank.subset(test)
