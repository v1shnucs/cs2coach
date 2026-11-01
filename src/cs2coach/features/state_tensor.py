"""Game state as a fixed set of entity tokens, plus team actions between states.

Tokens per snapshot (N_TOKENS x TOKEN_DIM):
  0..4  CT players (sorted by steamid), 5..9 T players, 10 global token.
Player token: x, y, z, vx, vy, cos(yaw), sin(yaw), hp, armor, helmet, defuser,
              has_bomb, equip, flashed, alive, is_ct
Global token: t_sec, bomb_planted, since_plant, bomb_x, bomb_y, ct_alive, t_alive, padding.

Action for a team over a window = each slot's normalised destination (x, y)
plus the count of smokes/flashes/molotovs/HEs it threw: 5*2 + 4 = 14 dims.
The full action vector is CT action then T action (28 dims).
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

N_PLAYERS = 10
N_TOKENS = 11
TOKEN_DIM = 16
TEAM_ACTION_DIM = 14
ACTION_DIM = 2 * TEAM_ACTION_DIM
POS_SCALE = 4000.0
GRENADES = ("smoke", "flash", "molotov", "he")


def _grenade_kind(t: str | None) -> int | None:
    if not t:
        return None
    t = t.lower()
    if "smoke" in t:
        return 0
    if "flash" in t:
        return 1
    if "molotov" in t or "incend" in t or "inferno" in t or "fire" in t:
        return 2
    if t.startswith("he") or "hegrenade" in t:
        return 3
    return None


@dataclass
class RoundStates:
    """All sampled states of one round, in tick order."""
    match_id: str
    round_num: int
    map_name: str
    ticks: np.ndarray          # (T,)
    t_sec: np.ndarray          # (T,)
    tokens: np.ndarray         # (T, N_TOKENS, TOKEN_DIM) float32
    pos: np.ndarray            # (T, 10, 2) raw world x, y (for actions/plots)
    alive: np.ndarray          # (T, 10) bool
    steamids: list[str]        # slot -> steamid ('' for empty slot)
    nade_counts: np.ndarray    # (T, 10, 4) cumulative grenades thrown by slot up to tick
    ct_win: int
    survived: np.ndarray       # (10,) alive at round end


def build_round_states(ticks: pl.DataFrame, rounds: pl.DataFrame, bomb: pl.DataFrame,
                       grenades: pl.DataFrame) -> list[RoundStates]:
    out = []
    plants = {(r["match_id"], r["round_num"]): r for r in bomb.filter(pl.col("event") == "plant").iter_rows(named=True)}
    winners = {(r["match_id"], r["round_num"]): r["winner"] for r in rounds.iter_rows(named=True)}
    nades_by_round = {k: g for k, g in grenades.partition_by(["match_id", "round_num"], as_dict=True).items()} \
        if not grenades.is_empty() else {}

    for (mid, rn), g in ticks.sort("tick").partition_by(["match_id", "round_num"], as_dict=True).items():
        if (mid, rn) not in winners:
            continue
        rs = _round_tensor(mid, rn, g, plants.get((mid, rn)), winners[(mid, rn)], nades_by_round.get((mid, rn)))
        if rs is not None:
            out.append(rs)
    return out


def _round_tensor(mid, rn, g: pl.DataFrame, plant, winner, nades) -> RoundStates | None:
    ct = sorted(g.filter(pl.col("side") == "CT")["steamid"].unique().to_list())[:5]
    t = sorted(g.filter(pl.col("side") == "T")["steamid"].unique().to_list())[:5]
    slots = ct + [""] * (5 - len(ct)) + t + [""] * (5 - len(t))
    slot_of = {s: i for i, s in enumerate(slots) if s}

    tick_list = g["tick"].unique().sort().to_numpy()
    if len(tick_list) < 16:
        return None
    tix = {tk: i for i, tk in enumerate(tick_list)}
    T = len(tick_list)
    tok = np.zeros((T, N_TOKENS, TOKEN_DIM), np.float32)
    pos = np.zeros((T, 10, 2), np.float32)
    alive = np.zeros((T, 10), bool)

    g = g.filter(pl.col("steamid").is_in(list(slot_of)))
    ti = np.array([tix[x] for x in g["tick"].to_numpy()])
    si = np.array([slot_of[s] for s in g["steamid"].to_list()])
    f = lambda c, default=0.0: g[c].fill_null(default).cast(pl.Float32).to_numpy() if c in g.columns else np.full(g.height, default, np.float32)
    yaw = np.radians(f("yaw"))
    feats = np.stack([
        f("x") / POS_SCALE, f("y") / POS_SCALE, f("z") / 1000.0,
        f("vx") / 250.0, f("vy") / 250.0, np.cos(yaw), np.sin(yaw),
        f("hp") / 100.0, f("armor") / 100.0, f("helmet"), f("defuser"), f("has_bomb"),
        f("equip_value") / 6000.0, f("flashed"), f("alive"), (si < 5).astype(np.float32),
    ], axis=1)
    feats[feats[:, 14] == 0, :15] = 0.0  # dead players carry no state beyond side
    tok[ti, si] = feats
    pos[ti, si, 0], pos[ti, si, 1] = f("x"), f("y")
    alive[ti, si] = f("alive") > 0

    t_sec = np.zeros(T, np.float32)
    t_sec[ti] = f("t_sec")
    planted = np.zeros(T, np.float32)
    since = np.zeros(T, np.float32)
    bx = by = 0.0
    if plant is not None:
        planted = (tick_list >= plant["tick"]).astype(np.float32)
        since = np.maximum(tick_list - plant["tick"], 0) / 64.0 * planted
        bx, by = (plant["x"] or 0.0) / POS_SCALE, (plant["y"] or 0.0) / POS_SCALE
    tok[:, 10, 0] = t_sec / 115.0
    tok[:, 10, 1] = planted
    tok[:, 10, 2] = since / 40.0
    tok[:, 10, 3] = bx * planted
    tok[:, 10, 4] = by * planted
    tok[:, 10, 5] = alive[:, :5].sum(1) / 5.0
    tok[:, 10, 6] = alive[:, 5:].sum(1) / 5.0

    nade = np.zeros((T, 10, 4), np.float32)
    if nades is not None:
        for r in nades.iter_rows(named=True):
            k, s = _grenade_kind(r.get("grenade_type")), slot_of.get(r.get("steamid"))
            if k is None or s is None:
                continue
            nade[tick_list >= r["throw_tick"], s, k] += 1

    return RoundStates(mid, rn, g["map_name"][0] if "map_name" in g.columns else "unknown",
                       tick_list, t_sec, tok, pos, alive, slots, nade,
                       int(winner == "CT"), alive[-1].copy())


def team_action(rs: RoundStates, i: int, j: int, side: str) -> np.ndarray:
    """Action of one team between sample i and j: slot destinations + grenades thrown."""
    sl = slice(0, 5) if side == "CT" else slice(5, 10)
    dest = rs.pos[j, sl] / POS_SCALE
    dest = np.where(rs.alive[j, sl, None], dest, rs.pos[i, sl] / POS_SCALE)
    nades = (rs.nade_counts[j, sl] - rs.nade_counts[i, sl]).sum(0)
    return np.concatenate([dest.reshape(-1), nades]).astype(np.float32)


def full_action(rs: RoundStates, i: int, j: int) -> np.ndarray:
    return np.concatenate([team_action(rs, i, j, "CT"), team_action(rs, i, j, "T")])
