"""Synthetic matches in the canonical schema, for tests and for exercising the
pipeline before real demos are available.

Each team has fixed tendencies (preferred T site, CT stack habit, smoke spots)
so scouting should be able to recover them. Positions live on a toy map with
two bombsites; nothing here resembles a real map layout.
"""

import numpy as np
import polars as pl

SITES = {"A": np.array([1200.0, 800.0]), "B": np.array([-1200.0, 900.0])}
T_SPAWN = np.array([0.0, -1600.0])
CT_SPAWN = np.array([0.0, 1400.0])
MID = np.array([0.0, 0.0])
PLACES = {"A": "BombsiteA", "B": "BombsiteB"}

TEAM_STYLES = {
    "Alpha": dict(t_site_pref=0.75, ct_a_stack=0.6, smoke_offset=(150, -200)),
    "Bravo": dict(t_site_pref=0.30, ct_a_stack=0.3, smoke_offset=(-180, -150)),
    "Charlie": dict(t_site_pref=0.55, ct_a_stack=0.5, smoke_offset=(0, -250)),
}


def _place(p: np.ndarray) -> str:
    for s, c in SITES.items():
        if np.linalg.norm(p - c) < 450:
            return PLACES[s]
    if np.linalg.norm(p - MID) < 500:
        return "Middle"
    if p[1] < -900:
        return "TSpawn"
    if p[1] > 1000:
        return "CTSpawn"
    return "ALong" if p[0] > 0 else "BTunnels"


def generate_match(team_a: str, team_b: str, seed: int, n_rounds: int = 24, hz: int = 8) -> tuple[str, dict]:
    rng = np.random.default_rng(seed)
    mid = f"syn{seed:06d}"
    tick_rate = 64
    step = tick_rate // hz
    players = {
        team: [(f"{7650000 + seed * 100 + ti * 10 + i}", f"{team[:3].lower()}{i}") for i in range(5)]
        for ti, team in enumerate((team_a, team_b))
    }

    rows_r, rows_t, rows_k, rows_g, rows_b, rows_d = [], [], [], [], [], []
    tick0 = 1000
    for rn in range(1, n_rounds + 1):
        first_half = rn <= n_rounds // 2
        t_team, ct_team = (team_a, team_b) if first_half else (team_b, team_a)
        ts, cts = TEAM_STYLES.get(t_team, TEAM_STYLES["Charlie"]), TEAM_STYLES.get(ct_team, TEAM_STYLES["Charlie"])
        pistol = rn in (1, n_rounds // 2 + 1)
        t_buy = "pistol" if pistol else rng.choice(["full", "full", "force", "eco"])
        ct_buy = "pistol" if pistol else rng.choice(["full", "full", "full", "force", "eco"])
        equip = {"pistol": 800, "eco": 1200, "force": 2800, "full": 5200}
        t_equip, ct_equip = equip[t_buy] * 5, equip[ct_buy] * 5

        target = "A" if rng.random() < ts["t_site_pref"] else "B"
        n_ct_a = 3 if rng.random() < cts["ct_a_stack"] else 2
        exec_time = rng.uniform(25, 60)
        dur = exec_time + rng.uniform(15, 40)
        freeze_end = tick0 + 15 * tick_rate
        end_tick = freeze_end + int(dur * tick_rate)

        state = {}
        for i, (sid, name) in enumerate(players[t_team]):
            state[sid] = dict(name=name, team=t_team, side="T", pos=T_SPAWN + rng.normal(0, 80, 2), hp=100,
                              goal=SITES[target] + rng.normal(0, 150, 2), bomb=(i == 0))
        for i, (sid, name) in enumerate(players[ct_team]):
            site = "A" if i < n_ct_a else "B"
            state[sid] = dict(name=name, team=ct_team, side="CT", pos=CT_SPAWN + rng.normal(0, 80, 2), hp=100,
                              goal=SITES[site] + rng.normal(0, 200, 2), bomb=False)

        # Utility: each T throws a smoke toward the target site at execute time.
        for i, (sid, name) in enumerate(players[t_team][:3]):
            land = SITES[target] + np.array(ts["smoke_offset"]) + rng.normal(0, 40, 2)
            tt = freeze_end + int((exec_time - 3 + i) * tick_rate)
            rows_g.append(dict(grenade_id=rn * 100 + i, steamid=sid, name=name, team=t_team, side="T",
                               grenade_type="smoke", throw_tick=tt, land_tick=tt + 128, t_sec=exec_time - 3 + i,
                               throw_x=float(SITES[target][0] * 0.4), throw_y=-400.0,
                               x=float(land[0]), y=float(land[1]), z=0.0, round_num=rn))

        # Fight strength: equipment and defenders on the attacked site.
        defenders = n_ct_a if target == "A" else 5 - n_ct_a
        p_t_win = np.clip(0.5 + (t_equip - ct_equip) / 40000 + (3 - defenders) * 0.08, 0.1, 0.9)
        t_wins = rng.random() < p_t_win
        winner = "T" if t_wins else "CT"

        plant_tick = None
        kill_ticks = sorted(rng.uniform(exec_time - 5, dur - 2, size=rng.integers(5, 9)))
        n_t_dead, n_ct_dead = 0, 0
        loser_side = "CT" if t_wins else "T"
        kill_plan = []
        for kt in kill_ticks:
            victim_side = loser_side if rng.random() < 0.7 else ("T" if loser_side == "CT" else "CT")
            if victim_side == "T" and n_t_dead >= (5 if not t_wins else 4):
                victim_side = "CT"
            if victim_side == "CT" and n_ct_dead >= (5 if t_wins else 4):
                victim_side = "T"
            if victim_side == "T":
                n_t_dead += 1
            else:
                n_ct_dead += 1
            kill_plan.append((freeze_end + int(kt * tick_rate), victim_side))

        if rng.random() < (0.85 if t_wins else 0.35):
            plant_tick = freeze_end + int((exec_time + rng.uniform(4, 10)) * tick_rate)
            plant_tick = min(plant_tick, end_tick - 64)

        for tick in range(freeze_end, end_tick + 1, step):
            t_sec = (tick - freeze_end) / tick_rate
            while kill_plan and kill_plan[0][0] <= tick:
                kt, vside = kill_plan.pop(0)
                alive_v = [s for s, v in state.items() if v["side"] == vside and v["hp"] > 0]
                alive_a = [s for s, v in state.items() if v["side"] != vside and v["hp"] > 0]
                if not alive_v or not alive_a:
                    continue
                v, a = rng.choice(alive_v), rng.choice(alive_a)
                state[v]["hp"] = 0
                if state[v]["bomb"]:
                    state[v]["bomb"] = False
                    nxt = [s for s in alive_v if s != v]
                    if nxt:
                        state[nxt[0]]["bomb"] = True
                va, vv = state[a], state[v]
                rows_k.append(dict(
                    tick=kt, t_sec=(kt - freeze_end) / tick_rate, round_num=rn,
                    attacker_steamid=a, attacker_name=va["name"], attacker_team=va["team"], attacker_side=va["side"],
                    attacker_x=float(va["pos"][0]), attacker_y=float(va["pos"][1]), attacker_place=_place(va["pos"]),
                    victim_steamid=v, victim_name=vv["name"], victim_team=vv["team"], victim_side=vv["side"],
                    victim_x=float(vv["pos"][0]), victim_y=float(vv["pos"][1]), victim_place=_place(vv["pos"]),
                    assister_steamid=None, weapon=rng.choice(["ak47", "m4a1", "awp", "usp_silencer"]),
                    headshot=bool(rng.random() < 0.45), is_trade=bool(rng.random() < 0.2),
                ))
                rows_d.append(dict(tick=kt, round_num=rn, attacker_steamid=a, victim_steamid=v, damage=100, weapon="ak47"))

            for sid, v in state.items():
                if v["hp"] > 0:
                    if v["side"] == "T" and t_sec < exec_time:
                        goal = MID + (v["goal"] - MID) * 0.3 if t_sec < exec_time * 0.6 else v["goal"] * 0.75
                    else:
                        goal = v["goal"]
                    if v["side"] == "CT" and plant_tick and tick > plant_tick:
                        goal = SITES[target]
                    d = goal - v["pos"]
                    vel = d / max(np.linalg.norm(d), 1) * min(250 / hz, np.linalg.norm(d))
                    v["pos"] = v["pos"] + vel + rng.normal(0, 6, 2)
                else:
                    vel = np.zeros(2)
                rows_t.append(dict(
                    tick=tick, t_sec=t_sec, round_num=rn, steamid=sid, name=v["name"], team=v["team"],
                    side=v["side"], x=float(v["pos"][0]), y=float(v["pos"][1]), z=0.0,
                    vx=float(vel[0] * hz), vy=float(vel[1] * hz), yaw=float(np.degrees(np.arctan2(vel[1], vel[0]))),
                    hp=v["hp"], armor=100 if v["hp"] > 0 and t_buy != "eco" else 0, helmet=v["hp"] > 0,
                    defuser=v["side"] == "CT" and ct_buy == "full", has_bomb=v["bomb"],
                    money=800, equip_value=(t_equip if v["side"] == "T" else ct_equip) // 5 if v["hp"] > 0 else 0,
                    weapon="ak47" if v["side"] == "T" else "m4a1", flashed=False, alive=v["hp"] > 0,
                    place=_place(v["pos"]),
                ))

        if plant_tick:
            rows_b.append(dict(tick=plant_tick, event="plant", steamid=players[t_team][0][0],
                               x=float(SITES[target][0]), y=float(SITES[target][1]), z=0.0, site=target, round_num=rn))
        rows_r.append(dict(
            round_num=rn, start_tick=tick0, freeze_end_tick=freeze_end, end_tick=end_tick, winner=winner,
            reason="bomb_exploded" if (t_wins and plant_tick) else ("t_killed" if not t_wins else "ct_killed"),
            ct_team=ct_team, t_team=t_team, ct_equip=ct_equip, t_equip=t_equip, ct_buy=ct_buy, t_buy=t_buy,
            plant_tick=plant_tick, plant_site=target if plant_tick else None,
        ))
        tick0 = end_tick + 7 * tick_rate

    add = lambda rows: pl.DataFrame(rows, infer_schema_length=None).with_columns(match_id=pl.lit(mid))
    frames = {
        "matches": pl.DataFrame({"match_id": [mid], "map_name": ["de_synthetic"], "demo_path": [""],
                                 "tick_rate": [float(tick_rate)], "team_a": [team_a], "team_b": [team_b],
                                 "n_rounds": [n_rounds]}),
        "rounds": add(rows_r), "ticks": add(rows_t).with_columns(map_name=pl.lit("de_synthetic")),
        "kills": add(rows_k), "damages": add(rows_d), "grenades": add(rows_g), "bomb": add(rows_b),
    }
    return mid, frames


def generate_dataset(n_matches: int = 12, seed: int = 0) -> list[tuple[str, dict]]:
    teams = list(TEAM_STYLES)
    out = []
    for i in range(n_matches):
        a, b = teams[i % 3], teams[(i + 1) % 3]
        out.append(generate_match(a, b, seed=seed + i))
    return out
