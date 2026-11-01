"""Parse one .dem file with awpy into the canonical tables.

Canonical schema (all tables carry match_id and round_num):
  matches  : match_id, map_name, demo_path, tick_rate, team_a, team_b, n_rounds
  rounds   : round_num, start_tick, freeze_end_tick, end_tick, winner (CT/T), reason,
             ct_team, t_team, ct_equip, t_equip, ct_buy, t_buy, plant_tick, plant_site
  ticks    : tick, t_sec, steamid, name, team, side, x, y, z, vx, vy, yaw, hp, armor,
             helmet, defuser, has_bomb, money, equip_value, weapon, flashed, alive, place
  kills    : tick, t_sec, attacker_*/victim_* (steamid, name, team, side, x, y, place),
             weapon, headshot, is_trade
  damages  : tick, attacker_steamid, victim_steamid, damage, weapon
  grenades : grenade_id, steamid, name, team, side, grenade_type, throw_tick, land_tick,
             t_sec, throw_x, throw_y, x, y, z
  bomb     : tick, event (plant/defuse), steamid, x, y, z, site
"""

import hashlib
from pathlib import Path

import polars as pl
from awpy import Demo

from cs2coach import config


def match_id_for(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:12]


def _col(df: pl.DataFrame, *names: str) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def _pick(df: pl.DataFrame, mapping: dict[str, tuple[str, ...]]) -> pl.DataFrame:
    """Select/rename columns by first available candidate; missing ones become null."""
    exprs = []
    for out, cands in mapping.items():
        c = _col(df, *cands)
        exprs.append(pl.col(c).alias(out) if c else pl.lit(None).alias(out))
    return df.select(exprs)


def norm_side(expr: pl.Expr) -> pl.Expr:
    s = expr.cast(pl.Utf8).str.to_uppercase()
    return (
        pl.when(s.is_in(["CT", "3", "COUNTER-TERRORIST", "COUNTERTERRORIST"])).then(pl.lit("CT"))
        .when(s.is_in(["T", "2", "TERRORIST"])).then(pl.lit("T"))
        .otherwise(None)
    )


def assign_rounds(df: pl.DataFrame, rounds: pl.DataFrame, tick_col: str = "tick") -> pl.DataFrame:
    """Attach round_num to rows by tick, dropping rows outside any round."""
    r = rounds.select("round_num", "start_tick", "end_tick").sort("start_tick")
    out = df.sort(tick_col).join_asof(r, left_on=tick_col, right_on="start_tick", strategy="backward")
    return out.filter(pl.col(tick_col) <= pl.col("end_tick")).drop("start_tick", "end_tick")


def _rounds(demo: Demo) -> pl.DataFrame:
    r = _pick(demo.rounds, {
        "round_num": ("round_num",),
        "start_tick": ("start_tick",),
        "freeze_end_tick": ("freeze_end_tick", "freeze_end"),
        "end_tick": ("end_tick", "official_end_tick"),
        "winner": ("winner_side", "winner"),
        "reason": ("reason_name", "reason"),
    })
    r = r.with_columns(
        winner=norm_side(pl.col("winner")),
        freeze_end_tick=pl.col("freeze_end_tick").fill_null(pl.col("start_tick")),
        reason=pl.col("reason").cast(pl.Utf8),
    ).filter(pl.col("winner").is_not_null())

    try:
        eco = demo.round_economy.with_columns(side=norm_side(pl.col("side")))
        wide = eco.pivot(on="side", index="round_num", values=["equipment_value", "buy_type"])
        rename = {c: c.replace("equipment_value_", "").lower() + "_equip" if c.startswith("equipment_value")
                  else c.replace("buy_type_", "").lower() + "_buy" for c in wide.columns if c != "round_num"}
        r = r.join(wide.rename(rename), on="round_num", how="left")
    except Exception:
        pass
    for c in ("ct_equip", "t_equip"):
        if c not in r.columns:
            r = r.with_columns(pl.lit(None, pl.Int64).alias(c))
    for c in ("ct_buy", "t_buy"):
        if c not in r.columns:
            r = r.with_columns(pl.lit(None, pl.Utf8).alias(c))
    return r


def _ticks(demo: Demo, rounds: pl.DataFrame, teams: pl.DataFrame, tick_rate: float) -> pl.DataFrame:
    snaps = demo.snapshots(seconds=1.0 / config.SAMPLE_HZ)
    t = _pick(snaps, {
        "tick": ("tick",), "steamid": ("steamid",), "name": ("name",), "side": ("side", "team_num"),
        "x": ("x", "X"), "y": ("y", "Y"), "z": ("z", "Z"),
        "vx": ("velocity_x",), "vy": ("velocity_y",), "yaw": ("yaw",),
        "hp": ("health",), "armor": ("armor",), "helmet": ("has_helmet",), "defuser": ("has_defuser",),
        "has_bomb": ("has_bomb",), "money": ("money",), "equip_value": ("equipment_value",),
        "weapon": ("active_weapon", "primary_weapon"), "flashed": ("is_blinded",),
    })
    t = t.with_columns(
        side=norm_side(pl.col("side")),
        steamid=pl.col("steamid").cast(pl.Utf8),
        hp=pl.col("hp").fill_null(0).cast(pl.Int32),
        armor=pl.col("armor").fill_null(0).cast(pl.Int32),
        equip_value=pl.col("equip_value").fill_null(0).cast(pl.Int32),
        money=pl.col("money").fill_null(0).cast(pl.Int32),
        helmet=pl.col("helmet").fill_null(False).cast(pl.Boolean),
        defuser=pl.col("defuser").fill_null(False).cast(pl.Boolean),
        has_bomb=pl.col("has_bomb").fill_null(False).cast(pl.Boolean),
        flashed=pl.col("flashed").fill_null(False).cast(pl.Boolean),
        weapon=pl.col("weapon").cast(pl.Utf8),
    ).filter(pl.col("side").is_not_null())

    # Only live play: from freeze end to round end.
    r = rounds.select("round_num", "start_tick", "freeze_end_tick", "end_tick").sort("start_tick")
    t = t.sort("tick").join_asof(r, left_on="tick", right_on="start_tick", strategy="backward")
    t = t.filter((pl.col("tick") >= pl.col("freeze_end_tick")) & (pl.col("tick") <= pl.col("end_tick")))
    t = t.with_columns(
        t_sec=((pl.col("tick") - pl.col("freeze_end_tick")) / tick_rate).cast(pl.Float32),
        alive=pl.col("hp") > 0,
    ).drop("start_tick", "freeze_end_tick", "end_tick")

    t = _attach_place(demo, t)
    return t.join(teams, on="steamid", how="left")


def _attach_place(demo: Demo, t: pl.DataFrame) -> pl.DataFrame:
    """Callout name (e.g. 'BombsiteA', 'Banana') from the player's last_place_name."""
    try:
        p = demo.ticks(["last_place_name"], players_only=True)
        p = p.select(pl.col("tick"), pl.col("steamid").cast(pl.Utf8), pl.col("last_place_name").alias("place"))
        p = p.filter(pl.col("tick").is_in(t["tick"].unique().implode()))
        return t.join(p, on=["tick", "steamid"], how="left")
    except Exception:
        return t.with_columns(place=pl.lit(None, pl.Utf8))


def _player_cols(df: pl.DataFrame, role: str) -> dict[str, tuple[str, ...]]:
    return {
        f"{role}_steamid": (f"{role}_steamid",),
        f"{role}_name": (f"{role}_name",),
        f"{role}_side": (f"{role}_side", f"{role}_team_num"),
        f"{role}_x": (f"{role}_x", f"{role}_X"),
        f"{role}_y": (f"{role}_y", f"{role}_Y"),
        f"{role}_place": (f"{role}_place", f"{role}_last_place_name"),
    }


def _kills(demo: Demo, rounds: pl.DataFrame, teams: pl.DataFrame, tick_rate: float) -> pl.DataFrame:
    k = demo.kills
    k = _pick(k, {
        "tick": ("tick",), **_player_cols(k, "attacker"), **_player_cols(k, "victim"),
        "assister_steamid": ("assister_steamid",),
        "weapon": ("weapon",), "headshot": ("headshot",), "is_trade": ("is_trade",),
    })
    k = k.with_columns(
        attacker_side=norm_side(pl.col("attacker_side")), victim_side=norm_side(pl.col("victim_side")),
        attacker_steamid=pl.col("attacker_steamid").cast(pl.Utf8),
        victim_steamid=pl.col("victim_steamid").cast(pl.Utf8),
        assister_steamid=pl.col("assister_steamid").cast(pl.Utf8),
        headshot=pl.col("headshot").cast(pl.Utf8).str.to_lowercase().is_in(["true", "1"]),
        is_trade=pl.col("is_trade").fill_null(False).cast(pl.Boolean),
    )
    k = _add_round_time(assign_rounds(k, rounds), rounds, tick_rate)
    for role in ("attacker", "victim"):
        k = k.join(teams.rename({"steamid": f"{role}_steamid", "team": f"{role}_team"}),
                   on=f"{role}_steamid", how="left")
    return k


def _add_round_time(df: pl.DataFrame, rounds: pl.DataFrame, tick_rate: float, tick_col: str = "tick") -> pl.DataFrame:
    return df.join(rounds.select("round_num", "freeze_end_tick"), on="round_num", how="left").with_columns(
        t_sec=((pl.col(tick_col) - pl.col("freeze_end_tick")) / tick_rate).cast(pl.Float32)
    ).drop("freeze_end_tick")


def _damages(demo: Demo, rounds: pl.DataFrame) -> pl.DataFrame:
    d = demo.damages
    d = _pick(d, {
        "tick": ("tick",),
        "attacker_steamid": ("attacker_steamid",), "victim_steamid": ("victim_steamid",),
        "damage": ("dmg_health", "damage", "health_damage"), "weapon": ("weapon",),
    }).with_columns(
        pl.col("attacker_steamid").cast(pl.Utf8), pl.col("victim_steamid").cast(pl.Utf8),
        damage=pl.col("damage").cast(pl.Int32, strict=False),
    )
    return assign_rounds(d, rounds)


def _grenades(demo: Demo, rounds: pl.DataFrame, teams: pl.DataFrame, tick_rate: float) -> pl.DataFrame:
    """Collapse per-tick trajectories to one row per grenade: throw point and landing point."""
    g = demo.grenades
    g = _pick(g, {
        "tick": ("tick",), "grenade_id": ("entity_id", "grenade_entity_id", "entity_handle"),
        "steamid": ("thrower_steamid", "steamid"), "name": ("thrower_name", "thrower", "name"),
        "side": ("thrower_side", "side"), "grenade_type": ("grenade_type", "class_name", "weapon"),
        "x": ("x", "X"), "y": ("y", "Y"), "z": ("z", "Z"),
    }).with_columns(steamid=pl.col("steamid").cast(pl.Utf8), side=norm_side(pl.col("side")))
    g = g.sort("tick").group_by("grenade_id", "steamid", "grenade_type", maintain_order=True).agg(
        pl.col("name").first(), pl.col("side").first(),
        throw_tick=pl.col("tick").first(), land_tick=pl.col("tick").last(),
        throw_x=pl.col("x").first(), throw_y=pl.col("y").first(),
        x=pl.col("x").last(), y=pl.col("y").last(), z=pl.col("z").last(),
    )
    g = g.with_columns(
        grenade_type=pl.col("grenade_type").cast(pl.Utf8).str.to_lowercase()
        .str.replace(r"^(c?weapon_|c)", "").str.replace("projectile", "").str.replace("grenade$", "")
    )
    g = assign_rounds(g.with_columns(tick=pl.col("throw_tick")), rounds)
    g = _add_round_time(g, rounds, tick_rate).drop("tick")
    return g.join(teams, on="steamid", how="left")


def _bomb(demo: Demo, rounds: pl.DataFrame, ticks: pl.DataFrame) -> pl.DataFrame:
    b = demo.bomb
    b = _pick(b, {
        "tick": ("tick",), "event": ("action", "event", "type"),
        "steamid": ("steamid", "player_steamid"), "x": ("x", "X"), "y": ("y", "Y"), "z": ("z", "Z"),
        "site": ("site", "bombsite"),
    }).with_columns(
        event=pl.col("event").cast(pl.Utf8).str.to_lowercase().str.replace("^bomb_", "").str.replace("ed$", ""),
        steamid=pl.col("steamid").cast(pl.Utf8), site=pl.col("site").cast(pl.Utf8),
    ).filter(pl.col("event").is_in(["plant", "defuse"]))
    b = assign_rounds(b, rounds)

    # If the parser gives no usable site letter, derive it from the planter's callout.
    places = ticks.select("round_num", "tick", "steamid", "place").sort("tick")
    b = b.sort("tick").join_asof(places, on="tick", by=["round_num", "steamid"], strategy="backward")
    site_from_place = pl.col("place").str.extract(r"(?i)bombsite\s*([AB])", 1).str.to_uppercase()
    return b.with_columns(
        site=pl.when(pl.col("site").is_in(["A", "B"])).then(pl.col("site")).otherwise(site_from_place)
    ).drop("place")


def parse(path: Path) -> tuple[str, dict[str, pl.DataFrame]]:
    path = Path(path)
    mid = match_id_for(path)
    demo = Demo(path)
    tick_rate = float(demo.tick_rate or 64.0)
    map_name = demo.header.get("map_name", "unknown")

    rounds = _rounds(demo)
    teams = _pick(demo.players, {"steamid": ("steamid",), "team": ("team_clan_name", "team_name")}) \
        .with_columns(pl.col("steamid").cast(pl.Utf8), pl.col("team").cast(pl.Utf8)) \
        .unique("steamid", keep="last")
    # Pug demos have no clan names; fall back to a stable label from the starting side.
    if teams["team"].null_count() == teams.height:
        teams = teams.drop("team")

    ticks = _ticks(demo, rounds, teams if "team" in teams.columns else teams.with_columns(team=pl.lit(None, pl.Utf8)),
                   tick_rate)
    if ticks["team"].null_count() == ticks.height:
        first = ticks.filter(pl.col("round_num") == ticks["round_num"].min()) \
            .group_by("steamid").agg(start=pl.col("side").first())
        teams = first.select("steamid", team=pl.concat_str(pl.lit("Team starting "), pl.col("start")))
        ticks = ticks.drop("team").join(teams, on="steamid", how="left")

    side_team = ticks.group_by("round_num", "side").agg(team=pl.col("team").mode().first())
    rounds = rounds.join(side_team.filter(pl.col("side") == "CT").select("round_num", ct_team="team"),
                         on="round_num", how="left") \
                   .join(side_team.filter(pl.col("side") == "T").select("round_num", t_team="team"),
                         on="round_num", how="left")

    kills = _kills(demo, rounds, teams, tick_rate)
    damages = _damages(demo, rounds)
    grenades = _grenades(demo, rounds, teams, tick_rate)
    bomb = _bomb(demo, rounds, ticks)

    plants = bomb.filter(pl.col("event") == "plant").group_by("round_num") \
        .agg(plant_tick=pl.col("tick").first(), plant_site=pl.col("site").first())
    rounds = rounds.join(plants, on="round_num", how="left")

    team_names = sorted(t for t in teams["team"].unique().to_list() if t)
    matches = pl.DataFrame({
        "match_id": [mid], "map_name": [map_name], "demo_path": [str(path)], "tick_rate": [tick_rate],
        "team_a": [team_names[0] if team_names else None],
        "team_b": [team_names[1] if len(team_names) > 1 else None],
        "n_rounds": [rounds.height],
    })

    frames = {"matches": matches, "rounds": rounds, "ticks": ticks, "kills": kills,
              "damages": damages, "grenades": grenades, "bomb": bomb}
    frames = {k: (v if "match_id" in v.columns else v.with_columns(match_id=pl.lit(mid))) for k, v in frames.items()}
    frames["ticks"] = frames["ticks"].with_columns(pl.lit(map_name).alias("map_name"))
    return mid, frames
