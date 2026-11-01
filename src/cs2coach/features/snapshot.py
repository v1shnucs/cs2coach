"""Team-level snapshot features per tick, used by the win-probability baseline.

Input is the canonical `ticks` table (one row per player per sampled tick)
joined with `rounds` and `bomb`. Output is one row per (match, round, tick).
"""

import polars as pl

FEATURES = [
    "t_sec",
    "ct_alive", "t_alive", "alive_diff",
    "ct_hp", "t_hp",
    "ct_armor", "t_armor",
    "ct_equip", "t_equip", "equip_diff",
    "ct_defusers",
    "bomb_planted", "since_plant",
    "ct_dist_bomb",
]


def _side_agg(side: str) -> list[pl.Expr]:
    is_side = (pl.col("side") == side.upper()) & pl.col("alive")
    p = side.lower()
    exprs = [
        is_side.sum().alias(f"{p}_alive"),
        pl.col("hp").filter(is_side).sum().alias(f"{p}_hp"),
        pl.col("armor").filter(is_side).sum().alias(f"{p}_armor"),
        pl.col("equip_value").filter(is_side).sum().alias(f"{p}_equip"),
    ]
    if side == "CT":
        exprs.append((is_side & pl.col("defuser")).sum().alias("ct_defusers"))
    return exprs


def build_snapshots(ticks: pl.DataFrame, rounds: pl.DataFrame, bomb: pl.DataFrame) -> pl.DataFrame:
    keys = ["match_id", "round_num", "tick"]
    snap = (
        ticks.group_by(keys)
        .agg([pl.col("t_sec").first(), *_side_agg("CT"), *_side_agg("T")])
        .sort(keys)
    )

    plants = bomb.filter(pl.col("event") == "plant").select(
        "match_id", "round_num",
        pl.col("tick").alias("plant_tick"),
        pl.col("x").alias("bomb_x"), pl.col("y").alias("bomb_y"),
    ).unique(["match_id", "round_num"])
    snap = snap.join(plants, on=["match_id", "round_num"], how="left")

    tickrate = 64.0
    snap = snap.with_columns(
        bomb_planted=(pl.col("plant_tick").is_not_null() & (pl.col("tick") >= pl.col("plant_tick"))).cast(pl.Int8),
    ).with_columns(
        since_plant=pl.when(pl.col("bomb_planted") == 1)
        .then((pl.col("tick") - pl.col("plant_tick")) / tickrate)
        .otherwise(0.0),
    )

    # Mean distance of alive CTs to the planted bomb (0 before plant).
    ct_pos = ticks.filter((pl.col("side") == "CT") & pl.col("alive")).select(keys + ["x", "y"])
    dist = (
        ct_pos.join(snap.select(keys + ["bomb_x", "bomb_y", "bomb_planted"]), on=keys)
        .filter(pl.col("bomb_planted") == 1)
        .with_columns(d=((pl.col("x") - pl.col("bomb_x")) ** 2 + (pl.col("y") - pl.col("bomb_y")) ** 2).sqrt())
        .group_by(keys).agg(ct_dist_bomb=pl.col("d").mean())
    )
    snap = snap.join(dist, on=keys, how="left").with_columns(pl.col("ct_dist_bomb").fill_null(0.0))

    snap = snap.with_columns(
        alive_diff=pl.col("ct_alive") - pl.col("t_alive"),
        equip_diff=pl.col("ct_equip") - pl.col("t_equip"),
    )

    winners = rounds.select("match_id", "round_num", (pl.col("winner") == "CT").cast(pl.Int8).alias("ct_win"))
    return snap.join(winners, on=["match_id", "round_num"], how="inner").drop("plant_tick", "bomb_x", "bomb_y")
