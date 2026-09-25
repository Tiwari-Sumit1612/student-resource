"""Measurement only. The ONLY module that reads the ground truth; nothing here feeds a key or a rule."""
from __future__ import annotations

import numpy as np
import polars as pl

from preprocessing import config as PC

from .common import COL, fold

SLICES = ["nonlatin_match_name", "domain_only_name", "no_shared_name_token", "missing_address", "reordered_tokens"]


def truth_pairs(idx: pl.DataFrame, qdf: pl.DataFrame) -> pl.DataFrame:
    """True (q, s1) pairs of one train block, with the hard-slice flags."""
    gt = (pl.scan_parquet(PC.GT_PROCESSED).select(s1_eid="source1_entity_id", m_eid="matched_entity_ids")
          .explode("m_eid").drop_nulls("m_eid").collect())
    a = idx.select("s1", s1_eid=COL.entity_id, a_name=COL.name_clean)
    b = qdf.select("q", "src", m_eid=COL.entity_id, b_name=COL.name_clean, script=COL.name_script,
                   dom=COL.name_is_domain_only, miss=COL.address_missing)
    t = gt.join(a, on="s1_eid").join(b, on="m_eid")
    ta, tb = fold("a_name").str.split(" "), fold("b_name").str.split(" ")
    t = t.with_columns(
        nonlatin_match_name=~pl.col("script").is_in(["Latin", "None"]),
        domain_only_name=pl.col("dom"),
        no_shared_name_token=ta.list.set_intersection(tb).list.len() == 0,
        missing_address=pl.col("miss"),
        reordered_tokens=(fold("a_name") != fold("b_name")) & (ta.list.sort() == tb.list.sort()),
    )
    return t.select("q", "s1", "src", *SLICES).with_row_index("tid")


class Accumulator:
    """Collects per-channel hits and candidate counts over query chunks of one or more blocks."""

    def __init__(self, n_q: int, n_s1: int):
        self.n_q, self.n_s1 = n_q, n_s1
        self.truth: list[pl.DataFrame] = []
        self.hits: dict[str, list[np.ndarray]] = {}
        self.qcount: dict[str, np.ndarray] = {}
        self.s1count: dict[str, np.ndarray] = {}
        self.pairs: dict[str, int] = {}
        self.queries: list[pl.DataFrame] = []      # (q, src, country) evaluated
        self.s1_block: list[pl.DataFrame] = []     # (s1, country) of the index blocks

    def new_block(self, country: str, idx: pl.DataFrame, qdf: pl.DataFrame, truth: pl.DataFrame | None):
        self.queries.append(qdf.select("q", "src", country=pl.lit(country)))
        self.s1_block.append(idx.select("s1", country=pl.lit(country)))
        if truth is not None:
            self._truth = truth.with_columns(country=pl.lit(country))
            self.truth.append(self._truth)
            self._hit_block = {}
        else:
            self._truth = None

    def add(self, channel: str, cands: pl.DataFrame) -> None:
        if channel not in self.qcount:
            self.qcount[channel] = np.zeros(self.n_q, dtype=np.int32)
            self.s1count[channel] = np.zeros(self.n_s1, dtype=np.int32)
            self.pairs[channel] = 0
            self.hits[channel] = []
        c = cands.select("q", "s1")
        g = c.group_by("q").len()
        self.qcount[channel][g["q"].to_numpy()] += g["len"].to_numpy().astype(np.int32)
        self.s1count[channel] += np.bincount(c["s1"].to_numpy(), minlength=self.n_s1).astype(np.int32)
        self.pairs[channel] += c.height
        if self._truth is not None:
            hit = self._truth.join(c, on=["q", "s1"], how="semi")["tid"].to_numpy()
            arr = self._hit_block.setdefault(channel, np.zeros(self._truth.height, dtype=bool))
            arr[hit] = True

    def end_block(self) -> None:
        if self._truth is not None:
            for ch, arr in self._hit_block.items():
                self.hits[ch].append(arr)

    # ------------------------------------------------------------------ reporting
    def truth_frame(self) -> pl.DataFrame:
        t = pl.concat(self.truth, how="vertical_relaxed")
        for ch, parts in self.hits.items():
            t = t.with_columns(pl.Series(ch, np.concatenate(parts)))
        return t

    @staticmethod
    def _dist(x: np.ndarray, prefix: str) -> dict:
        if len(x) == 0:
            return {f"{prefix}_mean": None, f"{prefix}_median": None, f"{prefix}_p95": None, f"{prefix}_max": None}
        return {f"{prefix}_mean": round(float(x.mean()), 2), f"{prefix}_median": float(np.median(x)),
                f"{prefix}_p95": float(np.percentile(x, 95)), f"{prefix}_max": int(x.max())}

    def summary(self, channels: list[str], meters: dict | None = None, group: str | None = None) -> pl.DataFrame:
        """One row per channel (and per `group` value: country / src)."""
        q = pl.concat(self.queries)
        s1b = pl.concat(self.s1_block)
        t = self.truth_frame() if self.truth else None
        rows = []
        groups = [None] if group is None else sorted(q[group].unique().to_list())
        for g in groups:
            qg = q if g is None else q.filter(pl.col(group) == g)
            tg = None if t is None else (t if g is None else t.filter(pl.col(group) == g))
            for ch in channels:
                cnt = self.qcount[ch][qg["q"].to_numpy()]
                r = {"channel": ch} | ({group: g} if group else {})
                if tg is not None and tg.height:
                    r["pair_recall"] = round(float(tg[ch].mean()) * 100, 3)
                    ent = tg.group_by("s1").agg(pl.col(ch).all())
                    r["entity_recall"] = round(float(ent[ch].mean()) * 100, 3)
                    r["true_pairs"] = tg.height
                r["queries"] = qg.height
                r |= self._dist(cnt, "cand_per_record")
                r["total_candidate_pairs"] = int(cnt.sum())
                if group is None:
                    s1c = self.s1count[ch][s1b["s1"].to_numpy()]
                    r |= self._dist(s1c, "cand_per_s1")
                    if meters and ch in meters:
                        r |= {"runtime_s": round(meters[ch].seconds, 1), "peak_rss_gb": round(meters[ch].peak_gb, 2)}
                rows.append(r)
        return pl.DataFrame(rows)

    def slice_recall(self, channels: list[str]) -> pl.DataFrame:
        t = self.truth_frame()
        rows = []
        for sl in SLICES:
            ts = t.filter(pl.col(sl))
            rows.append({"slice": sl, "true_pairs": ts.height, "pct_of_pairs": round(ts.height / t.height * 100, 2),
                         **{ch: round(float(ts[ch].mean()) * 100, 2) if ts.height else None for ch in channels}})
        return pl.DataFrame(rows)
