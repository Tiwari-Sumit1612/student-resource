"""Candidate-generation recall study (measurement only).

    python -m modeling.candidates.study sample   # 10% hash sample of train S1 entities: channels, caps x top-k curve
    python -m modeling.candidates.study full     # recommended configuration on full train
    python -m modeling.candidates.study test     # unlabeled test: France block + US/India query sample (counts only)

Query side = S2+S3 records; index side = S1 of the same split and country block (country_clean is a hard block).
Ground truth is read only by evaluate.py.
"""
from __future__ import annotations

import json
import sys
import time

import polars as pl

from . import channels as CH
from . import evaluate as EV
from .common import COL, RESULTS, SEED, Meter, collect, index_frame, n_rows, proc, query_frame, save

COUNTRIES_TRAIN = ["US", "India"]
SAMPLE_PCT = 10
KS = [5, 10, 20, 30]
CAP_SETTINGS = {"tight": {"D": 300, "F": 20, "G": 200},
                "medium": {"D": 2000, "F": 100, "G": 1000},
                "loose": {"D": 10000, "F": 500, "G": 5000}}
E_CAP = 0.02
DEFAULT = ("medium", 10)
# Recommended configuration: chosen from the 10% sample curve + ablation (results/sample/ablation.csv),
# then confirmed on full train. channel -> top-k (exact channel C: k unused). A and B are subsumed by C.
RECOMMENDED = {"channels": {"C": 1, "E'": 15, "F": 5, "G": 15}, "caps": "medium"}


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _cut(c: pl.DataFrame, k: int) -> pl.DataFrame:
    return c.filter(pl.col("rank") <= k)


def _sample_block(country: str, pct: int):
    idx = index_frame("train", country)
    qall = query_frame("train", country)
    truth_all = EV.truth_pairs(idx, qall)
    samp = idx.filter(pl.col(COL.entity_id).hash(SEED) % 100 < pct).select("s1")
    matched = truth_all.join(samp, on="s1", how="semi").select("q").unique()
    unmatched = (qall.join(truth_all.select("q").unique(), on="q", how="anti")
                 .filter(pl.col(COL.entity_id).hash(SEED) % 100 < pct).select("q"))
    qdf = qall.join(pl.concat([matched, unmatched]), on="q", how="semi")
    truth = truth_all.join(samp, on="s1", how="semi").drop("tid").with_row_index("tid")
    return idx, qdf, truth


def _write_summaries(acc: EV.Accumulator, names: list[str], meters: dict, sub: str):
    tables = {"channel_summary": acc.summary(names, meters),
              "by_country": acc.summary(names, group="country"),
              "by_source": acc.summary(names, group="src"),
              "slice_recall": acc.slice_recall(names)}
    for k, v in tables.items():
        save(v, k, sub)
    return tables


# =========================================================================== sample study

def run_sample(pct: int = SAMPLE_PCT):
    n_q, n_s1 = n_rows("train", "S2") + n_rows("train", "S3"), n_rows("train", "S1")
    main = EV.Accumulator(n_q, n_s1)
    curve = {(s, k): EV.Accumulator(n_q, n_s1) for s in CAP_SETTINGS for k in KS}
    meters, params = {}, {}
    names = ["A", "B", "C", "D", "E", "E'", "F", "G", "UNION all", "UNION recommended"]

    for country in COUNTRIES_TRAIN:
        t0 = time.time()
        idx, qdf, truth = _sample_block(country, pct)
        _log(f"{country}: index {idx.height:,} queries {qdf.height:,} true pairs {truth.height:,} ({time.time() - t0:.0f}s)")
        main.new_block(country, idx, qdf, truth)
        for a in curve.values():
            a.new_block(country, idx, qdf, truth)
        got = {}

        def run(label, ch):
            m = meters.setdefault(label, Meter())
            with m.measure():
                ch.prepare(idx)
                got[label] = ch.retrieve(qdf)
            params.setdefault(label, {})[country] = ch.params
            _log(f"  {label:3s} {ch.name:34s} {m.seconds:7.1f}s  pairs {got[label].height:>11,}")

        run("A", CH.channel_A()); run("B", CH.channel_B()); run("C", CH.channel_C())
        run("E", CH.channel_E(E_CAP, 10))
        run("E'", CH.channel_E_region(E_CAP, max(KS)))
        exact = [got[x].select("q", "s1") for x in ("A", "B", "C")]

        for setting, caps in CAP_SETTINGS.items():
            ranked = {}
            for lab, mk in (("D", CH.channel_D), ("F", CH.channel_F), ("G", CH.channel_G)):
                ch = mk(caps[lab], max(KS))
                if setting == DEFAULT[0]:
                    run(lab, ch)
                    ranked[lab] = got[lab]
                else:
                    ch.prepare(idx)
                    ranked[lab] = ch.retrieve(qdf)
                    params.setdefault(f"{lab} ({setting})", {})[country] = ch.params
            ranked["E'"] = got["E'"]
            for k in KS:
                u = pl.concat(exact + [_cut(c, k).select("q", "s1") for c in ranked.values()]).unique()
                curve[(setting, k)].add("UNION", u)
            _log(f"  caps {setting}: curve points done")
            if setting == DEFAULT[0]:
                k = DEFAULT[1]
                cut = {lab: (got[lab] if lab in ("A", "B", "C", "E") else _cut(got[lab], k)).select("q", "s1")
                       for lab in ["A", "B", "C", "D", "E", "E'", "F", "G"]}
                for lab, c in cut.items():
                    main.add(lab, c)
                main.add("UNION all", pl.concat(list(cut.values())).unique())
                rec_parts = [(got[l] if l in ("A", "B", "C") else _cut(got[l], kk)).select("q", "s1")
                             for l, kk in RECOMMENDED["channels"].items()]
                main.add("UNION recommended", pl.concat(rec_parts).unique())
        main.end_block()
        for a in curve.values():
            a.end_block()
        del idx, qdf, truth, got

    sub = "sample"
    tables = _write_summaries(main, names, meters, sub)
    rows = []
    for (setting, k), a in curve.items():
        for grp in (None, "country"):
            s = a.summary(["UNION"], group=grp)
            s = s.with_columns(caps=pl.lit(setting), k=pl.lit(k), **{f"cap_{c}": pl.lit(v) for c, v in CAP_SETTINGS[setting].items()})
            rows.append(s.select([c for c in ["caps", "k", "cap_D", "cap_F", "cap_G", "country", "pair_recall", "entity_recall",
                                              "cand_per_record_mean", "cand_per_record_median", "cand_per_record_p95",
                                              "cand_per_record_max"] if c in s.columns]))
    save(pl.concat(rows, how="diagonal").sort("caps", "k", "country", nulls_last=False), "union_curve", sub)
    (RESULTS / sub / "channel_params.json").write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
    with pl.Config(tbl_cols=30, tbl_width_chars=260, tbl_hide_dataframe_shape=True, tbl_rows=60):
        print(tables["channel_summary"])
        print(tables["slice_recall"])
        print(pl.read_csv(RESULTS / sub / "union_curve.csv").filter(pl.col("country").is_null()))


# =========================================================================== ablation on the sample

ABLATIONS = {
    "all 7 @10 (A,B,C,D,E',F,G)": {"A": 1, "B": 1, "C": 1, "D": 10, "E'": 10, "F": 10, "G": 10},
    "- A": {"B": 1, "C": 1, "D": 10, "E'": 10, "F": 10, "G": 10},
    "- B": {"A": 1, "C": 1, "D": 10, "E'": 10, "F": 10, "G": 10},
    "- C": {"A": 1, "B": 1, "D": 10, "E'": 10, "F": 10, "G": 10},
    "- D": {"A": 1, "B": 1, "C": 1, "E'": 10, "F": 10, "G": 10},
    "- E'": {"A": 1, "B": 1, "C": 1, "D": 10, "F": 10, "G": 10},
    "- F": {"A": 1, "B": 1, "C": 1, "D": 10, "E'": 10, "G": 10},
    "- G": {"A": 1, "B": 1, "C": 1, "D": 10, "E'": 10, "F": 10},
    "A,B,C + E'20 G10 D5 F5": {"A": 1, "B": 1, "C": 1, "D": 5, "E'": 20, "F": 5, "G": 10},
    "A,B,C + E'15 G15 D5 F5": {"A": 1, "B": 1, "C": 1, "D": 5, "E'": 15, "F": 5, "G": 15},
    "A,B,C + E'20 G20 D5 F5": {"A": 1, "B": 1, "C": 1, "D": 5, "E'": 20, "F": 5, "G": 20},
    "B,C + E'15 G15 F5": {"B": 1, "C": 1, "E'": 15, "F": 5, "G": 15},
}


def run_ablation(pct: int = SAMPLE_PCT):
    """Medium caps; every channel computed once at k=30, unions evaluated for each config."""
    n_q, n_s1 = n_rows("train", "S2") + n_rows("train", "S3"), n_rows("train", "S1")
    accs = {name: EV.Accumulator(n_q, n_s1) for name in ABLATIONS}
    caps = CAP_SETTINGS["medium"]
    for country in COUNTRIES_TRAIN:
        idx, qdf, truth = _sample_block(country, pct)
        chans = {"A": CH.channel_A(), "B": CH.channel_B(), "C": CH.channel_C(), "D": CH.channel_D(caps["D"], 30),
                 "E'": CH.channel_E_region(E_CAP, 30), "F": CH.channel_F(caps["F"], 30), "G": CH.channel_G(caps["G"], 30)}
        got = {}
        for lab, ch in chans.items():
            ch.prepare(idx)
            got[lab] = ch.retrieve(qdf)
        _log(f"{country}: channels computed")
        for name, cfg in ABLATIONS.items():
            a = accs[name]
            a.new_block(country, idx, qdf, truth)
            a.add("UNION", pl.concat([_cut(got[l], k).select("q", "s1") for l, k in cfg.items()]).unique())
            a.end_block()
        del idx, qdf, truth, got
    rows = [a.summary(["UNION"]).with_columns(config=pl.lit(n)) for n, a in accs.items()]
    out = pl.concat(rows).select("config", "pair_recall", "entity_recall", "cand_per_record_mean", "cand_per_record_median",
                                 "cand_per_record_p95", "cand_per_record_max", "total_candidate_pairs")
    save(out, "ablation", "sample")
    with pl.Config(tbl_cols=20, tbl_width_chars=220, tbl_hide_dataframe_shape=True, tbl_rows=30, fmt_str_lengths=40):
        print(out)


# =========================================================================== configured channels

def make_channels(config: dict) -> dict:
    caps = CAP_SETTINGS[config["caps"]]
    makers = {"A": lambda k: CH.channel_A(), "B": lambda k: CH.channel_B(), "C": lambda k: CH.channel_C(),
              "D": lambda k: CH.channel_D(caps["D"], k), "E": lambda k: CH.channel_E(E_CAP, k),
              "E'": lambda k: CH.channel_E_region(E_CAP, k),
              "F": lambda k: CH.channel_F(caps["F"], k), "G": lambda k: CH.channel_G(caps["G"], k)}
    return {lab: makers[lab](k) for lab, k in config["channels"].items()}


def _run_blocks(split: str, blocks: list[tuple[str, pl.DataFrame, pl.DataFrame, pl.DataFrame | None]],
                config: dict, acc: EV.Accumulator, meters: dict, n_chunks: int):
    for country, idx, qdf, truth in blocks:
        acc.new_block(country, idx, qdf, truth)
        chans = make_channels(config)
        for lab, ch in chans.items():
            with meters.setdefault(lab, Meter()).measure():
                ch.prepare(idx)
        for i in range(n_chunks):
            qc = qdf.filter(pl.col("q") % n_chunks == i)
            parts = []
            for lab, ch in chans.items():
                with meters[lab].measure():
                    c = ch.retrieve(qc).select("q", "s1")
                acc.add(lab, c)
                parts.append(c)
            with meters.setdefault("UNION", Meter()).measure():
                u = pl.concat(parts).unique()
            acc.add("UNION", u)
            _log(f"  {split} {country} chunk {i + 1}/{n_chunks}: queries {qc.height:,} union pairs {u.height:,}")
        acc.end_block()


# =========================================================================== full train confirmation

def run_full(config: dict = RECOMMENDED, n_chunks: int = 8):
    sub = "full_train"
    n_q, n_s1 = n_rows("train", "S2") + n_rows("train", "S3"), n_rows("train", "S1")
    acc, meters = EV.Accumulator(n_q, n_s1), {}
    t0 = time.time()
    for country in COUNTRIES_TRAIN:
        idx, qdf = index_frame("train", country), query_frame("train", country)
        truth = EV.truth_pairs(idx, qdf)
        _log(f"{country}: index {idx.height:,} queries {qdf.height:,} true pairs {truth.height:,}")
        _run_blocks("train", [(country, idx, qdf, truth)], config, acc, meters, n_chunks)
        del idx, qdf, truth
    names = list(config["channels"]) + ["UNION"]
    tables = _write_summaries(acc, names, meters, sub)

    # S1 -> S2/S3 view by number of true matches of the S1 entity
    t = acc.truth_frame()
    s1b = pl.concat(acc.s1_block)
    nm = t.group_by("s1").agg(n_true=pl.len(), all_hit=pl.col("UNION").all())
    s1v = (s1b.join(nm, on="s1", how="left").with_columns(n_true=pl.col("n_true").fill_null(0))
           .with_columns(cands=pl.Series(acc.s1count["UNION"][s1b["s1"].to_numpy()]),
                         group=pl.when(pl.col("n_true") == 0).then(pl.lit("0 (singleton)"))
                         .when(pl.col("n_true") == 1).then(pl.lit("1")).when(pl.col("n_true") <= 3).then(pl.lit("2-3"))
                         .otherwise(pl.lit("4+"))))
    s1_view = (s1v.group_by("country", "group").agg(
        s1_entities=pl.len(), entity_recall=(pl.col("all_hit").mean() * 100).round(3),
        cand_mean=pl.col("cands").mean().round(2), cand_median=pl.col("cands").median(),
        cand_p95=pl.col("cands").quantile(0.95), cand_max=pl.col("cands").max()).sort("country", "group"))
    save(s1_view, "s1_view", sub)

    # true pairs no channel retrieves
    missed = t.filter(~pl.col("UNION"))
    by_slice = pl.DataFrame([{"slice": s, "missed_pairs": int(missed[s].sum()),
                              "pct_of_missed": round(float(missed[s].mean()) * 100, 2),
                              "slice_recall": round(float(t.filter(pl.col(s))["UNION"].mean()) * 100, 3)} for s in EV.SLICES]
                            + [{"slice": "(none of the slices)", "missed_pairs": int(missed.filter(~pl.any_horizontal(EV.SLICES)).height),
                                "pct_of_missed": round(missed.filter(~pl.any_horizontal(EV.SLICES)).height / max(missed.height, 1) * 100, 2),
                                "slice_recall": round(float(t.filter(~pl.any_horizontal(EV.SLICES))["UNION"].mean()) * 100, 3)}])
    save(by_slice, "missed_by_slice", sub)
    save(missed.group_by("country", "src").len("missed_pairs").sort("country", "src"), "missed_by_country_source", sub)
    ex = missed.with_columns(h=pl.struct("q", "s1").hash(SEED)).sort("h").head(20)
    n2 = n_rows("train", "S2")
    s1raw = collect(proc("train", "S1").with_row_index("s1").join(ex.select("s1").lazy(), on="s1", how="semi")
                    .select("s1", s1_name=COL.business_name_raw, s1_address=COL.business_address_raw))
    qraw = pl.concat([
        collect(proc("train", "S2").with_row_index("q").select("q", m_name=COL.business_name_raw, m_address=COL.business_address_raw)
                .join(ex.select("q").lazy(), on="q", how="semi")),
        collect(proc("train", "S3").with_row_index("q").with_columns(q=(pl.col("q") + n2).cast(pl.UInt32))
                .select("q", m_name=COL.business_name_raw, m_address=COL.business_address_raw)
                .join(ex.select("q").lazy(), on="q", how="semi"))])
    examples = (ex.join(s1raw, on="s1").join(qraw, on="q")
                .select("country", "src", *EV.SLICES, "s1_name", "m_name", "s1_address", "m_address"))
    save(examples, "missed_examples", sub)
    (RESULTS / sub / "config.json").write_text(json.dumps({"config": config, "caps": CAP_SETTINGS[config["caps"]],
                                                           "e_cap": E_CAP, "total_s": round(time.time() - t0)}, indent=2))
    with pl.Config(tbl_cols=30, tbl_width_chars=260, tbl_hide_dataframe_shape=True, tbl_rows=60, fmt_str_lengths=60):
        for k in ("channel_summary", "by_country", "by_source", "slice_recall"):
            print(k); print(tables[k])
        print(s1_view); print(by_slice); print(examples)


# =========================================================================== unlabeled test blocks

def run_test(config: dict = RECOMMENDED, pct: int = SAMPLE_PCT):
    """France: every test query. US/India: `pct`% own-hash sample of test queries against the full test index."""
    sub = "test_counts"
    n_q, n_s1 = n_rows("test", "S2") + n_rows("test", "S3"), n_rows("test", "S1")
    acc, meters = EV.Accumulator(n_q, n_s1), {}
    for country in ["France", "US", "India"]:
        idx, qdf = index_frame("test", country), query_frame("test", country)
        if country != "France":
            qdf = qdf.filter(pl.col(COL.entity_id).hash(SEED) % 100 < pct)
        _log(f"test {country}: index {idx.height:,} queries {qdf.height:,}")
        _run_blocks("test", [(country, idx, qdf, None)], config, acc, meters, 4 if country == "France" else 2)
    names = list(config["channels"]) + ["UNION"]
    by_c = save(acc.summary(names, group="country"), "by_country", sub)
    region = collect(pl.concat([proc("test", s).group_by(COL.country_clean).agg(
        pct_region_found=(pl.col(COL.address_region_found).mean() * 100).round(1), src=pl.lit(s)) for s in ("S1", "S2", "S3")]))
    save(region, "region_coverage", sub)
    with pl.Config(tbl_cols=30, tbl_width_chars=260, tbl_hide_dataframe_shape=True, tbl_rows=60):
        print(by_c); print(region)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sample"
    {"sample": run_sample, "ablate": run_ablation, "full": run_full, "test": run_test}[cmd]()
