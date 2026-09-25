"""Quantifies what preprocessing changed. Produces CSVs in preprocessing/results/.

Nothing here makes or scores matching decisions. The ground truth is used only in the two
`gt_*` diagnostics, which check whether a representation *destroys* information
(e.g. makes records of different entities identical) - as allowed for validation.
"""
from __future__ import annotations

import logging

import polars as pl

from . import config as C
from . import transforms as T

RES = C.RESULTS_DIR
log = logging.getLogger("preprocess")


def _proc(split, source) -> pl.LazyFrame:
    return pl.scan_parquet(C.processed_path(split, source))


def _per_file(query) -> pl.DataFrame:
    """Run `query(lazyframe)` on each processed file separately and stack the results.
    Every file carries its own `split`/`source` columns, so grouped results are identical to a
    query over the concatenation - but peak memory is bounded by the largest single file."""
    return pl.concat([_c(query(_proc(sp, s))) for sp, s in _files()], how="vertical_relaxed")


def _c(lf) -> pl.DataFrame:
    return lf.collect(engine="streaming")


def _files():
    return [(sp, s) for sp in C.SPLITS for s in C.SOURCES]


def _order(df: pl.DataFrame) -> pl.DataFrame:
    keys = []
    if "split" in df.columns:
        keys.append(pl.col("split").replace_strict({"train": 0, "test": 1}, default=9, return_dtype=pl.Int8))
    keys += [c for c in ("source", "country", "field", "stage_no") if c in df.columns]
    return df.sort(keys) if keys else df


def _save(df: pl.DataFrame, name: str) -> pl.DataFrame:
    df.write_csv(RES / f"{name}.csv")
    log.info("  wrote %s.csv (%d rows)", name, df.height)
    return df


# --------------------------------------------------------------------------- 1. stage-by-stage effects + collisions

def stage_frame(split: str, source: str, field: str) -> tuple[pl.LazyFrame, list[str]]:
    """All intermediate stages for one field of one file. Streamed once to a temporary Parquet
    file (bounded memory) and returned as a lazy scan; the caller deletes the file."""
    raw = f"business_{field}_raw"
    prefix = "name" if field == "name" else "address"
    lf = _proc(split, source).select(
        raw=pl.col(raw), stored_norm=pl.col(f"{prefix}_norm"), stored_clean=pl.col(f"{prefix}_clean"),
        **({"stored_base": pl.col("name_base")} if field == "name" else {}))
    lf = (lf.with_columns(unicode=T.unicode_stage("raw"))
            .with_columns(lowercase=T.lowercase_stage("unicode"))
            .with_columns(whitespace=T.whitespace_stage("lowercase")))
    stages = ["raw", "unicode", "lowercase", "whitespace"]
    if field == "address":
        lf = lf.with_columns(placeholder=T.address_placeholder_stage("whitespace"))
        stages.append("placeholder")
    lf = lf.with_columns(punctuation=pl.col("stored_clean")).with_columns(diacritic_fold=T.fold_diacritics("punctuation"))
    stages += ["punctuation", "diacritic_fold"]
    if field == "name":
        lf = lf.with_columns(legal_removed=pl.col("stored_base"))
        stages.append("legal_removed")
    tmp = _tmp_path(split, source, field)
    lf.sink_parquet(tmp, compression="lz4")
    return pl.scan_parquet(tmp), stages


def _tmp_path(split, source, field):
    return C.PROCESSED_DIR / f"_tmp_stages_{split}_{source}_{field}.parquet"


def _cleanup_tmp():
    for p in C.PROCESSED_DIR.glob("_tmp_stages_*.parquet"):
        try:
            p.unlink()
        except OSError:
            log.warning("could not delete temp file %s", p)


STAGE_NOTES = {
    "unicode": "mojibake repair, NFKC, drop zero-width/format chars, control->space, unify quotes/dashes",
    "lowercase": "lower-case",
    "whitespace": "collapse/strip whitespace  (= *_norm, stored)",
    "placeholder": "drop whole segments NULL/null/<NULL>/N/A",
    "punctuation": "apostrophes joined, dotted acronyms joined, punctuation->space  (= *_clean, stored)",
    "diacritic_fold": "strip accents on Latin letters (NOT stored; on-demand transform)",
    "legal_removed": "drop legal-form tokens  (= name_base, stored as extra representation)",
}


def normalization_effects() -> tuple[pl.DataFrame, pl.DataFrame]:
    rows, examples = [], []
    for sp, s in _files():
        for field in ("name", "address"):
            lf, stages = stage_frame(sp, s, field)
            # consistency: recomputed whitespace stage must equal the stored *_norm column
            assert _c(lf.select((~pl.col("whitespace").eq_missing(pl.col("stored_norm"))).sum())).item() == 0, \
                f"norm mismatch {sp} {s} {field}"
            basic = _c(lf.select(
                records=pl.len(), n_nonnull_raw=pl.col("raw").is_not_null().sum(),
                **{f"nn_{st}": pl.col(st).is_not_null().sum() for st in stages},
                **{f"nu_{st}": pl.col(st).drop_nulls().n_unique() for st in stages},
                **{f"chg_{cur}": (~pl.col(cur).eq_missing(pl.col(prev))).sum() for prev, cur in zip(stages, stages[1:])},
                **{f"null_{cur}": (pl.col(cur).is_null() & pl.col(prev).is_not_null()).sum() for prev, cur in zip(stages, stages[1:])},
            )).row(0, named=True)
            for i, st in enumerate(stages):
                r = {"split": sp, "source": s, "field": field, "stage_no": i, "stage": st, "note": STAGE_NOTES.get(st, "original value"),
                     "records": basic["records"], "non_null": basic[f"nn_{st}"], "n_unique": basic[f"nu_{st}"]}
                if i == 0:
                    r.update(records_changed=0, pct_changed=0.0, became_null=0, new_collision_keys=0, rows_in_new_collisions=0,
                             cum_collision_keys=0, rows_in_cum_collisions=0)
                else:
                    prev = stages[i - 1]
                    chg = basic[f"chg_{st}"]
                    g = (lf.filter(pl.col(st).is_not_null())
                           .group_by(st).agg(n_prev=pl.col(prev).n_unique(), n_raw=pl.col("raw").n_unique(), n=pl.len())
                           .filter(pl.col("n_raw") > 1))
                    cum = _c(g)                                   # only colliding keys are materialised
                    new = cum.filter(pl.col("n_prev") > 1)
                    r.update(records_changed=int(chg), pct_changed=round(100 * chg / max(basic["n_nonnull_raw"], 1), 3),
                             became_null=int(basic[f"null_{st}"]),
                             new_collision_keys=new.height, rows_in_new_collisions=int(new["n"].sum() or 0),
                             cum_collision_keys=cum.height, rows_in_cum_collisions=int(cum["n"].sum() or 0))
                    if new.height:
                        keys = new.sort(pl.col(st).hash(C.SEED)).head(3)[st]
                        ex = _c(lf.filter(pl.col(st).is_in(keys.implode()))
                                .group_by(st).agg(pl.col(prev).unique().sort().head(4).alias("distinct_previous_values"))
                                .sort(st))
                        for e in ex.iter_rows(named=True):
                            examples.append({"split": sp, "source": s, "field": field, "stage": st, "merged_into": e[st],
                                             "distinct_previous_values": " || ".join(e["distinct_previous_values"])})
                    del cum, new
                rows.append(r)
            log.info("  effects %s %s %s done", sp, s, field)
            del lf
            _cleanup_tmp()
    eff = _order(pl.DataFrame(rows).with_columns(
        unique_reduction_vs_prev=(pl.col("n_unique").shift(1).over("split", "source", "field") - pl.col("n_unique")).fill_null(0),
        pct_unique_reduction_vs_raw=((1 - pl.col("n_unique") / pl.col("n_unique").first().over("split", "source", "field")) * 100).round(3)))
    return _save(eff, "normalization_effects"), _save(pl.DataFrame(examples), "collision_examples")


# --------------------------------------------------------------------------- 2. S1 key collisions (information loss inside the de-duplicated reference)

def _keys() -> dict:
    fold = T.fold_diacritics
    srt = lambda c: pl.col(c).list.sort().list.join(" ")
    return {
        "name_raw + address_raw": [pl.col("business_name_raw"), pl.col("business_address_raw")],
        "name_norm + address_norm": [pl.col("name_norm"), pl.col("address_norm")],
        "name_clean + address_clean": [pl.col("name_clean"), pl.col("address_clean")],
        "folded name_clean + folded address_clean": [fold("name_clean"), fold("address_clean")],
        "name_base + address_clean": [pl.col("name_base"), pl.col("address_clean")],
        "sorted name_tokens + sorted address_tokens": [srt("name_tokens"), srt("address_tokens")],
        "name_raw only": [pl.col("business_name_raw")],
        "name_clean only": [pl.col("name_clean")],
        "name_base only": [pl.col("name_base")],
        "address_clean only": [pl.col("address_clean")],
    }


def s1_key_collisions() -> pl.DataFrame:
    """S1 is de-duplicated on raw (name, address). A processed key that makes two S1 records identical
    means that representation can no longer tell two reference entities apart."""
    rows, examples = [], []
    for sp in C.SPLITS:
        for kname, exprs in _keys().items():
            kcols = [f"k{i}" for i in range(len(exprs))]
            g = _c(_proc(sp, "S1").select([e.alias(k) for e, k in zip(exprs, kcols)]).group_by(kcols).len())
            rep = g.filter(pl.col("len") > 1)
            n = int(g["len"].sum())
            rows.append({"split": sp, "source": "S1", "key": kname, "records": n, "distinct_keys": g.height,
                         "colliding_keys": rep.height, "records_in_collisions": int(rep["len"].sum() or 0),
                         "pct_records_in_collisions": round(100 * (rep["len"].sum() or 0) / n, 4),
                         "max_group": int(g["len"].max())})
            if len(exprs) == 2 and 0 < rep.height:
                keys = rep.sort(pl.struct(kcols).hash(C.SEED)).head(5).select(kcols)
                ex = _c(_proc(sp, "S1").with_columns([e.alias(k) for e, k in zip(exprs, kcols)])
                        .join(keys.lazy(), on=kcols, how="semi")
                        .select(pl.lit(sp).alias("split"), pl.lit(kname).alias("key"), "entity_id",
                                "business_name_raw", "business_address_raw", "country_clean", *kcols)).sort(kcols)
                examples.append(ex.with_columns(pl.col(kcols).cast(pl.String)))
    if examples:
        _save(pl.concat(examples, how="diagonal"), "s1_key_collision_examples")
    return _save(pl.DataFrame(rows), "s1_key_collisions")


# --------------------------------------------------------------------------- 3. GT diagnostic: do collisions in S2/S3 merge different entities?

def _gt_pairs() -> pl.LazyFrame:
    return (pl.scan_parquet(C.GT_PROCESSED).select(s1_id="source1_entity_id", entity_id="matched_entity_ids")
            .explode("entity_id").drop_nulls("entity_id"))


def gt_s2s3_collision_purity() -> pl.DataFrame:
    """Train S2+S3 records grouped by a representation key. A group is 'mixed' when its records belong
    to >= 2 different S1 entities (unmatched records ignored). Compares raw vs processed keys."""
    base = pl.concat([_proc("train", "S2"), _proc("train", "S3")]).join(_gt_pairs(), on="entity_id", how="left")
    rows = []
    for kname, exprs in _keys().items():
        if kname.startswith("address_clean only"):
            continue
        g = _c(base.select([e.alias(f"k{i}") for i, e in enumerate(exprs)] + [pl.col("s1_id")])
               .drop_nulls([f"k{i}" for i in range(len(exprs))])
               .group_by([f"k{i}" for i in range(len(exprs))])
               .agg(n=pl.len(), n_entities=pl.col("s1_id").drop_nulls().n_unique()))
        grp = g.filter(pl.col("n") > 1)
        mixed = grp.filter(pl.col("n_entities") > 1)
        rows.append({"key": kname, "records": int(g["n"].sum()), "groups_size_gt1": grp.height,
                     "records_in_groups_gt1": int(grp["n"].sum() or 0),
                     "mixed_entity_groups": mixed.height, "records_in_mixed_groups": int(mixed["n"].sum() or 0),
                     "pct_records_in_mixed_groups": round(100 * (mixed["n"].sum() or 0) / g["n"].sum(), 3)})
    return _save(pl.DataFrame(rows), "gt_s2s3_collision_purity")


def gt_representation_agreement(pct: int = 5) -> pl.DataFrame:
    """For a hash sample of S1 entities: % of their true pairs whose representations are *exactly equal*.
    Shows whether normalisation preserves / recovers agreement between records of the same entity."""
    cols = ["entity_id", "business_name_raw", "business_address_raw", "name_norm", "name_clean", "name_base",
            "address_norm", "address_clean", "name_tokens", "address_tokens", "address_numbers", "postal_code_candidate",
            "address_missing", "country_clean"]
    s1 = _proc("train", "S1").select(cols).filter(pl.col("entity_id").hash(C.SEED) % 100 < pct)
    oth = pl.concat([_proc("train", "S2"), _proc("train", "S3")]).select(cols + ["source"])
    pairs = _c(_gt_pairs().filter(pl.col("s1_id").hash(C.SEED) % 100 < pct)).lazy()
    oth = oth.join(pairs.select("entity_id"), on="entity_id", how="semi")   # keep only sampled records (low memory)
    df = _c(pairs.join(s1.rename(lambda c: f"a_{c}"), left_on="s1_id", right_on="a_entity_id")
            .join(oth.rename(lambda c: f"b_{c}"), left_on="entity_id", right_on="b_entity_id"))
    fold = T.fold_diacritics
    eq = lambda x, y: (x == y)
    srt = lambda c: pl.col(c).list.sort().list.join(" ")
    reps = {
        "name raw": eq(pl.col("a_business_name_raw"), pl.col("b_business_name_raw")),
        "name norm": eq(pl.col("a_name_norm"), pl.col("b_name_norm")),
        "name clean": eq(pl.col("a_name_clean"), pl.col("b_name_clean")),
        "name clean folded": eq(fold("a_name_clean"), fold("b_name_clean")),
        "name base": eq(pl.col("a_name_base"), pl.col("b_name_base")),
        "name sorted tokens": eq(srt("a_name_tokens"), srt("b_name_tokens")),
        "address raw": eq(pl.col("a_business_address_raw"), pl.col("b_business_address_raw")),
        "address norm": eq(pl.col("a_address_norm"), pl.col("b_address_norm")),
        "address clean": eq(pl.col("a_address_clean"), pl.col("b_address_clean")),
        "address clean folded": eq(fold("a_address_clean"), fold("b_address_clean")),
        "address sorted tokens": eq(srt("a_address_tokens"), srt("b_address_tokens")),
        "address numbers (as sets)": eq(pl.col("a_address_numbers").list.unique().list.sort(), pl.col("b_address_numbers").list.unique().list.sort())
                                     & (pl.col("a_address_numbers").list.len() > 0),
    }
    out = (df.group_by(country="a_country_clean", match_source="b_source")
             .agg(pairs=pl.len(), **{k: (e.fill_null(False).mean() * 100).round(2) for k, e in reps.items()})
             .sort("country", "match_source"))
    long = (out.unpivot(index=["country", "match_source", "pairs"], variable_name="representation", value_name="pct")
               .with_columns(col=pl.format("pct_equal_{}_S1-{} (n={})", "country", "match_source", "pairs")))
    return _save(long.pivot(on="col", index="representation", values="pct"), "gt_representation_agreement")


# --------------------------------------------------------------------------- 4. source comparison after preprocessing

FLAG_COLS = ["name_missing", "address_missing", "country_missing",
             "name_is_upper", "name_is_lower", "name_has_digits", "name_has_non_latin", "name_has_latin_diacritics",
             "name_has_domain", "name_is_domain_only", "name_has_special_chars", "name_has_brackets", "name_has_leading_junk",
             "name_has_repeated_whitespace", "name_has_encoding_artifact", "name_has_legal_form", "name_legal_at_start",
             "name_legal_only", "name_clean_empty",
             "address_is_upper", "address_has_digits", "address_has_non_latin", "address_has_latin_diacritics",
             "address_has_special_chars", "address_has_encoding_artifact", "address_has_placeholder", "address_placeholder_only",
             "address_has_zero_padded_number", "address_has_landmark", "address_has_care_of"]


def source_comparison() -> tuple[pl.DataFrame, pl.DataFrame]:
    def aggs():
        return [pl.len().alias("records"),
                *[(pl.col(f).mean() * 100).round(3).alias(f"pct_{f}") for f in FLAG_COLS],
                (pl.col("postal_code_candidate").is_not_null().mean() * 100).round(3).alias("pct_postal_code_candidate"),
                (pl.col("name_numbers").list.len() > 0).mean().mul(100).round(3).alias("pct_name_has_numbers"),
                (pl.col("address_numbers").list.len() > 0).mean().mul(100).round(3).alias("pct_address_has_numbers"),
                pl.col("name_n_tokens").mean().round(3).alias("name_tokens_mean"),
                pl.col("name_n_tokens").median().alias("name_tokens_median"),
                pl.col("address_n_tokens").mean().round(3).alias("address_tokens_mean"),
                pl.col("address_n_tokens").median().alias("address_tokens_median"),
                pl.col("address_n_segments").mean().round(3).alias("address_segments_mean"),
                pl.col("business_name_raw").str.len_chars().mean().round(2).alias("name_len_raw_mean"),
                pl.col("name_clean").str.len_chars().mean().round(2).alias("name_len_clean_mean"),
                pl.col("business_address_raw").str.len_chars().mean().round(2).alias("address_len_raw_mean"),
                pl.col("address_clean").str.len_chars().mean().round(2).alias("address_len_clean_mean")]
    by_src = _order(_per_file(lambda lf: lf.group_by("split", "source").agg(*aggs())))
    by_cty = _order(_per_file(lambda lf: lf.group_by("split", "source", country="country_clean").agg(*aggs())))
    return _save(by_src, "source_comparison"), _save(by_cty, "source_comparison_by_country")


def uniqueness_by_source() -> pl.DataFrame:
    """Distinct values per representation, per file (full data)."""
    rows = []
    for sp, s in _files():
        r = _c(_proc(sp, s).select(
            records=pl.len(),
            name_raw=pl.col("business_name_raw").n_unique(), name_norm=pl.col("name_norm").n_unique(),
            name_clean=pl.col("name_clean").n_unique(), name_base=pl.col("name_base").n_unique(),
            address_raw=pl.col("business_address_raw").drop_nulls().n_unique(), address_norm=pl.col("address_norm").drop_nulls().n_unique(),
            address_clean=pl.col("address_clean").drop_nulls().n_unique(),
        )).with_columns(split=pl.lit(sp), source=pl.lit(s))
        rows.append(r)
    df = pl.concat(rows)
    df = df.select("split", "source", "records",
                   *[c for c in df.columns if c.startswith("name_")],
                   ((1 - pl.col("name_clean") / pl.col("name_raw")) * 100).round(3).alias("name_pct_unique_reduction_raw_to_clean"),
                   ((1 - pl.col("name_base") / pl.col("name_raw")) * 100).round(3).alias("name_pct_unique_reduction_raw_to_base"),
                   *[c for c in df.columns if c.startswith("address_")],
                   ((1 - pl.col("address_clean") / pl.col("address_raw")) * 100).round(3).alias("address_pct_unique_reduction_raw_to_clean"))
    return _save(_order(df), "uniqueness_by_source")


# --------------------------------------------------------------------------- 5. descriptive distributions

def distributions():
    _save(_order(_per_file(lambda lf: lf.group_by("split", "source", "country_raw", "country_clean").len("records"))), "country_values")
    miss = _per_file(lambda lf: lf.group_by("split", "source").agg(
        records=pl.len(),
        name_null_raw=pl.col("business_name_raw").is_null().sum(), address_null_raw=pl.col("business_address_raw").is_null().sum(),
        country_null_raw=pl.col("country_raw").is_null().sum(),
        name_norm_null=pl.col("name_norm").is_null().sum(), name_clean_null=pl.col("name_clean").is_null().sum(),
        name_base_null=pl.col("name_base").is_null().sum(),
        address_norm_null=pl.col("address_norm").is_null().sum(), address_clean_null=pl.col("address_clean").is_null().sum(),
        address_placeholder_only=pl.col("address_placeholder_only").sum(), country_clean_null=pl.col("country_clean").is_null().sum()))
    _save(_order(miss), "missing_values")
    for fld in ("name_script", "address_script"):
        d = _per_file(lambda lf: lf.group_by("split", "source", country="country_clean", script=fld).len("records"))
        d = d.with_columns(pct=(pl.col("records") / pl.col("records").sum().over("split", "source", "country") * 100).round(3))
        _save(_order(d), f"{fld}_distribution")
    forms = _per_file(lambda lf: lf.select("split", "source", country="country_clean", form="name_legal_forms")
                      .explode("form").drop_nulls("form").group_by("split", "source", "country", "form").len("occurrences"))
    tot = _per_file(lambda lf: lf.group_by("split", "source", country="country_clean").len("records"))
    forms = forms.join(tot, on=["split", "source", "country"]).with_columns(
        per_100_records=(pl.col("occurrences") / pl.col("records") * 100).round(3))
    _save(_order(forms).sort("split", "source", "country", "occurrences", descending=[False, False, False, True]), "legal_forms")
    tld = _per_file(lambda lf: lf.filter(pl.col("name_domain").is_not_null())
                    .group_by("split", "source", tld=pl.col("name_domain").str.extract(r"\.([^.]+(?:\.in)?)$", 1)).len("records"))
    _save(_order(tld).sort("split", "source", "records", descending=[False, False, True]), "name_domain_tlds")
    pc = _per_file(lambda lf: lf.group_by("split", "source", country="country_clean").agg(
        records=pl.len(), with_candidate=pl.col("postal_code_candidate").is_not_null().sum(),
        example=pl.col("postal_code_candidate").drop_nulls().sort().first()))
    _save(_order(pc), "postal_code_candidates")
    # token-count distribution after cleaning (for plots / later sizing)
    tc = _per_file(lambda lf: lf.group_by("split", "source", n=pl.col("name_n_tokens").clip(0, 15)).len("records")
                   .with_columns(field=pl.lit("name")))
    ta = _per_file(lambda lf: lf.group_by("split", "source", n=pl.col("address_n_tokens").clip(0, 40)).len("records")
                   .with_columns(field=pl.lit("address")))
    _save(_order(pl.concat([tc, ta])).sort("split", "source", "field", "n"), "token_count_distribution")


# --------------------------------------------------------------------------- 6. noise that preprocessing deliberately leaves in place

def remaining_noise() -> tuple[pl.DataFrame, pl.DataFrame]:
    nc, ac = pl.col("name_clean"), pl.col("address_clean")
    patterns = {
        "name: letter-digit mixed token (typo such as hea1th, or real e.g. 3m)": nc.str.contains(r"\p{L}\d|\d\p{L}"),
        "name: non-Latin script (untransliterated)": pl.col("name_has_non_latin"),
        "name: whole name is a domain (words concatenated)": pl.col("name_is_domain_only"),
        "name: legal form first token (reordered)": pl.col("name_legal_at_start"),
        "name: very long token (>= 18 chars, concatenation)": pl.col("name_tokens").list.eval(pl.element().str.len_chars() >= 18).list.any(),
        "name: honorific / prefix token (m s, shri, sri, the, dr)": nc.str.contains(r"^(?:m s|shri|sri|smt|the|dr|mr|mrs|ms)\b"),
        "name: repeated token": pl.col("name_tokens").list.len() > pl.col("name_tokens").list.unique().list.len(),
        "name: '&' present (vs 'and')": nc.str.contains("&", literal=True),
        "name: 'and' present": nc.str.contains(r"\band\b"),
        "address: abbreviated street type (st rd ave dr ln blvd r)": ac.str.contains(r"\b(?:st|rd|ave|av|dr|ln|blvd|bd|r)\b"),
        "address: full street type (street road avenue drive lane boulevard rue)": ac.str.contains(r"\b(?:street|road|avenue|drive|lane|boulevard|rue)\b"),
        "address: non-Latin script": pl.col("address_has_non_latin"),
        "address: US state 2-letter code as a token": (pl.col("country_clean") == "US") & ac.str.contains(r"\b(?:al|ak|az|ar|ca|co|ct|de|fl|ga|hi|id|il|in|ia|ks|ky|la|me|md|ma|mi|mn|ms|mo|mt|ne|nv|nh|nj|nm|ny|nc|nd|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|vt|va|wa|wv|wi|wy|dc)$"),
        "address: zero-padded number": pl.col("address_has_zero_padded_number"),
        "address: 'null'/'none'/'n a' token inside a segment": ac.str.contains(r"\b(?:null|none|n a)\b"),
        "address: letter-digit mixed token": ac.str.contains(r"\p{L}\d|\d\p{L}"),
        "address: single segment only": pl.col("address_n_segments") == 1,
    }
    rates = _order(_per_file(lambda lf: lf.group_by("split", "source").agg(
        pl.len().alias("records"), *[(e.fill_null(False).mean() * 100).round(3).alias(k) for k, e in patterns.items()])))
    rates = (rates.unpivot(index=["split", "source", "records"], variable_name="pattern", value_name="pct_records")
                  .with_columns(col=pl.format("pct_{}_{}", "split", "source")))
    rates = rates.pivot(on="col", index="pattern", values="pct_records")
    ex = []
    for k, e in patterns.items():
        for s in C.SOURCES:
            d = _c(_proc("train", s)
                   .filter(e.fill_null(False) & (pl.col("entity_id").hash(C.SEED) % 50 == 0))
                   .select(pl.lit(k).alias("pattern"), "entity_id", "country_clean", "business_name_raw", "name_clean",
                           "business_address_raw", "address_clean").head(2))
            ex.append(d)
    fr = []
    for k in ["name: letter-digit mixed token (typo such as hea1th, or real e.g. 3m)", "address: abbreviated street type (st rd ave dr ln blvd r)"]:
        fr.append(_c(_proc("test", "S2").filter(patterns[k].fill_null(False) & (pl.col("country_clean") == "France")
                                                & (pl.col("entity_id").hash(C.SEED) % 50 == 0))
                     .select(pl.lit(k).alias("pattern"), "entity_id", "country_clean", "business_name_raw", "name_clean",
                             "business_address_raw", "address_clean").head(2)))
    return _save(rates, "remaining_noise"), _save(pl.concat(ex + fr), "remaining_noise_examples")


def representation_examples() -> pl.DataFrame:
    """Real records showing every stored representation side by side (hash sample, all files)."""
    out = []
    interesting = (pl.col("name_has_special_chars") | pl.col("name_has_non_latin") | pl.col("name_is_domain_only")
                   | pl.col("address_has_placeholder") | pl.col("name_legal_at_start") | pl.col("address_has_encoding_artifact")
                   | pl.col("name_has_latin_diacritics") | pl.col("business_name_raw").str.contains(r"\p{L}\.\p{L}\.|\p{L}'\p{L}"))
    for sp, s in _files():
        d = _c(_proc(sp, s).filter(interesting & (pl.col("entity_id").hash(C.SEED) % 20 == 0))
               .group_by("country_clean").head(3)
               .with_columns([pl.col(c).list.join(" | ") for c in ("name_tokens", "name_legal_forms", "name_numbers",
                                                                     "address_segments", "address_numbers")])
               .select("entity_id", "split", "source", "country_clean", "business_name_raw", "name_norm", "name_clean",
                       "name_tokens", "name_base", "name_legal_forms", "name_domain", "name_script",
                       "business_address_raw", "address_norm", "address_clean", "address_segments", "address_numbers",
                       "postal_code_candidate"))
        out.append(d.sort("entity_id"))
    return _save(pl.concat(out), "representation_examples")


# --------------------------------------------------------------------------- driver

def run_all(step=None):
    """`step(name)` is an optional context-manager factory (timing / memory logging)."""
    from contextlib import nullcontext
    step = step or (lambda name: nullcontext())
    RES.mkdir(parents=True, exist_ok=True)
    steps = [("preprocessing summary", preprocessing_summary),
             ("distributions", distributions), ("source comparison", source_comparison),
             ("uniqueness by source", uniqueness_by_source), ("normalization effects", normalization_effects),
             ("S1 key collisions", s1_key_collisions), ("GT collision purity (validation only)", gt_s2s3_collision_purity),
             ("GT representation agreement (validation only)", gt_representation_agreement),
             ("remaining noise", remaining_noise), ("representation examples", representation_examples)]
    try:
        for name, fn in steps:
            with step(name):
                fn()
    finally:
        _cleanup_tmp()


def preprocessing_summary() -> pl.DataFrame:
    rows = []
    for sp, s in _files():
        raw_n = pl.scan_csv(C.raw_path(sp, s), separator="\t", quote_char=None, infer_schema=False).select(pl.len()).collect().item()
        r = _c(_proc(sp, s).select(
            records_out=pl.len(), unique_ids=pl.col("entity_id").n_unique(),
            name_missing=pl.col("name_missing").sum(), address_missing=pl.col("address_missing").sum(),
            country_missing=pl.col("country_missing").sum(), address_placeholder_only=pl.col("address_placeholder_only").sum(),
            name_clean_empty=pl.col("name_clean_empty").sum(), name_legal_only=pl.col("name_legal_only").sum(),
            name_changed_by_norm=(pl.col("business_name_raw") != pl.col("name_norm")).sum(),
            name_changed_by_clean=(pl.col("name_norm") != pl.col("name_clean")).sum(),
            address_changed_by_norm=(pl.col("business_address_raw") != pl.col("address_norm")).sum(),
            address_changed_by_clean=(pl.col("address_norm") != pl.col("address_clean")).sum(),
        )).row(0, named=True)
        rows.append({"split": sp, "source": s, "records_in": raw_n, **r,
                     "rows_dropped": raw_n - r["records_out"],
                     "raw_size_mb": round(C.raw_path(sp, s).stat().st_size / 1e6, 1),
                     "parquet_size_mb": round(C.processed_path(sp, s).stat().st_size / 1e6, 1)})
    df = pl.DataFrame(rows).with_columns(
        [(pl.col(c) / pl.col("records_out") * 100).round(3).alias(f"pct_{c}") for c in
         ("name_changed_by_norm", "name_changed_by_clean", "address_changed_by_norm", "address_changed_by_clean")])
    return _save(_order(df), "preprocessing_summary")
