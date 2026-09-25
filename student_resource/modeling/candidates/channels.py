"""Retrieval channels A-G. Keys are built only from processed columns (names from preprocessing.schema)
and the read-only preprocessing.transforms expressions. No ground truth is used anywhere in this module.

Every channel:   prepare(index_df)  once per country block (S1 side)
                 retrieve(query_df) -> DataFrame[q, s1, score, rank]   (rank 1 = best; exact channels rank 1)
DF / IDF statistics are computed on the unlabeled S1 index of the same split and country block.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import scipy.sparse as sp

from .common import COL, SEED, fold

TLD_RE = r"\.(?:co\.in|org\.in|net\.in|com|net|org|biz|info|in|co|us|fr|io)$"
OUT_SCHEMA = {"q": pl.UInt32, "s1": pl.UInt32, "score": pl.Float32, "rank": pl.UInt32}


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=OUT_SCHEMA)


def compact(e: pl.Expr) -> pl.Expr:
    return e.str.replace_all(" ", "", literal=True)


def domain_stem() -> pl.Expr:
    """richardsonmexico.com -> richardsonmexico ; www.abc-def.co.in -> abcdef"""
    return (pl.col(COL.name_domain).str.replace(r"^www\.", "").str.replace(TLD_RE, "")
            .str.replace_all(r"[.\-]", ""))


# =========================================================================== exact channels (A, B, C)

class Exact:
    kind = "exact"

    def __init__(self, name: str, s1_keys: list[pl.Expr], q_keys: list[pl.Expr]):
        self.name, self._s1_keys, self._q_keys = name, s1_keys, q_keys
        self.params = {}

    @staticmethod
    def _keys(df: pl.DataFrame, idcol: str, exprs: list[pl.Expr]) -> pl.DataFrame:
        parts = [df.select(pl.col(idcol), key=e) for e in exprs]
        return (pl.concat(parts).filter(pl.col("key").is_not_null() & (pl.col("key").str.len_chars() > 0))
                .unique())

    def prepare(self, idx: pl.DataFrame) -> None:
        self.s1k = self._keys(idx, "s1", self._s1_keys)
        self.params = {"distinct_s1_keys": self.s1k["key"].n_unique()}

    def retrieve(self, qdf: pl.DataFrame) -> pl.DataFrame:
        qk = self._keys(qdf, "q", self._q_keys)
        out = qk.join(self.s1k, on="key").select("q", "s1").unique()
        return out.with_columns(score=pl.lit(1.0, pl.Float32), rank=pl.lit(1, pl.UInt32)).cast(OUT_SCHEMA)


def channel_A() -> Exact:
    e = pl.col(COL.name_clean)
    return Exact("A exact name_clean", [e], [e])


def channel_B() -> Exact:
    e = fold(COL.name_base)
    return Exact("B exact fold(name_base)", [e], [e])


def channel_C() -> Exact:
    base, clean = compact(fold(COL.name_base)), compact(fold(COL.name_clean))
    return Exact("C compact key / domain stem", [base, clean], [domain_stem(), base])


# =========================================================================== IDF-weighted overlap (D, F, G)

class Overlap:
    """Inverted index over hashed keys. Keys whose S1 document frequency exceeds `df_cap` are dropped.
    Candidates are ranked by the summed IDF of shared keys; the top `k` per query are kept."""
    kind = "ranked"
    MAX_JOIN_ROWS = 40_000_000

    def __init__(self, name: str, postings, df_cap: int, k: int):
        self.name, self._postings, self.df_cap, self.k = name, postings, df_cap, k
        self.params = {"df_cap": df_cap, "k": k}

    def prepare(self, idx: pl.DataFrame) -> None:
        post = self._postings(idx, "s1").unique()
        n = idx.height
        df = post.group_by("key").agg(df=pl.len())
        self.keep = df.filter(pl.col("df") <= self.df_cap).with_columns(
            idf=(pl.lit(math.log(n)) - pl.col("df").cast(pl.Float64).log()).cast(pl.Float32))
        self.post = post.join(self.keep.select("key", "idf"), on="key")
        self.params |= {"s1_keys": df.height, "s1_keys_kept": self.keep.height,
                        "pct_s1_postings_kept": round(self.post.height / max(post.height, 1) * 100, 2)}

    def retrieve(self, qdf: pl.DataFrame) -> pl.DataFrame:
        qp = self._postings(qdf, "q").unique()
        if qp.height == 0:
            return _empty()
        est = qp.join(self.keep.select("key", "df"), on="key")["df"].sum() or 0
        n_chunks = max(1, math.ceil(est / self.MAX_JOIN_ROWS))
        out = []
        for i in range(n_chunks):
            part = qp.filter(pl.col("q") % n_chunks == i) if n_chunks > 1 else qp
            pairs = (part.join(self.post, on="key").group_by("q", "s1").agg(score=pl.col("idf").sum()))
            top = (pairs.group_by("q").agg(c=pl.struct("s1", "score").top_k_by(["score", "s1"], self.k, reverse=[False, True]))
                   .explode("c").unnest("c"))
            out.append(top)
        res = pl.concat(out).sort(["q", "score", "s1"], descending=[False, True, False])
        return res.with_columns(rank=pl.int_range(1, pl.len() + 1).over("q")).cast(OUT_SCHEMA)


def _name_token_postings(df: pl.DataFrame, idcol: str) -> pl.DataFrame:
    return (df.select(pl.col(idcol), tok=fold(COL.name_clean).str.split(" ")).explode("tok")
            .filter(pl.col("tok").str.len_chars() > 0).select(pl.col(idcol), key=pl.col("tok").hash(SEED)))


def _address_number_postings(df: pl.DataFrame, idcol: str) -> pl.DataFrame:
    return (df.filter(pl.col(COL.address_region).is_not_null())
            .select(pl.col(idcol), pl.col(COL.address_region), num=pl.col(COL.address_numbers)).explode("num")
            .drop_nulls("num")
            .select(pl.col(idcol), key=pl.concat_str([pl.col(COL.address_region), pl.lit("|"), pl.col("num")]).hash(SEED)))


def _address_token_postings(df: pl.DataFrame, idcol: str) -> pl.DataFrame:
    return (df.filter(pl.col(COL.address_region).is_not_null())
            .select(pl.col(idcol), pl.col(COL.address_region), tok=fold(COL.address_clean).str.split(" ")).explode("tok")
            .filter(pl.col("tok").str.len_chars() > 0)
            .select(pl.col(idcol), key=pl.concat_str([pl.col(COL.address_region), pl.lit("|"), pl.col("tok")]).hash(SEED)))


def channel_D(df_cap: int = 1000, k: int = 10) -> Overlap:
    return Overlap("D name tokens (IDF, DF cap)", _name_token_postings, df_cap, k)


def channel_F(df_cap: int = 50, k: int = 10) -> Overlap:
    return Overlap("F address numbers | region", _address_number_postings, df_cap, k)


def channel_G(df_cap: int = 500, k: int = 10) -> Overlap:
    return Overlap("G address tokens | region", _address_token_postings, df_cap, k)


# =========================================================================== character 3-gram TF-IDF (E)

def _trigrams(df: pl.DataFrame, idcol: str, text: pl.Expr) -> pl.DataFrame:
    s = pl.concat_str([pl.lit(" "), text, pl.lit(" ")])
    return (df.select(id=pl.col(idcol), s=s).filter(pl.col("s").str.len_chars() >= 3)
            .with_columns(off=pl.int_ranges(0, pl.col("s").str.len_chars() - 2)).explode("off")
            .select("id", tri=pl.col("s").str.slice(pl.col("off"), 3))
            .group_by("id", "tri").agg(tf=pl.len()))


class Trigram:
    """TF-IDF over padded character 3-grams, cosine top-k via sparse_dot_topn. Query text is name_latin when
    present (romanised non-Latin names), else fold(name_clean); index text is fold(name_clean).
    3-grams present in more than `df_cap_frac` of the S1 block are dropped (compute control)."""
    kind = "ranked"
    QUERY_CHUNK = 400_000

    def __init__(self, df_cap_frac: float = 0.02, k: int = 10, n_threads: int = 12):
        self.name = "E char 3-gram TF-IDF"
        self.df_cap_frac, self.k, self.n_threads = df_cap_frac, k, n_threads
        self.params = {"df_cap_frac": df_cap_frac, "k": k}

    def _weights(self, tri: pl.DataFrame) -> pl.DataFrame:
        w = (tri.join(self.vocab, on="tri")
             .with_columns(w=((1 + pl.col("tf").cast(pl.Float32).log()) * pl.col("idf")).cast(pl.Float32)))
        return w.with_columns(w=pl.col("w") / (pl.col("w") ** 2).sum().over("id").sqrt())

    def prepare(self, idx: pl.DataFrame) -> None:
        tri = _trigrams(idx, "s1", fold(COL.name_clean))
        n = idx.height
        df = tri.group_by("tri").agg(df=pl.len())
        cap = max(1, int(self.df_cap_frac * n))
        self.vocab = (df.filter(pl.col("df") <= cap).sort("tri").with_row_index("col")
                      .with_columns(idf=(pl.lit(math.log(1 + n)) - (1 + pl.col("df").cast(pl.Float64)).log() + 1).cast(pl.Float32))
                      .select("tri", "col", "idf"))
        self.s1_ids = idx["s1"].to_numpy()
        pos = pl.DataFrame({"id": self.s1_ids, "row": np.arange(n, dtype=np.int64)})
        w = self._weights(tri).join(pos, on="id")
        m = sp.csr_matrix((w["w"].to_numpy(), (w["row"].to_numpy(), w["col"].to_numpy())),
                          shape=(n, self.vocab.height), dtype=np.float32)
        self.BT = m.T.tocsr()
        self.params |= {"df_cap_abs": cap, "vocab": self.vocab.height, "s1_nnz": m.nnz}

    def retrieve(self, qdf: pl.DataFrame) -> pl.DataFrame:
        from sparse_dot_topn import sp_matmul_topn
        text = pl.coalesce(pl.col(COL.name_latin), fold(COL.name_clean))
        out = []
        for start in range(0, qdf.height, self.QUERY_CHUNK):
            chunk = qdf.slice(start, self.QUERY_CHUNK)
            q_ids = chunk["q"].to_numpy()
            pos = pl.DataFrame({"id": q_ids, "row": np.arange(len(q_ids), dtype=np.int64)})
            w = self._weights(_trigrams(chunk, "q", text)).join(pos, on="id")
            Q = sp.csr_matrix((w["w"].to_numpy(), (w["row"].to_numpy(), w["col"].to_numpy())),
                              shape=(len(q_ids), self.vocab.height), dtype=np.float32)
            R = sp_matmul_topn(Q, self.BT, top_n=self.k, sort=True, n_threads=self.n_threads).tocoo()
            out.append(pl.DataFrame({"q": q_ids[R.row], "s1": self.s1_ids[R.col], "score": R.data.astype(np.float32)}))
        res = pl.concat(out).filter(pl.col("score") > 0).sort(["q", "score", "s1"], descending=[False, True, False])
        return res.with_columns(rank=pl.int_range(1, pl.len() + 1).over("q")).cast(OUT_SCHEMA)


class TrigramRegion(Trigram):
    """Same TF-IDF 3-gram retrieval, but a query only searches S1 records of its own address_region
    (country-level vocabulary and IDF). Queries without a region search the whole country block.
    Motivation: compute (the index each query scans is 5-10x smaller) - see RECALL_STUDY.md."""

    def __init__(self, df_cap_frac: float = 0.02, k: int = 10, n_threads: int = 12):
        super().__init__(df_cap_frac, k, n_threads)
        self.name = "E' char 3-gram TF-IDF | region"

    def prepare(self, idx: pl.DataFrame) -> None:
        super().prepare(idx)                       # country matrix (fallback) + vocab
        m = self.BT.T.tocsr()                      # rows = S1 in idx order
        regions = idx[COL.address_region].to_numpy()
        self.blocks = {}
        for r in pl.Series(regions).drop_nulls().unique().to_list():
            rows = np.flatnonzero(regions == r)
            self.blocks[r] = (m[rows].T.tocsr(), self.s1_ids[rows])
        self.params |= {"regions": len(self.blocks)}

    def retrieve(self, qdf: pl.DataFrame) -> pl.DataFrame:
        from sparse_dot_topn import sp_matmul_topn
        text = pl.coalesce(pl.col(COL.name_latin), fold(COL.name_clean))
        out = []
        for (region,), part in qdf.group_by([COL.address_region], maintain_order=True):
            BT, ids = self.blocks.get(region, (self.BT, self.s1_ids)) if region is not None else (self.BT, self.s1_ids)
            for start in range(0, part.height, self.QUERY_CHUNK):
                chunk = part.slice(start, self.QUERY_CHUNK)
                q_ids = chunk["q"].to_numpy()
                pos = pl.DataFrame({"id": q_ids, "row": np.arange(len(q_ids), dtype=np.int64)})
                w = self._weights(_trigrams(chunk, "q", text)).join(pos, on="id")
                Q = sp.csr_matrix((w["w"].to_numpy(), (w["row"].to_numpy(), w["col"].to_numpy())),
                                  shape=(len(q_ids), self.vocab.height), dtype=np.float32)
                R = sp_matmul_topn(Q, BT, top_n=self.k, sort=True, n_threads=self.n_threads).tocoo()
                out.append(pl.DataFrame({"q": q_ids[R.row], "s1": ids[R.col], "score": R.data.astype(np.float32)}))
        if not out:
            return _empty()
        res = pl.concat(out).filter(pl.col("score") > 0).sort(["q", "score", "s1"], descending=[False, True, False])
        return res.with_columns(rank=pl.int_range(1, pl.len() + 1).over("q")).cast(OUT_SCHEMA)


def channel_E(df_cap_frac: float = 0.02, k: int = 10) -> Trigram:
    return Trigram(df_cap_frac, k)


def channel_E_region(df_cap_frac: float = 0.02, k: int = 10) -> TrigramRegion:
    return TrigramRegion(df_cap_frac, k)


def all_channels(d_cap=1000, e_cap=0.02, f_cap=50, g_cap=500, k=10) -> list:
    return [channel_A(), channel_B(), channel_C(), channel_D(d_cap, k), channel_E(e_cap, k),
            channel_F(f_cap, k), channel_G(g_cap, k)]
