"""Pure, deterministic Polars expressions for every preprocessing step.

Each function takes a column name / expression and returns an expression, so the same code is
used by the pipeline (materialised columns), by the measurement step (intermediate stages that
are *not* stored) and by any later stage that wants to recompute a representation lazily.

Stage chain for a text field (each stage consumes the previous one):

    raw ──unicode──> ──lowercase──> ──whitespace──> NORM ──[address: placeholder segments]──> ──punctuation──> CLEAN
                                                                                                         └─ tokens / fold / base
"""
from __future__ import annotations

import unicodedata

import polars as pl

from . import config as C

# --------------------------------------------------------------------------- helpers

def _e(x) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def _null_if_empty(e: pl.Expr) -> pl.Expr:
    return pl.when(e.str.len_chars() > 0).then(e)


def _canon_key(s: str) -> str:
    """Apply the Python equivalent of the unicode stage to config keys so they match cleaned tokens."""
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Cf").lower()


LEGAL_MAP = {_canon_key(k): v for k, v in C.LEGAL_FORMS.items()}
LEGAL_KEYS = list(LEGAL_MAP)

# --------------------------------------------------------------------------- stages (text -> text)

def unicode_stage(x) -> pl.Expr:
    """Repair Latin-1 mojibake of UTF-8 punctuation, NFKC-normalise, drop invisible format chars
    (zero-width joiners, BOM, soft hyphen), turn control chars into spaces, unify quote/dash variants.
    Case, spacing and punctuation are otherwise untouched."""
    e = _e(x)
    for pat, rep in C.MOJIBAKE_REPAIRS:
        e = e.str.replace_all(pat, rep)
    return (e.str.normalize("NFKC")
             .str.replace_all(r"\p{Cf}", "")
             .str.replace_all(r"\p{Cc}", " ")
             .str.replace_all(C.QUOTE_SINGLE, "'")
             .str.replace_all(C.QUOTE_DOUBLE, '"')
             .str.replace_all(C.DASHES, "-"))


def lowercase_stage(x) -> pl.Expr:
    return _e(x).str.to_lowercase()


def whitespace_stage(x) -> pl.Expr:
    return _null_if_empty(_e(x).str.replace_all(r"\s+", " ").str.strip_chars())


def norm(x) -> pl.Expr:
    """NORM representation: unicode + lowercase + whitespace. Punctuation is fully preserved
    (so "o'reilly", "a.b.c.", "abc-123" remain recoverable)."""
    return whitespace_stage(lowercase_stage(unicode_stage(x)))


def punctuation_stage(x) -> pl.Expr:
    """Punctuation normalisation applied to a NORM string:
       * apostrophes inside words are dropped          o'reilly  -> oreilly
       * dotted single-letter acronyms are joined       l.l.c.    -> llc,  a.b.c. -> abc
       * '&' is kept as its own token                   b&b       -> b & b
       * every other punctuation char becomes a space  abc-123   -> abc 123 (digits kept)
    """
    e = _e(x)
    for _ in range(2):                                         # twice: rock'n'roll
        e = e.str.replace_all(r"(\p{L})'(\p{L})", "${1}${2}")
    for k in (6, 5, 4, 3, 2):                                  # longest acronyms first
        pat = r"\b" + r"\.".join([r"(\p{L})"] * k) + r"\b\.?"
        e = e.str.replace_all(pat, "".join(f"${{{i}}}" for i in range(1, k + 1)))
    e = (e.str.replace_all("&", " & ", literal=True)
          .str.replace_all(r"[^\p{L}\p{M}\p{N}\s&]", " "))
    return whitespace_stage(e)


def fold_diacritics(x) -> pl.Expr:
    """Remove combining marks that follow *Latin* letters only (é -> e, ç -> c). Indic vowel signs
    (also combining marks) are preserved. Not materialised: available on demand."""
    return (_e(x).str.normalize("NFD")
            .str.replace_all(r"(\p{Latin})\p{M}+", "${1}")
            .str.normalize("NFC"))


# --------------------------------------------------------------------------- addresses

def address_segments_norm(norm_col) -> pl.Expr:
    """Comma-separated segments of a NORM address, stripped, with empty and placeholder segments removed."""
    seg = pl.element().str.strip_chars()
    return (_e(norm_col).str.split(",")
            .list.eval(seg)
            .list.eval(pl.element().filter((pl.element().str.len_chars() > 0)
                                           & ~pl.element().str.contains(C.ADDRESS_PLACEHOLDER_SEGMENT))))


def address_placeholder_stage(norm_col) -> pl.Expr:
    """NORM address with placeholder segments removed (still punctuation-preserving)."""
    return _null_if_empty(address_segments_norm(norm_col).list.join(", "))


def address_segments_clean(norm_col) -> pl.Expr:
    return (address_segments_norm(norm_col)
            .list.eval(punctuation_stage(pl.element()))
            .list.eval(pl.element().drop_nulls()))


# --------------------------------------------------------------------------- derived structure

def tokens(clean_col) -> pl.Expr:
    return _e(clean_col).str.split(" ")


def is_legal_token(tok) -> pl.Expr:
    """Legal-form membership is tested on the diacritic-folded token, so injected accents
    ("Có", "Lìmited") are still recognised. The token itself is never altered."""
    return fold_diacritics(tok).is_in(LEGAL_KEYS)


def legal_forms(tokens_col) -> pl.Expr:
    """Canonical legal-form classes found in the token list, in order of appearance."""
    t = pl.element()
    return _e(tokens_col).list.eval(
        fold_diacritics(t).filter(is_legal_token(t)).replace_strict(LEGAL_MAP, return_dtype=pl.String))


def name_base(tokens_col) -> pl.Expr:
    """Clean name with legal-form tokens removed (a separate representation, never a replacement).
    A dangling '&' left at either end ("jp morgan & co" -> "jp morgan") is trimmed."""
    t = pl.element()
    return _null_if_empty(_e(tokens_col).list.eval(t.filter(~is_legal_token(t))).list.join(" ").str.strip_chars(" &"))


_TLD = "|".join(sorted((t.replace(".", r"\.") for t in C.DOMAIN_TLDS), key=len, reverse=True))
DOMAIN_RE = rf"(?:https?://)?(?:www\.)?((?:[\p{{L}}\p{{N}}][\p{{L}}\p{{N}}-]*\.)+(?:{_TLD}))\b"
DOMAIN_ONLY_RE = rf"^(?:https?://)?(?:www\.)?(?:[\p{{L}}\p{{N}}][\p{{L}}\p{{N}}-]*\.)+(?:{_TLD})/?$"


def domain(norm_col) -> pl.Expr:
    return _e(norm_col).str.extract(DOMAIN_RE, 1)


def numbers(norm_col, strip_leading_zeros: bool = False) -> pl.Expr:
    e = _e(norm_col).str.extract_all(r"\d+")
    if strip_leading_zeros:
        z = pl.element().str.strip_chars_start("0")
        e = e.list.eval(pl.when(z == "").then(pl.lit("0")).otherwise(z))
    return e


def postal_code_candidate(norm_col, country_col) -> pl.Expr:
    expr = pl.lit(None, dtype=pl.String)
    for country, pat in reversed(list(C.POSTAL_RULES.items())):
        expr = pl.when(_e(country_col) == country).then(_e(norm_col).str.extract(pat, 1)).otherwise(expr)
    return expr.str.replace_all(" ", "", literal=True)


def _canon_text(keys: list[str]) -> list[str]:
    """Run gazetteer keys through exactly the pipeline chain the address segments went through."""
    if not keys:
        return []
    df = pl.DataFrame({"k": keys}, schema={"k": pl.String}).select(fold_diacritics(punctuation_stage(norm("k"))))
    return df["k"].to_list()


def _region_maps() -> dict:
    """{country: {"code": {canonical key: region}, "name": {...}}}"""
    out = {}
    for country, kinds in C.region_gazetteer().items():
        out[country] = {}
        for kind, gaz in kinds.items():
            keys = list(gaz)
            out[country][kind] = {ck: gaz[k] for k, ck in zip(keys, _canon_text(keys)) if ck}
    return out


REGION_MAPS = _region_maps()


def _segment_regions(segments_col, mapping: dict) -> pl.Expr:
    if not mapping:
        return pl.lit([], dtype=pl.List(pl.String))
    return (_e(segments_col).list.eval(fold_diacritics(pl.element()).replace_strict(mapping, default=None, return_dtype=pl.String))
            .list.drop_nulls().list.unique().list.sort())


def address_region_matches(segments_col, country_col, kind: str | None = None) -> pl.Expr:
    """Distinct regions whose gazetteer key equals a WHOLE segment (accent-folded), per country.
    kind = "code" / "name" restricts to one key type; None = both."""
    expr = pl.lit(None, dtype=pl.List(pl.String))
    for country, maps in REGION_MAPS.items():
        mp = maps.get(kind, {}) if kind else maps["code"] | maps["name"]
        expr = pl.when(_e(country_col) == country).then(_segment_regions(segments_col, mp)).otherwise(expr)
    return expr


def address_region(segments_col, country_col) -> pl.Expr:
    """Canonical region code (ISO-3166-2 style) named by whole address segments.
    One distinct region -> that region. Several -> the country's precedence key type decides if it names
    exactly one region (config.REGION_PRECEDENCE); otherwise null (genuinely ambiguous). No match -> null."""
    allm = address_region_matches(segments_col, country_col)
    out = pl.when(allm.list.len() == 1).then(allm.list.first())
    for country, kind in C.REGION_PRECEDENCE.items():
        if kind:
            pref = _segment_regions(segments_col, REGION_MAPS[country][kind])
            out = out.when((_e(country_col) == country) & (pref.list.len() == 1)).then(pref.list.first())
    return out


def script_label(raw_col) -> pl.Expr:
    """'Latin', a non-Latin script name, 'Latin+<script>' for mixed text, or 'None' (no letters)."""
    c = _e(raw_col)
    nonlatin = pl.coalesce([pl.when(c.str.contains(rf"\p{{{s}}}")).then(pl.lit(s)) for s in C.SCRIPTS]
                           + [pl.when(c.str.contains(C.NON_LATIN_LETTER)).then(pl.lit("Other"))])
    has_latin = c.str.contains(r"\p{Latin}")
    return (pl.when(nonlatin.is_null() & has_latin).then(pl.lit("Latin"))
              .when(nonlatin.is_not_null() & has_latin).then(pl.lit("Latin+") + nonlatin)
              .when(nonlatin.is_not_null()).then(nonlatin)
              .otherwise(pl.lit("None")))


def is_upper(raw_col) -> pl.Expr:
    c = _e(raw_col)
    return c.str.contains(r"[A-Za-z]") & (c == c.str.to_uppercase())


def is_lower(raw_col) -> pl.Expr:
    c = _e(raw_col)
    return c.str.contains(r"[A-Za-z]") & (c == c.str.to_lowercase())


def encoding_artifact(raw_col) -> pl.Expr:
    return _e(raw_col).str.contains(r"[\x{00}-\x{08}\x{0b}\x{0c}\x{0e}-\x{1f}\x{7f}-\x{9f}\x{fffd}]")


def country_clean(raw_col) -> pl.Expr:
    key = _e(raw_col).str.strip_chars().str.replace_all(r"\s+", " ")
    return (_null_if_empty(key.str.to_lowercase())
            .replace_strict(C.COUNTRY_CANONICAL, default=key, return_dtype=pl.String))
