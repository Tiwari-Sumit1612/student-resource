"""Data dictionary for the processed layer: the single source of truth for column names.

Later stages must import names from here instead of hard-coding strings:

    from preprocessing.schema import COL, columns
    lf.select(COL.entity_id, COL.name_clean, COL.address_region)     # typo -> AttributeError
    columns(role="canonical", stored=True)                           # list of names by role

    python -m preprocessing.schema           # check against every processed Parquet + write DATA_DICTIONARY.md

Fields per column
    role       identity | raw | canonical | auxiliary | descriptive
    stored     True = column in processed/*.parquet; False = computed on demand (`expr` says how)
    exact_key  identity   - unique record id
               hard block - exact partition; never separates a true pair
               channel    - exact equality is a usable candidate channel, but collisions/decoys are expected
               no         - never use for exact matching / exact blocking (fuzzy, soft, descriptive or sparse)
Evidence numbers come from preprocessing/results (train ground truth used for evaluation only).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    group: str
    role: str
    stored: bool
    exact_key: str
    meaning: str
    expr: str = ""          # for on-demand columns: the transforms.py expression that computes it


_F = "descriptive"
_B = "Boolean"

COLUMNS: tuple[Column, ...] = (
    # ------------------------------------------------------------------ identity / raw
    Column("entity_id", "String", "identity", "identity", True, "identity",
           "Record id, unique per file, prefix S1-/S2-/S3-. Row order and the numeric part carry no signal "
           "(row-order Spearman vs matched S1 = 0.001); never use either as a feature."),
    Column("split", "String", "identity", "identity", True, "hard block", "'train' or 'test'."),
    Column("source", "String", "identity", "identity", True, "hard block", "'S1', 'S2' or 'S3'."),
    Column("business_name_raw", "String", "raw", "raw", True, "no", "Original business_name, byte-identical to the TSV."),
    Column("business_address_raw", "String", "raw", "raw", True, "no", "Original business_address, byte-identical; null = missing."),
    Column("country_raw", "String", "raw", "raw", True, "no", "Original country value, byte-identical."),
    # ------------------------------------------------------------------ country / missingness
    Column("country_clean", "String", "country", "canonical", True, "hard block",
           "Trimmed canonical country label (US, India, France). True pairs never cross countries."),
    Column("name_missing", _B, "missingness", _F, True, "no", "Raw name null or blank."),
    Column("address_missing", _B, "missingness", _F, True, "no", "Raw address null or blank (S2/S3 only, ~3%)."),
    Column("country_missing", _B, "missingness", _F, True, "no", "Raw country null or blank (never true in this data)."),
    # ------------------------------------------------------------------ name representations
    Column("name_norm", "String", "name", "canonical", True, "channel",
           "NFKC + mojibake/zero-width repair + lowercase + whitespace collapse; punctuation kept."),
    Column("name_clean", "String", "name", "canonical", True, "channel",
           "name_norm with punctuation normalised (o'reilly->oreilly, l.l.c.->llc, & kept). Exactly equal on "
           "15-17% (India) / 27% (US) of true pairs; 39% of S1 records share their name_clean."),
    Column("name_tokens", "List(String)", "name", "canonical", True, "no",
           "name_clean split on spaces (index terms, not a key). Rejoins exactly to name_clean."),
    Column("name_base", "String", "name", "auxiliary", True, "channel",
           "name_clean without legal-form tokens (incl. Indic scripts). Equal on 41-50% of true pairs, but merges "
           "4 train / 30 test S1 pairs that differ only in legal form: use with name_legal_forms, never alone. "
           "Null when the name is only a legal form."),
    Column("name_legal_forms", "List(String)", "name", "auxiliary", True, "no",
           "Canonical legal-form classes in order (llc, inc, ltd, pvt, llp, sarl, ...), all scripts."),
    Column("name_domain", "String", "name", "auxiliary", True, "channel",
           "Domain found in the name (richardsonmexico.com); null otherwise. ~3-4% of S2/S3 names."),
    Column("name_numbers", "List(String)", "name", "auxiliary", True, "no",
           "Digit runs in the name, as written (includes appended phone numbers)."),
    Column("name_script", "String", "name", _F, True, "no",
           "'Latin', a script name ('Devanagari', 'Tamil', ...), 'Latin+<script>' or 'None' (no letters)."),
    # ------------------------------------------------------------------ address representations
    Column("address_norm", "String", "address", "canonical", True, "channel",
           "Address with the same norm chain as name_norm; punctuation and placeholder segments kept."),
    Column("address_clean", "String", "address", "canonical", True, "channel",
           "Placeholder segments (NULL/null/<NULL>/N/A) removed, punctuation normalised. Exactly equal on only "
           "4-14% of true pairs: address variation is token-level."),
    Column("address_tokens", "List(String)", "address", "canonical", True, "no", "address_clean split on spaces."),
    Column("address_segments", "List(String)", "address", "canonical", True, "no",
           "Comma segments of the address, each cleaned, original order, placeholders dropped."),
    Column("address_numbers", "List(String)", "address", "auxiliary", True, "channel",
           "Digit runs, leading zeros stripped (001->1). Same digit SET on 59-70% of true pairs. As-written digits: "
           "see numbers_as_written."),
    Column("postal_code_candidate", "String", "address", "auxiliary", True, "no",
           "Conservative country-scoped postcode; present for < 0.1% of records. Not a key."),
    Column("address_script", "String", "address", _F, True, "no", "Script label of the raw address (as name_script)."),
    # ------------------------------------------------------------------ counts
    Column("name_n_tokens", "UInt16", "counts", _F, True, "no", "len(name_tokens)."),
    Column("address_n_tokens", "UInt16", "counts", _F, True, "no", "len(address_tokens)."),
    Column("address_n_segments", "UInt16", "counts", _F, True, "no", "len(address_segments)."),
    # ------------------------------------------------------------------ name flags (describe the RAW value)
    Column("name_is_upper", _B, "name flags", _F, True, "no", "Raw name is all upper-case (S2: ~18%)."),
    Column("name_is_lower", _B, "name flags", _F, True, "no", "Raw name is all lower-case."),
    Column("name_has_digits", _B, "name flags", _F, True, "no", "Raw name contains a digit."),
    Column("name_has_non_latin", _B, "name flags", _F, True, "no", "Raw name contains a non-Latin letter."),
    Column("name_has_latin_diacritics", _B, "name flags", _F, True, "no", "Raw name has accented Latin letters (real in France, injected noise in US/India)."),
    Column("name_has_domain", _B, "name flags", _F, True, "no", "name_domain is not null."),
    Column("name_is_domain_only", _B, "name flags", _F, True, "no", "Whole name is a domain (words concatenated)."),
    Column("name_has_special_chars", _B, "name flags", _F, True, "no", "Raw name contains # @ * _ | ~ ^ < > { } [ ] \\ \"."),
    Column("name_has_brackets", _B, "name flags", _F, True, "no", "Raw name contains brackets."),
    Column("name_has_leading_junk", _B, "name flags", _F, True, "no", "Raw name starts with a non-alphanumeric character (***, >>, --)."),
    Column("name_has_repeated_whitespace", _B, "name flags", _F, True, "no", "Raw name contains 2+ consecutive spaces."),
    Column("name_has_encoding_artifact", _B, "name flags", _F, True, "no", "Raw name contains control / C1 / replacement characters."),
    Column("name_has_legal_form", _B, "name flags", _F, True, "no", "name_legal_forms is non-empty."),
    Column("name_legal_at_start", _B, "name flags", _F, True, "no", "First name token is a legal form (reordered name)."),
    Column("name_legal_only", _B, "name flags", _F, True, "no", "Name consists only of legal forms (name_base is null)."),
    Column("name_clean_empty", _B, "name flags", _F, True, "no", "Raw name present but name_clean empty (never true in this data)."),
    # ------------------------------------------------------------------ address flags
    Column("address_is_upper", _B, "address flags", _F, True, "no", "Raw address is all upper-case (S2: 52-66%)."),
    Column("address_has_digits", _B, "address flags", _F, True, "no", "Raw address contains a digit."),
    Column("address_has_non_latin", _B, "address flags", _F, True, "no", "Raw address contains a non-Latin letter (Indic state names)."),
    Column("address_has_latin_diacritics", _B, "address flags", _F, True, "no", "Raw address has accented Latin letters."),
    Column("address_has_special_chars", _B, "address flags", _F, True, "no", "Raw address contains special characters (#, ##, <, ...)."),
    Column("address_has_encoding_artifact", _B, "address flags", _F, True, "no", "Raw address contains control / C1 / replacement characters (mojibake)."),
    Column("address_has_placeholder", _B, "address flags", _F, True, "no", "Some comma segment is NULL/null/<NULL>/N/A."),
    Column("address_placeholder_only", _B, "address flags", _F, True, "no", "Address consisted only of placeholders (never true in this data)."),
    Column("address_has_zero_padded_number", _B, "address flags", _F, True, "no", "Address contains a zero-padded number (0123)."),
    Column("address_has_landmark", _B, "address flags", _F, True, "no", "Landmark words (near, opp, behind, ...)."),
    Column("address_has_care_of", _B, "address flags", _F, True, "no", "Care-of markers (c/o, s/o, w/o, d/o, h/o)."),
    # ------------------------------------------------------------------ audit additions (section 9)
    Column("address_region", "String", "address", "auxiliary", True, "no",
           "ISO-3166-2-style region (US-OH, IN-MH, FR-HDF) named by a WHOLE address segment, per-country gazetteer "
           "(codes, full names, native-script names, French departments). Null when absent or ambiguous. Agrees on "
           "99.5% of true pairs where both sides are found; disagrees on 85% (India) / 95% (US) of same-name decoys. "
           "Soft feature / optional sub-block only, never a hard filter (TG vs AP after the 2014 state split)."),
    Column("name_latin", "String", "name", "auxiliary", True, "no",
           "anyascii 0.3.3 romanisation of name_clean for names containing non-Latin script, lowercase ASCII "
           "[a-z0-9 &]; null for Latin / letterless names. Fuzzy by construction (lksmi yunivrsl ...): for character "
           "similarity only. Median fuzz ratio to the S1 name 9.8 -> 77.6 on non-Latin true pairs."),
    Column("address_region_found", _B, "address flags", _F, True, "no", "address_region is not null."),
    # ------------------------------------------------------------------ on demand (not stored)
    Column("name_clean_folded", "String", "name", "auxiliary", False, "channel",
           "Accents removed from Latin letters (Indic vowel signs kept). Real spelling in France: keep both.",
           'T.fold_diacritics("name_clean")'),
    Column("name_base_folded", "String", "name", "auxiliary", False, "channel",
           "Accent-folded name_base. Equal on 43-56% of true pairs; same S1 caveat as name_base.",
           'T.fold_diacritics("name_base")'),
    Column("address_clean_folded", "String", "address", "auxiliary", False, "channel",
           "Accent-folded address_clean.", 'T.fold_diacritics("address_clean")'),
    Column("numbers_as_written", "List(String)", "address", "auxiliary", False, "no",
           "Address digit runs with leading zeros kept.", 'T.numbers("address_norm")'),
    Column("address_region_matches", "List(String)", "address", "auxiliary", False, "no",
           "Every region named by a whole segment (before ambiguity resolution).",
           'T.address_region_matches("address_segments", "country_clean")'),
)

BY_NAME = {c.name: c for c in COLUMNS}
COL = SimpleNamespace(**{c.name: c.name for c in COLUMNS})
STORED = [c.name for c in COLUMNS if c.stored]
ON_DEMAND = [c.name for c in COLUMNS if not c.stored]


def columns(*, role: str | None = None, group: str | None = None, stored: bool | None = None,
            exact_key: str | None = None) -> list[str]:
    """Column names filtered by any combination of fields."""
    return [c.name for c in COLUMNS
            if (role is None or c.role == role) and (group is None or c.group == group)
            and (stored is None or c.stored == stored) and (exact_key is None or c.exact_key == exact_key)]


def check() -> list[str]:
    """Compare the dictionary with every processed source file. Returns a list of problems (empty = OK)."""
    import polars as pl
    from . import config as C
    problems = []
    for sp in C.SPLITS:
        for s in C.SOURCES:
            schema = pl.scan_parquet(C.processed_path(sp, s)).collect_schema()
            if list(schema) != STORED:
                missing = [c for c in STORED if c not in schema]
                extra = [c for c in schema if c not in BY_NAME]
                problems.append(f"{sp} {s}: column set/order differs (undocumented: {extra}, missing: {missing})")
            for name, dtype in schema.items():
                if name in BY_NAME and str(dtype) != BY_NAME[name].dtype:
                    problems.append(f"{sp} {s}: {name} is {dtype}, dictionary says {BY_NAME[name].dtype}")
    return problems


def to_markdown() -> str:
    head = ["# Data dictionary: processed layer", "",
            "Generated from `preprocessing/schema.py` (edit that file, not this one). Import names from it:",
            "`from preprocessing.schema import COL` then `COL.name_clean`.", "",
            "**exact_key**: `identity` unique record id · `hard block` exact partition, never separates a true pair · "
            "`channel` exact equality is a usable candidate channel, collisions expected · `no` never an exact key.", ""]
    out = head
    for stored, title in ((True, "Stored columns (`processed/*.parquet`, in file order)"),
                          (False, "On-demand representations (not stored)")):
        out += [f"## {title}", "", "| # | column | type | role | exact_key | meaning |" + (" computed by |" if not stored else ""),
                "|---|---|---|---|---|---|" + ("---|" if not stored else "")]
        for i, c in enumerate([c for c in COLUMNS if c.stored == stored], 1):
            row = f"| {i} | `{c.name}` | {c.dtype} | {c.role} | {c.exact_key} | {c.meaning.replace('|', '/')} |"
            out.append(row + (f" `{c.expr}` |" if not stored else ""))
        out.append("")
    return "\n".join(out)


def main() -> int:
    from . import config as C
    problems = check()
    for p in problems:
        print("PROBLEM:", p)
    path = C.ROOT / "preprocessing" / "DATA_DICTIONARY.md"
    path.write_text(to_markdown(), encoding="utf-8")
    print(f"{len(STORED)} stored + {len(ON_DEMAND)} on-demand columns documented -> {path}")
    print("dictionary matches every processed file" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
