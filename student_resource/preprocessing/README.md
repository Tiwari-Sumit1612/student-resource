# Stage 2 — Preprocessing layer

Converts the raw S1/S2/S3 TSVs into clean, **information-preserving** Parquet representations.
It performs **no matching**: no blocking, candidate generation, similarity scoring, embeddings or thresholds.

## Run

From `student_resource/`:

```bash
.venv\Scripts\python -m preprocessing.preprocess --clean        # full run from a clean state (~9.5 min, peak ~9 GB RAM)
.venv\Scripts\python -m preprocessing.preprocess --skip-measure # only process + validate (~5 min)

# RAW vs PROCESSED comparison notebook (~5 min, reads processed/ and results/)
.venv\Scripts\jupyter-nbconvert --to notebook --execute --inplace preprocessing\notebooks\preprocessing_comparison.ipynb
```

Requirements: Python 3.12, `polars==1.44.2`, `psutil` (see `requirements.txt`).
Peak memory, per-step timings and memory peaks, raw-file SHA-256 (before/after) and per-column output
fingerprints are written to `results/run_manifest.json`. The step log is in `results/run.log`.
`--clean` deletes `processed/` and all of `results/` (including `results/comparison/`), so re-run the notebook afterwards.
Findings of the executed run: **`REPORT.md`**.

## Layout

| Path | Content |
|---|---|
| `preprocessing/config.py` | every rule/pattern/list in one place (legal forms, placeholders, TLDs, postal rules, scripts) |
| `preprocessing/transforms.py` | pure, deterministic Polars expressions for each stage (reusable lazily later) |
| `preprocessing/preprocess.py` | orchestration: stream raw TSV → Parquet (temp file + atomic rename), GT conversion, fingerprints |
| `preprocessing/validate.py` | sanity checks → `results/validation_checks.csv` |
| `preprocessing/measure.py` | effect/collision/source measurements → `results/*.csv` |
| `preprocessing/REPORT.md` | findings of the executed run |
| `preprocessing/notebooks/preprocessing_comparison.ipynb` | executed RAW vs PROCESSED evaluation (reuses `transforms`/`validate`; outputs in `results/comparison/`) |
| `preprocessing/schema.py` | **data dictionary** (single source of column names; `from preprocessing.schema import COL`); `python -m preprocessing.schema` checks it against the Parquet files and writes `DATA_DICTIONARY.md` |
| `preprocessing/audit_eval.py` | evaluation for `PREPROCESSING_AUDIT.md` §9 (row-order leakage, region / romanisation / legal-form checks); ground truth used for evaluation only; outputs in `results/audit/` |
| `preprocessing/FROZEN_FINGERPRINTS.json` | frozen per-column content fingerprints of the current version (survives `--clean`) |
| `processed/{split}_source{n}.parquet` | one file per raw source file, same rows, same `entity_id`s |
| `processed/train_ground_truth.parquet` | GT with the id list parsed (format conversion only) |

Read everything lazily with `pl.scan_parquet("processed/*_source*.parquet")`.

## Stage chain

```
raw ─unicode─▶ ─lowercase─▶ ─whitespace─▶ *_norm ─(address: drop placeholder segments)─▶ ─punctuation─▶ *_clean ─split─▶ *_tokens
                                                                                                         ├─ name_base (legal forms removed)
                                                                                                         └─ fold_diacritics() (on demand, not stored)
```

| Stage | What it does | What it deliberately does NOT do |
|---|---|---|
| unicode | repair `â€“`-style mojibake of dashes/quotes, NFKC (full-width, ligatures, NBSP), drop zero-width/format chars (U+200C etc.), control chars → space, unify curly quotes and dashes | no transliteration, no accent removal |
| lowercase | Unicode lower-case | — |
| whitespace | collapse runs of whitespace, strip ends; empty → null | — |
| placeholder (address) | drop whole comma segments equal to `NULL`/`null`/`<NULL>`/`N/A` | `NA`, `Na`, `Bldg N/A`, `Null Bazar` are kept |
| punctuation | `o'reilly`→`oreilly`, `l.l.c.`→`llc`, `&` kept as its own token, other punctuation → space; digits and letters never removed | no stop-word, legal-form, abbreviation or number rewriting |

## Schema (one row per raw record)

The authoritative, machine-checked list of all 58 stored columns is **`DATA_DICTIONARY.md`** (generated from
`schema.py`). Added by the audit (appended after the original 55; no original column moved):
`address_region` (ISO-3166-2-style region from whole address segments, per-country gazetteer),
`address_region_found`, and `name_latin` (anyascii 0.3.3 romanisation of non-Latin names, null otherwise).

**Identity / raw (unchanged):** `entity_id`, `split`, `source`, `business_name_raw`, `business_address_raw`, `country_raw`

**Country:** `country_clean` (trimmed + canonical label; equals raw for all observed values), `country_missing`

**Name representations**

| Column | Type | Meaning |
|---|---|---|
| `name_norm` | str | unicode + lowercase + whitespace; punctuation preserved (`o'reilly & sons, l.l.c.`) |
| `name_clean` | str | punctuation-normalised (`oreilly & sons llc`) |
| `name_tokens` | list[str] | `name_clean` split on spaces |
| `name_base` | str | `name_clean` without legal-form tokens (`oreilly & sons`); null if the name is only legal forms |
| `name_legal_forms` | list[str] | canonical legal classes found (`llc`, `ltd`, `pvt`, `sarl`, …, incl. Indic-script forms) |
| `name_domain` | str | domain found in the name (`richardsonmexico.com`) |
| `name_numbers` | list[str] | digit runs as written |
| `name_script` | str | `Latin`, `Devanagari`, `Latin+Tamil`, …, `None` |
| `name_n_tokens` | u16 | token count |

**Address representations**

| Column | Type | Meaning |
|---|---|---|
| `address_norm` | str | unicode + lowercase + whitespace; punctuation and placeholders preserved |
| `address_clean` | str | placeholder segments removed, punctuation-normalised |
| `address_tokens` | list[str] | `address_clean` split on spaces |
| `address_segments` | list[str] | comma-separated components, each cleaned (order preserved, no parsing) |
| `address_numbers` | list[str] | digit runs with leading zeros stripped (`001`→`1`); as-written digits remain in `address_norm` |
| `postal_code_candidate` | str | conservative, country-scoped rule (see config); null when not reliably present |
| `address_script` | str | as for names |
| `address_n_tokens`, `address_n_segments` | u16 | counts |

**Descriptive flags (bool):** `name_missing`, `address_missing`, `name_is_upper`, `name_is_lower`, `name_has_digits`,
`name_has_non_latin`, `name_has_latin_diacritics`, `name_has_domain`, `name_is_domain_only`, `name_has_special_chars`,
`name_has_brackets`, `name_has_leading_junk`, `name_has_repeated_whitespace`, `name_has_encoding_artifact`,
`name_has_legal_form`, `name_legal_at_start`, `name_legal_only`, `name_clean_empty`, `address_is_upper`,
`address_has_digits`, `address_has_non_latin`, `address_has_latin_diacritics`, `address_has_special_chars`,
`address_has_encoding_artifact`, `address_has_placeholder`, `address_placeholder_only`,
`address_has_zero_padded_number`, `address_has_landmark`, `address_has_care_of`.
Flags computed on raw text describe the *original* record, so they stay meaningful after cleaning.

**On-demand (not stored, to avoid duplicating GBs):** `transforms.fold_diacritics(col)` (accent folding on Latin
letters only, Indic vowel signs preserved) and the intermediate `unicode_stage` / `lowercase_stage` outputs.

## Evidence for rules (raw data only; ground truth not used)

* Placeholder segments: S2/S3 contain ≈44k each of `NULL`, `null`, `<NULL>`, `N/A` as whole comma segments per file.
* Mojibake: ≈14k addresses contain `â`/`Â` + U+0080 + C1 byte (UTF-8 `–`/`’` decoded as Latin-1).
* Zero-width non-joiners: ≈45k Indic names.
* Indic legal forms: the most frequent non-Latin tokens are transliterations of *limited/private/LLP*.
* Postal codes: US 5-digit numbers are house numbers (1.06M of 1.24M are the first token, no ZIP+4); India has no
  bare 6-digit PIN; France has ~1.2k standalone 5-digit segments — hence the conservative candidate rule.
* Country: exactly three values (`US`, `India`, `France`), no whitespace or case variants.

## Outputs in `results/`

`preprocessing_summary.csv`, `validation_checks.csv`, `normalization_effects.csv`, `collision_examples.csv`,
`uniqueness_by_source.csv`, `source_comparison.csv`, `source_comparison_by_country.csv`, `s1_key_collisions.csv`,
`s1_key_collision_examples.csv`,
`gt_s2s3_collision_purity.csv`*, `gt_representation_agreement.csv`*, `remaining_noise.csv`, `remaining_noise_examples.csv`,
`representation_examples.csv`, `missing_values.csv`, `country_values.csv`, `legal_forms.csv`, `name_domain_tlds.csv`,
`postal_code_candidates.csv`, `name_script_distribution.csv`, `address_script_distribution.csv`,
`token_count_distribution.csv`, `output_files.csv`, `run_manifest.json`, `run.log`.

\* validation-only diagnostics that use the ground truth to check for information loss; they define no rule.

`results/comparison/` (written by the notebook): 21 CSVs and 7 PNGs, numbered by notebook section
(`01_` integrity … `10_` information-loss audit).
