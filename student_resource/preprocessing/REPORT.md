# Stage 2 preprocessing report

This report covers a full-dataset run (24.2 M records) of `python -m preprocessing.preprocess --clean`, repeated three
times from a clean state. Every number below comes from `preprocessing/results/` and from the executed notebook
`preprocessing/notebooks/preprocessing_comparison.ipynb` (tables and plots in `results/comparison/`).
No matching, blocking or candidate generation was performed.

> **Update: `PREPROCESSING_AUDIT.md` §9 applied.** Three columns were appended (`address_region`,
> `address_region_found`, `name_latin`), giving 58 columns. Indic LLP forms were added to the legal-form lexicon,
> so `name_legal_forms`, `name_base` and `name_has_legal_form` changed for S2/S3 Indic names only. The fingerprint
> diff shows every other existing column unchanged in every file. Validation: 149/149 original + 30/30 new checks.
> The rebuild took 13.0 min with a peak of 9.1 GB. Evidence is in `results/audit/`; the frozen version is in
> `FROZEN_FINGERPRINTS.json`; the column reference is `DATA_DICTIONARY.md`. The facts below describe the original
> 55-column build unless stated otherwise.

## Run facts

| | |
|---|---|
| Records processed | 24,229,173 source records in 6 files, plus 2,206,821 ground-truth rows |
| Records dropped | **0** (row count, `entity_id` set and row order identical to raw in every file) |
| Validation | **149 / 149 checks pass** |
| Raw TSVs | unchanged (SHA-256 identical before and after every run) |
| Reproducibility | two independent clean runs give identical per-column content fingerprints for all 7 outputs |
| Runtime | 9.3 min end to end (process 3.5 min, validate 1 min, measure ≈ 4.7 min); comparison notebook 4.6 min |
| Peak memory | processing ≈ 4 GB per file; whole run 8.9 GB (the ground-truth diagnostics) of 15.6 GB |
| Output | `processed/*.parquet`, 4.2 GB, 55 columns per source record |

## A. What preprocessing was performed?

The same deterministic stage chain is applied to names and addresses (`transforms.py`):

1. **unicode**: repair UTF-8-as-Latin-1 mojibake of dashes and quotes; NFKC; drop zero-width and format characters;
   control characters become spaces; unify curly quotes and Unicode dashes.
2. **lowercase**.
3. **whitespace**: collapse runs and strip. Stages 1–3 produce the stored **`*_norm`** columns.
4. **placeholder segments** (addresses only): drop whole comma segments equal to `NULL` / `null` / `<NULL>` / `N/A`.
5. **punctuation**: join apostrophes (`o'reilly`→`oreilly`) and dotted acronyms (`l.l.c.`→`llc`), keep `&` as a token,
   turn other punctuation into spaces. This produces **`*_clean`**, `*_tokens` and `address_segments`.
6. **Derived structure**: `name_base` (legal forms removed), `name_legal_forms` (canonical classes, including Indic
   scripts), `name_domain`, `name_numbers`, `address_numbers` (leading zeros stripped), `postal_code_candidate`
   (conservative, country-scoped), script labels, 30 descriptive flags, token and segment counts.
7. **Country**: `country_clean` (trim plus canonical label). The raw values `US` / `India` / `France` were already clean.

## B. What information was preserved?

* **Every raw field, byte for byte** (`business_name_raw`, `business_address_raw`, `country_raw`), with row order
  unchanged.
* **Punctuation, case information and placeholders**: punctuation stays in `*_norm`, case in the raw fields and the
  `*_is_upper` flags, placeholders in `address_norm` and `address_has_placeholder`.
* **All digits**: a validation check confirms that no digit of `name_norm` is lost in `name_clean`.
* **Legal forms, accents, non-Latin scripts and domains**: legal forms stay in `name_clean` and `name_legal_forms`,
  accents in every stored column, and non-Latin script text is never transliterated. Domains stay in the name and
  are also extracted to `name_domain`.
* **Missing values**: no representation is null unless the raw value is null. No address is emptied by placeholder
  removal. The only new nulls are 4–569 `name_base` values per file, for names that consist only of a legal form
  (`SA`, `PC`).

## C. What information was normalised?

* **Case.**
* **Unicode representation**: NFKC, invisible characters, mojibake, quote and dash variants.
* **Whitespace.**
* **Punctuation**, in `*_clean` only.
* **Placeholder address segments**, in `address_clean`.
* **Leading zeros in `address_numbers`.**
* **Legal-form tokens**, only in `name_base`.
* **Accent folding** is available as `transforms.fold_diacritics` but is **not stored**.

## D. How much did each transformation change the dataset?

Share of records changed by each step (`results/comparison/03_pct_changed_by_source.csv`):

| step | names S1 | names S2/S3 | addresses S1 | addresses S2/S3 |
|---|---|---|---|---|
| unicode | 0 % | 0.3–0.6 % | 0.04–0.16 % | 0.03–0.4 % |
| lowercase | 100 % | 84–89 % | 100 % | ~97 % |
| whitespace | 0 % | 10–11 % | 0.05 % | 0.07–2.1 % |
| placeholder segments | – | – | ≈0 % | 2.8–6.1 % |
| punctuation | 14–17 % | 28–33 % | 100 % (commas) | ~97 % |
| legal-form removal (`name_base`) | 68–71 % | 61–67 % | – | – |
| accent fold (not stored) | 0 % train, 2.4 % test | 5.8–8.2 % | 0 % train, 4.2 % test | ≈0 % train, 2.6–2.7 % test |

Share of distinct raw values merged by each representation (`results/uniqueness_by_source.csv`):

| | names `clean` | names `base` | addresses `clean` |
|---|---|---|---|
| S1 | 0.9–1.4 % | 13.8–15.0 % | 0.02–0.2 % |
| S2/S3 | 7.1–8.7 % | 23.1–25.1 % | 0.4–1.8 % |

Cleaning shortens names by only 0.3–0.6 characters and addresses by about 3.5 characters. Punctuation splits add
tokens to 9–11 % of S2/S3 names and 25–42 % of addresses.

## E. Did normalisation create significant collisions?

* **Between different entities, no, except through legal-form removal.**
  * S1 is de-duplicated on raw (name, address).
  * Under `name_clean + address_clean` it stays unique in train. In test it has 1 merged pair, which is a
    near-duplicate inside S1 itself: two `Lille Club SAS` records at `72-74` vs `72 -74 RUE Royale`.
  * `name_base + address_clean` merges **4 train / 30 test S1 pairs** that differ only in legal form
    (`Lille Sante SARL` vs `SCI`).
  * On the train ground truth, S2/S3 records of different entities merged by a name+address key are: 0 raw,
    1 group (`clean`), 7 groups (`base`).
* **Within the same entity, yes, and that is the intended effect.** On a 10 % sample of true pairs (764 k), exact
  name equality rises from 2.7–6.2 % raw to 15–27 % (`clean`) and 41–50 % (`base`). No stage removes an agreement
  that already existed.
* **Name-only keys become more ambiguous.** The share of S2/S3 records in groups spanning several entities rises
  from 15.5 % (raw) to 22.3 % (`clean`) and 32.2 % (`base`). Names were already ambiguous raw: 38 % of S1 records
  share their raw name.

## F. Are S1/S2/S3 different after preprocessing?

Yes, and the source noise profiles remain visible after preprocessing:

* **S1 is clean.** It has no missing addresses, no non-ASCII text outside France, and no whitespace or casing
  anomalies.
* **S2 upper-cases its values**: 17–19 % of names and 52–66 % of addresses. It carries 9–11 % non-Latin names.
* **S3 keeps mixed case** (3 % upper-case names) and writes US states in full while abbreviating Indian states.
  Normalisation therefore does not raise S1–S3 whole-address equality, which stays at 4.1–4.4 % raw and clean,
  whereas S1–S2 rises from 0 % to 11.5–13.9 %.
* **Both S2 and S3** have about 3 % missing addresses, about 3.5 % placeholder segments, about 10 % repeated
  whitespace, about 3.5 % domain-only names and about 4 % legal forms placed first.

## G. What types of noisy representations still remain?

These are left untouched on purpose (`results/remaining_noise.csv`, notebook §8):

* Digit-for-letter and letter typos (`At1antic`, `Hea1th`, `Venutsre`): 2–3 % of S2/S3 names contain mixed
  letter-digit tokens.
* Untransliterated Indic names: 5–11 % of S2/S3 names.
* Domain concatenations (`physicaltherapyvalley com`): about 3 %.
* Legal form first (`llc liberty cgpoley`) and honorifics (`m s`, `shri`): about 4 % each.
* Appended phone numbers.
* `&` vs `and`.
* Abbreviated vs full street types: 22–27 % of S2/S3 addresses abbreviate vs 3 % of S1.
* State name vs state code.
* Component reordering.
* Zero-padded house numbers: about 5.5 % of S2/S3 addresses.
* Names with no letters at all: 111–423 per S2/S3 file.

## H. What useful representations are now available?

* **Text, in two strengths:** `*_norm` (punctuation kept) and `*_clean` / `*_tokens` (punctuation-free), plus
  `address_segments` in the original order.
* **Name-core view:** `name_base` together with `name_legal_forms`.
* **Numeric structure:** `address_numbers` (the digit set agrees for 59–70 % of true pairs, far more than whole
  addresses) and `name_numbers`.
* **Structural extras:** `name_domain`, `name_script` / `address_script`, `country_clean`.
* **Descriptive flags and counts** describing the raw record: upper-case, non-Latin, domain, placeholders,
  landmarks, care-of, encoding artefacts, and others.
* **On-demand functions:** `transforms.fold_diacritics` and every intermediate stage in `transforms.py`.

## I. What preprocessing decisions should NOT be made yet?

* **Whether to drop legal forms from matching.** They give the largest agreement gain (+23 % of true pairs), but they
  are the only step that merges distinct S1 entities.
* **Whether to fold accents.** It is helpful on train, where accents are injected noise. On France (test only,
  without labels) accents are genuine spelling in 16–28 % of values.
* **Abbreviation expansion** (`st`/`rd`/`r`), **state name ↔ code mapping**, **transliteration of Indic scripts**,
  **splitting domain concatenations**, **typo or digit repair**, and **stripping honorifics or phone numbers**. Each
  depends on the similarity method chosen next.
* **Treating `address_numbers` or `postal_code_candidate` as keys.** Postal candidates cover < 0.1 % of records.
* **Any rule derived from the ground truth.** It was used here only for validation diagnostics.
