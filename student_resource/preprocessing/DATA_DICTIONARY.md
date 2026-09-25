# Data dictionary: processed layer

Generated from `preprocessing/schema.py` (edit that file, not this one). Import names from it:
`from preprocessing.schema import COL` then `COL.name_clean`.

**exact_key**: `identity` unique record id · `hard block` exact partition, never separates a true pair · `channel` exact equality is a usable candidate channel, collisions expected · `no` never an exact key.

## Stored columns (`processed/*.parquet`, in file order)

| # | column | type | role | exact_key | meaning |
|---|---|---|---|---|---|
| 1 | `entity_id` | String | identity | identity | Record id, unique per file, prefix S1-/S2-/S3-. Row order and the numeric part carry no signal (row-order Spearman vs matched S1 = 0.001); never use either as a feature. |
| 2 | `split` | String | identity | hard block | 'train' or 'test'. |
| 3 | `source` | String | identity | hard block | 'S1', 'S2' or 'S3'. |
| 4 | `business_name_raw` | String | raw | no | Original business_name, byte-identical to the TSV. |
| 5 | `business_address_raw` | String | raw | no | Original business_address, byte-identical; null = missing. |
| 6 | `country_raw` | String | raw | no | Original country value, byte-identical. |
| 7 | `country_clean` | String | canonical | hard block | Trimmed canonical country label (US, India, France). True pairs never cross countries. |
| 8 | `name_missing` | Boolean | descriptive | no | Raw name null or blank. |
| 9 | `address_missing` | Boolean | descriptive | no | Raw address null or blank (S2/S3 only, ~3%). |
| 10 | `country_missing` | Boolean | descriptive | no | Raw country null or blank (never true in this data). |
| 11 | `name_norm` | String | canonical | channel | NFKC + mojibake/zero-width repair + lowercase + whitespace collapse; punctuation kept. |
| 12 | `name_clean` | String | canonical | channel | name_norm with punctuation normalised (o'reilly->oreilly, l.l.c.->llc, & kept). Exactly equal on 15-17% (India) / 27% (US) of true pairs; 39% of S1 records share their name_clean. |
| 13 | `name_tokens` | List(String) | canonical | no | name_clean split on spaces (index terms, not a key). Rejoins exactly to name_clean. |
| 14 | `name_base` | String | auxiliary | channel | name_clean without legal-form tokens (incl. Indic scripts). Equal on 41-50% of true pairs, but merges 4 train / 30 test S1 pairs that differ only in legal form: use with name_legal_forms, never alone. Null when the name is only a legal form. |
| 15 | `name_legal_forms` | List(String) | auxiliary | no | Canonical legal-form classes in order (llc, inc, ltd, pvt, llp, sarl, ...), all scripts. |
| 16 | `name_domain` | String | auxiliary | channel | Domain found in the name (richardsonmexico.com); null otherwise. ~3-4% of S2/S3 names. |
| 17 | `name_numbers` | List(String) | auxiliary | no | Digit runs in the name, as written (includes appended phone numbers). |
| 18 | `name_script` | String | descriptive | no | 'Latin', a script name ('Devanagari', 'Tamil', ...), 'Latin+<script>' or 'None' (no letters). |
| 19 | `address_norm` | String | canonical | channel | Address with the same norm chain as name_norm; punctuation and placeholder segments kept. |
| 20 | `address_clean` | String | canonical | channel | Placeholder segments (NULL/null/<NULL>/N/A) removed, punctuation normalised. Exactly equal on only 4-14% of true pairs: address variation is token-level. |
| 21 | `address_tokens` | List(String) | canonical | no | address_clean split on spaces. |
| 22 | `address_segments` | List(String) | canonical | no | Comma segments of the address, each cleaned, original order, placeholders dropped. |
| 23 | `address_numbers` | List(String) | auxiliary | channel | Digit runs, leading zeros stripped (001->1). Same digit SET on 59-70% of true pairs. As-written digits: see numbers_as_written. |
| 24 | `postal_code_candidate` | String | auxiliary | no | Conservative country-scoped postcode; present for < 0.1% of records. Not a key. |
| 25 | `address_script` | String | descriptive | no | Script label of the raw address (as name_script). |
| 26 | `name_n_tokens` | UInt16 | descriptive | no | len(name_tokens). |
| 27 | `address_n_tokens` | UInt16 | descriptive | no | len(address_tokens). |
| 28 | `address_n_segments` | UInt16 | descriptive | no | len(address_segments). |
| 29 | `name_is_upper` | Boolean | descriptive | no | Raw name is all upper-case (S2: ~18%). |
| 30 | `name_is_lower` | Boolean | descriptive | no | Raw name is all lower-case. |
| 31 | `name_has_digits` | Boolean | descriptive | no | Raw name contains a digit. |
| 32 | `name_has_non_latin` | Boolean | descriptive | no | Raw name contains a non-Latin letter. |
| 33 | `name_has_latin_diacritics` | Boolean | descriptive | no | Raw name has accented Latin letters (real in France, injected noise in US/India). |
| 34 | `name_has_domain` | Boolean | descriptive | no | name_domain is not null. |
| 35 | `name_is_domain_only` | Boolean | descriptive | no | Whole name is a domain (words concatenated). |
| 36 | `name_has_special_chars` | Boolean | descriptive | no | Raw name contains # @ * _ / ~ ^ < > { } [ ] \ ". |
| 37 | `name_has_brackets` | Boolean | descriptive | no | Raw name contains brackets. |
| 38 | `name_has_leading_junk` | Boolean | descriptive | no | Raw name starts with a non-alphanumeric character (***, >>, --). |
| 39 | `name_has_repeated_whitespace` | Boolean | descriptive | no | Raw name contains 2+ consecutive spaces. |
| 40 | `name_has_encoding_artifact` | Boolean | descriptive | no | Raw name contains control / C1 / replacement characters. |
| 41 | `name_has_legal_form` | Boolean | descriptive | no | name_legal_forms is non-empty. |
| 42 | `name_legal_at_start` | Boolean | descriptive | no | First name token is a legal form (reordered name). |
| 43 | `name_legal_only` | Boolean | descriptive | no | Name consists only of legal forms (name_base is null). |
| 44 | `name_clean_empty` | Boolean | descriptive | no | Raw name present but name_clean empty (never true in this data). |
| 45 | `address_is_upper` | Boolean | descriptive | no | Raw address is all upper-case (S2: 52-66%). |
| 46 | `address_has_digits` | Boolean | descriptive | no | Raw address contains a digit. |
| 47 | `address_has_non_latin` | Boolean | descriptive | no | Raw address contains a non-Latin letter (Indic state names). |
| 48 | `address_has_latin_diacritics` | Boolean | descriptive | no | Raw address has accented Latin letters. |
| 49 | `address_has_special_chars` | Boolean | descriptive | no | Raw address contains special characters (#, ##, <, ...). |
| 50 | `address_has_encoding_artifact` | Boolean | descriptive | no | Raw address contains control / C1 / replacement characters (mojibake). |
| 51 | `address_has_placeholder` | Boolean | descriptive | no | Some comma segment is NULL/null/<NULL>/N/A. |
| 52 | `address_placeholder_only` | Boolean | descriptive | no | Address consisted only of placeholders (never true in this data). |
| 53 | `address_has_zero_padded_number` | Boolean | descriptive | no | Address contains a zero-padded number (0123). |
| 54 | `address_has_landmark` | Boolean | descriptive | no | Landmark words (near, opp, behind, ...). |
| 55 | `address_has_care_of` | Boolean | descriptive | no | Care-of markers (c/o, s/o, w/o, d/o, h/o). |
| 56 | `address_region` | String | auxiliary | no | ISO-3166-2-style region (US-OH, IN-MH, FR-HDF) named by a WHOLE address segment, per-country gazetteer (codes, full names, native-script names, French departments). Null when absent or ambiguous. Agrees on 99.5% of true pairs where both sides are found; disagrees on 85% (India) / 95% (US) of same-name decoys. Soft feature / optional sub-block only, never a hard filter (TG vs AP after the 2014 state split). |
| 57 | `name_latin` | String | auxiliary | no | anyascii 0.3.3 romanisation of name_clean for names containing non-Latin script, lowercase ASCII [a-z0-9 &]; null for Latin / letterless names. Fuzzy by construction (lksmi yunivrsl ...): for character similarity only. Median fuzz ratio to the S1 name 9.8 -> 77.6 on non-Latin true pairs. |
| 58 | `address_region_found` | Boolean | descriptive | no | address_region is not null. |

## On-demand representations (not stored)

| # | column | type | role | exact_key | meaning | computed by |
|---|---|---|---|---|---|---|
| 1 | `name_clean_folded` | String | auxiliary | channel | Accents removed from Latin letters (Indic vowel signs kept). Real spelling in France: keep both. | `T.fold_diacritics("name_clean")` |
| 2 | `name_base_folded` | String | auxiliary | channel | Accent-folded name_base. Equal on 43-56% of true pairs; same S1 caveat as name_base. | `T.fold_diacritics("name_base")` |
| 3 | `address_clean_folded` | String | auxiliary | channel | Accent-folded address_clean. | `T.fold_diacritics("address_clean")` |
| 4 | `numbers_as_written` | List(String) | auxiliary | no | Address digit runs with leading zeros kept. | `T.numbers("address_norm")` |
| 5 | `address_region_matches` | List(String) | auxiliary | no | Every region named by a whole segment (before ambiguity resolution). | `T.address_region_matches("address_segments", "country_clean")` |
