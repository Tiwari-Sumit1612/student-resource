# Preprocessing Readiness Audit

Business Entity Resolution, Stage 2 (preprocessing) before candidate generation.

## Verdict

**B. KEEP BUT ADD AUXILIARY REPRESENTATIONS.**

The canonical layer is sound and should be frozen. It is lossless for identity, deterministic, row-wise, label-free, and every lossy variant has a richer stored twin. No redesign and no change to existing columns is needed.

Two gaps are real, measured, and cut across both candidate generation and scoring. Both are deterministic, closed-vocabulary or rule-based, and add new columns without touching old ones:

1. `address_region`: canonical state/region code per address.
2. `name_latin`: romanised form of non-Latin names (null for Latin names).

Plus one small lexicon fix (Indic legal-form coverage) and one leakage check (row order). Everything else proposed below is computed on demand in later stages, not stored.

## What was inspected

| Input | Status |
|---|---|
| `entity_resolution_eda.ipynb` (124 cells, all outputs) | read in full |
| `preprocessing_comparison.ipynb` (65 cells, all outputs) | read in full |
| `preprocess.py`, `transforms.py`, `config.py`, `validate.py`, `measure.py` | **not available here**. Behaviour inferred from how the notebooks call them and from the 149 re-executed integrity checks |
| `REPORT.md`, processed Parquet, `results/*.csv` | **not available here**. Numbers taken from notebook outputs, which print those CSVs |
| Transliteration test | run here on the real non-Latin records printed in your notebooks |

Anything below marked **VERIFY** needs a look at the code or a run on the data, because I could not do it from the notebooks alone.

---

## 1. Current strengths

- **Integrity is proven, not claimed.** 149/149 checks pass: row counts, id sets, row order, byte-identical raw fields, missing flags equal raw nulls, tokens rejoin to clean strings, every digit of `name_norm` survives in `name_clean`.
- **Deterministic and row-wise.** A 1% rebuild equals the stored Parquet. That also proves there are no corpus-level statistics inside preprocessing (no IDF, no frequency thresholds), so no train/test statistic leakage is possible at this layer.
- **Layered, not destructive.** raw → norm → clean → base, with fold on demand. Each step's loss is still recoverable from the step before it.
- **Name + address keys are collision-free.** `name_clean + address_clean` merges 0 S1 pairs in train and 1 in test (a genuine near-duplicate: `72-74 RUE Royale` vs `72 -74 RUE Royale`). In S2/S3 it creates 1 mixed-entity group out of 114,605 shared keys.
- **No step destroys true-pair agreement.** `lost = 0` at every stage on 764,210 true pairs. The only exception is 2 pairs at legal-form removal where both names are only a legal form.
- **Right things kept auxiliary.** `name_base` is never canonical. The comparison measured why: 4 train / 30 test S1 pairs merge on `name_base + address_clean` (`Lille Sante SARL` vs `SCI`, `Orthopedic Center LLC` vs `Inc`).
- **Numbers handled well.** Zero-stripping in `address_numbers` gains +3.3% of true pairs and loses none. Digit sets agree on 59–70% of true pairs, far above whole-address equality (4–14%).
- **Accents stay in stored columns.** France is protected. Fold is only an on-demand expression.

## 2. Current weaknesses

Ranked by impact on F0.5.

### W1. No canonical region/state (high impact, precision and recall)

Region is written three different ways across sources and the pipeline treats them as unrelated tokens.

| Evidence (EDA 4.3, 6.3; comparison 8) | Value |
|---|---|
| US S1 addresses ending in a 2-letter state code | 86.4% |
| US S3 addresses ending in a 2-letter code (S3 writes `Ohio`, `Massachusetts`) | 4.3% |
| India S1 addresses ending in a 2-letter code | 0.0% |
| India S3 addresses ending in a 2-letter code (`MH`, `UP`, `KL`) | 58.9% |
| India S2/S3 addresses with non-Latin script, mostly the state name (`महाराष्ट्र`, `ગુજરાત`) | 10–12% |
| Median address-token Jaccard of true pairs, US S1–S2 vs US S1–S3 | 0.667 vs 0.455 |
| Exact address equality S1–S3 across every normalisation stage | flat at 4.1–4.4% |

So for most S1–S3 pairs, the state token can never agree today. At the same time the EDA decoy examples are separated almost entirely by state: `Ear Nose & Throat Specialists Corp` IL vs Montana, `Integrated Generation LLC` TX vs OH, `Dream Healthcare Pvt Ltd` Telangana vs Delhi. Name-only decoys hit 35–41% of S1 entities, including singletons. A region disagreement is the cheapest strong negative signal available, and today it is invisible.

### W2. Non-Latin names have zero usable name signal (high impact on India)

| Evidence (EDA 6.3, 6.5) | Value |
|---|---|
| True pairs with the match name in non-Latin script, overall | 7.3% |
| Same, India S1–S2 | 23.6% |
| Same, India S1–S3 | 13.1% |
| India S1–S2 true pairs sharing no name token | 28.6% |

Every stored name representation of these records is in Indic script, so name exact, token and character similarity against the Latin S1 name are all about zero. Address can still retrieve them (Indian S2 address Jaccard median is 0.83), but at scoring time the model sees "address matches, name unknown". Indian addresses are shared by different businesses (4.8% of India S1 addresses repeat), so that is exactly where false positives will come from.

Tested here on the real records from your notebooks (fuzzy ratio against the S1 name, 0–100):

| S1 name | raw Indic | romanised (anyascii) |
|---|---|---|
| Laxmi Universal Investment Private Limited | 9 | 84 |
| Dream Healthcare Pvt Ltd | 10 | 65 |
| Alpha Media Limited | 10 | 92 |
| Perfect Constructions Limited | 6 | 70 |

Romanisation does not create exact keys (`lksmi yunivrsl investmemt praivet limited`), but it moves these pairs from "no evidence" into the range where character similarity works. That is the whole point.

### W3. Indic legal-form lexicon has gaps (low impact, quick fix)

Devanagari and Kannada forms are recognised (`लिमिटेड` → ltd, `प्रा. लि.` → pvt,ltd, `ಲಿಮಿಟೆಡ್` → ltd). The Telugu example `బ్రైట్ కన్‌స్ట్రక్షన్ ఎల్‌ఎల్‌పీ` (LLP) got an empty `name_legal_forms` and kept the LLP inside `name_base`. That only affects the auxiliary `name_base` and legal-form features, but it will silently weaken the legal-form agreement feature for those scripts. **VERIFY**: coverage count of `name_has_legal_form` by `name_script` on India S2/S3.

### W4. Row-order leakage not checked (unknown impact, must check)

The pipeline preserves raw row order exactly (by design). Nobody has checked whether raw row order in S2/S3 correlates with the row order of their matched S1 entity. If it does, any sort-based or window-based blocking can silently exploit it on train and fail on test. See section 9.

### Not weaknesses (checked and fine)

- `postal_code_candidate` at < 0.1% coverage looks like a gap next to "5-digit tokens in 10.8% of US addresses", but US house numbers are often 5 digits (`14131 Jane Seymour Drive`), so those tokens are not ZIPs. Real postcodes are already inside `address_numbers`. No action.
- Legal-form removal harm is small in absolute terms and already contained by keeping `name_base` auxiliary.
- Name-only ambiguity rising after normalisation (15.5% → 22.3% of S2/S3 records in mixed groups) is a property of names, not a pipeline defect. The fix is address evidence at scoring time.

## 3. Information that is missing

| Missing | Failure mode it leaves open | Evidence |
|---|---|---|
| Canonical region | state code vs full name vs Indic script, decoys in other states | W1 |
| Latin form of non-Latin names | transliterated names, 7.3% of true pairs, 23.6% India S2 | W2 |
| Compact (space-free) name key | domain names `RICHARDSONMEXICO.COM`, collapsed spacing | 4% of names are domains |
| Street-type equivalence | `street`/`st`, `road`/`rd`, `rue`/`r` | US `street` 20.7% in S1 vs ~9.5% full + ~10.7% abbr in S2/S3; France `rue` 66% S1 vs 40% `rue` + 26% `r` in S2/S3 |
| Token-order invariant form | reordered names | 5.0% of true pairs differ only by order |
| Digit/letter confusion variant | `At1antic`, `Hea1th`, `D.0.` | residual examples in comparison 8 |
| Token rarity (IDF) | generic names (`Primary Care Group` ×253, `Bordeaux Club SARL` ×205) | EDA 2.3 |

Only the first two belong in the stored layer. Section 5 says why.

## 4. Transformations that should NOT be added

| Transformation | Why not | Evidence |
|---|---|---|
| Replace `name_clean` with `name_base` | merges different legal entities at one address | 4 train / 30 test S1 merges, 7 mixed S2/S3 groups |
| Store accent-folded columns as canonical | removes real French spelling | fold changes 15.7% of French S1 names, 27.8% of French S1 addresses; no French labels to check harm |
| Abbreviation expansion (`st` → `street`) | `st` = Saint, `dr` = Doctor, `r` = initial | EDA 4.4 |
| Destructive typo repair (`1` → `l`, `0` → `o`) in stored columns | breaks real alphanumerics (`3m`, `b2b`, `106 107 north owners corp`) | S1 names legitimately start with numbers |
| Stripping honorifics (`M/s`, `Shri`) or phone numbers from stored names | loses tokens, and the model can learn to discount them | residual examples in comparison 8 |
| Phonetic codes (Soundex, Metaphone) | English-only, useless on Indic and weak on French, collapses many distinct names so it raises decoy rate; typos like `Venutsre`/`Exelleht` are handled better by character n-grams | not measured, low expected value |
| Stored character n-grams | about 23 trigrams per name × 24.2M records ≈ 550M entries per field, and they are only needed inside an index | size estimate |
| Any rule derived from ground truth, or any corpus statistic (IDF, frequency cutoffs) inside preprocessing | breaks the row-wise, label-free guarantee | section 9 |
| Deleting records (distractors, junk names, no-letter names) | the 26% distractors are needed as negatives; singletons need a no-match decision | EDA 6.2 |

## 5. Transformations/features worth adding

### Store (auxiliary columns, new, nothing existing changes)

**A. `address_region`** (String, nullable) plus `address_region_found` (Boolean)

- What: canonical region code per country. US: 50 states + DC (name ↔ code). India: 36 states/UTs with full name, common codes (`MH`, `UP`, `KL`, `TG`/`TS`, `OD`/`OR`...) and native-script names. France: the 13 metropolitan regions (full name, accent-folded variant).
- How: per-country gazetteer, matched on **whole comma segments** of `address_segments` (and last token of a segment), never on arbitrary tokens. Match on both clean and folded text.
- Why store: used as a candidate sub-block, a hard-negative feature and a filter. Computing it repeatedly in every stage wastes time and risks drift between stages.
- Collision risk: code clashes across countries (`GA` Georgia/Goa, `OR` Oregon/Odisha, `MN`, `AR`, `CT`, `UT`) disappear because the gazetteer is chosen by `country_clean`, which is never null and never crosses a match. Inside the US, short codes like `IN`, `OR`, `ME`, `OK` are only accepted as a full segment.
- Storage: one short dictionary-encoded string column. A few MB per file.
- Use as a **feature and optional sub-block, never a hard filter.** 0.42% of true pairs have an equal name but a clearly different address (different city, PO box). A hard region filter would kill those.

**B. `name_latin`** (String, nullable)

- What: romanised `name_clean` when `name_script` is non-Latin, else null. Lowercased, ASCII, whitespace-collapsed.
- How: script-aware transliterator. Candidates: `anyascii` (all scripts, fast, drops inherent vowels: `mharastr`) or `indic-transliteration` ITRANS (schwa-aware: `maharashtra`, but needs script detection). Pick by the validation in section 9. Keep it deterministic and pinned to one library version.
- Why store: it is a Python UDF, not a vectorised Polars expression, so recomputing it per stage is slow. It feeds retrieval (character index) and every name-similarity feature.
- Scope: about 15% of Indian S2/S3 names, roughly 0.8M rows per split, about 6% of all rows. Null elsewhere.
- Collision risk: none on stored identity, because it is an extra column. As a key it is fuzzy by design, so it must never be used for exact matching or exact blocking.
- Also rerun legal-form detection on the Indic originals after fixing W3, so `name_legal_forms` is consistent across scripts.

### Compute on demand (do not store)

| Representation | Solves | Where | Cost |
|---|---|---|---|
| `compact = name_base without spaces` | domain names (`richardsonmexico` = compact S1 base), collapsed spacing | exact/containment channel in candidate generation, and a feature | 1 Polars expression; materialise only inside the index |
| sorted `name_tokens` | reordered names (5% of true pairs) | exact channel + feature | cheap |
| `fold_diacritics` on clean/base | French accent drops (`ECOLE` vs `École`), fake accents in train | retrieval uses folded, features keep both folded and unfolded equality | cheap, already exists |
| street-type canonicalisation, full → short only (`street`→`st`, `road`→`rd`, `avenue`→`ave`, `rue`→`r`), per country | abbreviation mismatch | address token features | cheap; full → short never invents meaning, short → full does |
| digit/letter fold inside mixed alphanumeric tokens only (`1`→`l`, `0`→`o`) | `at1antic`, `hea1th` | extra similarity feature | cheap; **VERIFY** frequency first |
| `&` = `and` = `et` equivalence | connector swaps | token features | unmeasured; measure before using |
| phone-like digit run (10+ digits) in name | `- 9635314554` suffixes inflate `name_numbers` mismatch | feature: discount name-number mismatch when phone-like | cheap |
| character 3-grams (hashed, sparse) | typos, partial tokens, romanised Indic | inside the candidate index per country block | never materialised as a table |
| token IDF | generic names, generic address words | features and blocking-key pruning | compute per split from unlabeled S1+S2+S3 text (see section 9) |

## 6. Candidate-generation readiness

**Ready now**, with expected ceilings from the ground-truth agreement table (comparison 8, share of true pairs exactly equal):

| Channel | Columns | Recall ceiling alone | Note |
|---|---|---|---|
| Hard block | `country_clean` | 100% | matches never cross countries |
| Exact clean name | `name_clean` | 15–17% India, 27% US | high decoy rate (7–15% precision) |
| Exact base name | `name_base` | 41–50% | needs address evidence after retrieval |
| Exact base + fold | `fold(name_base)` | 43–56% | best exact name channel; also covers France |
| Name token index | `name_tokens` | high | prune tokens by document frequency |
| Address number index | `address_numbers` | 59–70% digit-set equality; overlap much higher | cap postings for common numbers (`1`, `2`, `12`) |
| Address token index | `address_tokens` | median Jaccard 0.46–0.83 | prune generic tokens (`road`, `no`, state names) |
| Domain channel | `name_domain` + compact S1 base | covers most of the 4% domain names | compact key computed on demand |

**Not ready without the additions:**

- Region sub-blocking (needs `address_region`).
- Name retrieval for non-Latin records (needs `name_latin`). Until then these depend entirely on address channels.

**Not a preprocessing concern:** multilingual embeddings, BM25, FAISS. Decide those from the candidate-recall study, per channel and per script.

## 7. Model-training readiness

The processed layer supports a strong pairwise feature set. With the two additions, nothing needed for a first LightGBM model is missing.

**NAME**
- exact equality on norm / clean / base / fold(base) / compact / sorted tokens
- character similarity (Jaro-Winkler, normalised Levenshtein, 3-gram cosine) on clean, base, and `name_latin` vs S1 clean
- token Jaccard, IDF-weighted token overlap, containment (short name inside long name, truncated names like `Richardson Mexico,`)
- legal forms: equal / one side missing / conflicting (LLC vs Inc), legal form at start
- `name_numbers` agreement, with phone-like numbers discounted
- domain stem vs compact S1 base: equal, prefix, containment

**ADDRESS**
- `address_numbers` set: Jaccard, equality, first-number equality, IDF-weighted overlap
- token Jaccard on clean and on street-canonicalised tokens
- character similarity on clean
- `address_region` equal / conflict / unknown
- missing address on either side, placeholder present, landmark and care-of flags
- segment count difference, token count difference

**COUNTRY**
- country as a categorical feature, used for interactions only. It is already a hard block.
- France gets no special rules; it rides on language-agnostic features.

**STRUCTURAL**
- `name_script` pair (Latin–Latin, Latin–Devanagari...), script mismatch flag
- token counts and length differences
- `name_is_domain_only`, `name_legal_at_start`

**NOISE/QUALITY**
- the 30 descriptive flags from both sides (upper-case, repeated whitespace, encoding artifact, brackets, diacritics)
- note: several flags are near-perfect source fingerprints (S2 upper-cases 52–66% of addresses). Fine, because test has the same sources, but include `source` explicitly so the model does not learn it indirectly.

**CROSS-FIELD**
- name similarity × address similarity
- strong name AND region conflict (the decoy pattern)
- weak name AND strong address (the transliteration and shared-building pattern)
- name frequency in S1 (how many S1 entities share this name) × address agreement

**GLOBAL (post-scoring, not pairwise)**
- rank of this candidate among the S1 entity's candidates, score gap to the best
- rank of this S1 among all S1 entities competing for the same S2/S3 record (one-S1-per-record constraint)

## 8. Scalability concerns

| Item | Assessment |
|---|---|
| Current storage | 4.13 GB Parquet total (1.6–1.8× TSV). Fine. Read with column projection; candidate generation needs about 10 of 55 columns |
| `address_region` | vectorised Polars on `address_segments` (`list.eval` + `replace_strict` against a small mapping). Seconds per file |
| `name_latin` | Python UDF on ~0.8M strings per split. Filter to non-Latin rows first, then `map_elements` or batch. Minutes, once |
| Token inverted indexes | name tokens ≈ 45M rows per split exploded, address tokens ≈ 100M. Build per country, streaming, and drop tokens above a document-frequency cap before joining |
| Character n-grams | only as hashed sparse vectors or MinHash signatures inside a country block. Never as a long table |
| Largest block | India test: 0.81M S1 × 4.7M S2+S3. Size candidate caps (top-k per channel) against this block |
| Things to avoid | pandas, Python loops over 24M rows, pairwise all-vs-all inside a country, recomputing transliteration per stage |

## 9. Exact changes required before modeling

In this order. None of them modifies an existing column.

**1. Row-order leakage check (15 minutes).** On train, for every true pair take the row index of the S1 record and of the S2/S3 record, and compute Spearman correlation per source. Also compare the mean absolute index gap of true pairs against random pairs. If there is any structure, write it down and make sure no blocking step sorts or windows by raw row position. Never use row position or the numeric part of `entity_id` as a feature.

**2. Add `address_region` + `address_region_found`.** Then validate on the 10% ground-truth sample (evaluation only):

| Check | Pass bar |
|---|---|
| coverage per country × source | ≥ 90% US, ≥ 85% India, report France |
| agreement on true pairs where both sides are found | ≥ 97% |
| disagreement on same-name decoys (EDA 7.2 set) | clearly higher than on true pairs (report the rate) |
| S1 identity check: any new S1 merges | 0 (it is a new column, so this must hold) |

If agreement on true pairs is below the bar, inspect failures before using it as more than a soft feature.

**3. Add `name_latin`.** Validate on true pairs whose match name is non-Latin (7.3% of pairs):

| Check | Pass bar |
|---|---|
| median character similarity to the S1 name, before vs after | clear lift (my spot test: ~10 → 65–92) |
| share of those pairs above a usable similarity (for example Jaro-Winkler ≥ 0.8) | report per script |
| compare `anyascii` vs ITRANS | keep the one with higher median lift; pin the version |

**4. Fix Indic legal-form lexicon coverage.** Count `name_has_legal_form` by `name_script` on India S2/S3 and add missing forms (Telugu LLP at least). Rebuild only `name_legal_forms` and `name_base`. Rerun the 149 integrity checks.

**5. Rerun the comparison notebook sections 1, 6 and 8** on the rebuilt files, and store the new content fingerprints as the frozen version.

**6. Write the data dictionary** (one line per column: meaning, stored or on demand, canonical or auxiliary, safe as exact key or not). Candidate generation and feature code should import column names from it, not hard-code them.

**Leakage rules to keep for every later stage:**

- Ground truth is used only for labels and evaluation, never to create rules. Already true for preprocessing.
- IDF and other corpus statistics: compute from train text for train/validation experiments, and from test text (unlabeled S1+S2+S3) for test inference. Using unlabeled test text is not label leakage and is required for French vocabulary (0.01% of French names appear in train). Never mix train labels into those statistics.
- The French legal-form and region lists come from domain knowledge and from looking at unlabeled test text. That is fine, not target leakage.
- Validation split by S1 entity, stratified by country, keeping all of an entity's S2/S3 records in the same fold.

## 10. What should remain unchanged

- All raw columns, row order, ids, `split`, `source`, `country_clean`.
- `name_norm`, `name_clean`, `name_tokens` as the canonical name layer.
- `address_norm`, `address_clean`, `address_tokens`, `address_segments` as the canonical address layer.
- `name_base` + `name_legal_forms` as auxiliary, never alone.
- `address_numbers` with zero-stripping (as-written still recoverable from `address_norm`).
- `name_domain`, `name_numbers`, `name_script`, `address_script`.
- All 30 flags and 3 counts.
- Accent folding stays on demand.
- `postal_code_candidate` stays stored but unused as a key.
- The 149 integrity checks and the determinism check stay as the gate for any rebuild.

---

## Summary table

| Area | Status | Action |
|---|---|---|
| Information preservation | strong | none |
| Name preprocessing | good for Latin names; blind for non-Latin | add `name_latin` |
| Address preprocessing | good; region invisible | add `address_region` |
| Country handling | agnostic, France safe | none |
| GT validation | thorough, no lost agreement | rerun after additions |
| Collision risk | only `name_base`, contained | none |
| Candidate-generation readiness | exact, token, number, domain channels ready | region + Latin channels after additions |
| Model-training readiness | rich | none beyond additions |
| Leakage | clean at this layer | run row-order check |
| Scalability | fine | keep n-grams and IDF out of storage |
