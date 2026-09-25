"""Configuration for the preprocessing stage.

Every rule that encodes domain knowledge lives here so it can be reviewed in one place.
All lists were derived from the raw source files (see README.md "Evidence for rules");
the ground truth is NOT used to define any rule.
"""
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent            # .../student_resource
DATA_DIR = ROOT / "dataset"
PROCESSED_DIR = ROOT / "processed"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SPLITS = ("train", "test")
SOURCES = ("S1", "S2", "S3")


def raw_path(split: str, source: str) -> Path:
    return DATA_DIR / split / f"{split}_source{source[1]}.tsv"


def processed_path(split: str, source: str) -> Path:
    return PROCESSED_DIR / f"{split}_source{source[1]}.parquet"


GT_RAW = DATA_DIR / "train" / "train_ground_truth.tsv"
GT_PROCESSED = PROCESSED_DIR / "train_ground_truth.parquet"

RAW_COLUMNS = ("entity_id", "business_name", "business_address", "country")

# Parquet output settings (zstd gives ~3-4x on this text; statistics enable predicate push-down)
PARQUET = dict(compression="zstd", compression_level=6, statistics=True, row_group_size=250_000)

SEED = 42

# --------------------------------------------------------------------------- unicode repair
# UTF-8 punctuation that was decoded as Latin-1 somewhere upstream ("â€“" family): lead byte â/Â,
# then U+0080 + a C1 code point. Found in ~14k addresses (e.g. "Gali No. Â\x80\x93 01").
MOJIBAKE_REPAIRS = [
    (r"[âÂ]\x{80}[\x{93}\x{94}]", "-"),     # en/em dash
    (r"[âÂ]\x{80}[\x{98}\x{99}]", "'"),     # single quotes
    (r"[âÂ]\x{80}[\x{9c}\x{9d}]", '"'),     # double quotes
]
QUOTE_SINGLE = r"[‘’‚‛′`´ʼ]"
QUOTE_DOUBLE = r"[“”„‟″]"
DASHES = r"[‐‑‒–—―−]"

# --------------------------------------------------------------------------- scripts
LATIN_DIACRITIC = r"[\p{Latin}&&[^A-Za-z]]"
NON_LATIN_LETTER = r"[\p{L}&&[^\p{Latin}]]"
# Scripts observed in the data (all Indic, S2/S3 India only). Anything else falls back to "Other".
SCRIPTS = ("Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya", "Tamil", "Telugu", "Kannada", "Malayalam")

# --------------------------------------------------------------------------- names
# Legal / company-form tokens -> canonical class. Matched against *cleaned* tokens
# (lower-case, dotted acronyms joined: "L.L.C." -> "llc"). Indic forms are the most frequent
# non-Latin tokens in the data (e.g. लिमिटेड 636k, प्राइवेट 526k occurrences).
LEGAL_FORMS = {
    # US / generic English
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "ltd": "ltd", "limited": "ltd", "llp": "llp", "lp": "lp",
    "pllc": "pllc", "plc": "plc", "pc": "pc",
    # India
    "pvt": "pvt", "private": "pvt",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc",
    # Other
    "gmbh": "gmbh",
    # Indic transliterations (limited / private / abbreviations / LLP)
    "लिमिटेड": "ltd", "प्राइवेट": "pvt", "लि": "ltd", "प्रा": "pvt", "एलएलपी": "llp",
    "లిమిటెడ్": "ltd", "ప్రైవేట్": "pvt", "ఎల్ఎల్పీ": "llp",
    "ಲಿಮಿಟೆಡ್": "ltd", "ಪ್ರೈವೇಟ್": "pvt", "ಎಲ್ಎಲ್ಪಿ": "llp",
    "லிமிடெட்": "ltd", "பிரைவேட்": "pvt", "எல்எல்பி": "llp",
    "লিমিটেড": "ltd", "প্রাইভেট": "pvt", "এলএলপি": "llp",
    "લિમિટેડ": "ltd", "પ્રાઇવેટ": "pvt", "લિ": "ltd", "પ્રા": "pvt", "એલએલપી": "llp",
    "ലിമിറ്റഡ്": "ltd", "പ്രൈവറ്റ്": "pvt", "എൽഎൽപി": "llp",
    "ଲିମିଟେଡ୍": "ltd", "ପ୍ରାଇଭେଟ୍": "pvt", "ଏଲ୍ଏଲ୍ପି": "llp",
    "ਲਿਮਟਿਡ": "ltd", "ਪ੍ਰਾਈਵੇਟ": "pvt", "ਲਿ": "ltd", "ਪ੍ਰਾ": "pvt", "ਐਲਐਲਪੀ": "llp",
}

# Top-level domains accepted when detecting domain-like names ("wilfordhancock.com").
DOMAIN_TLDS = ("com", "net", "org", "biz", "info", "in", "co", "us", "fr", "io", "co.in", "org.in", "net.in")

NAME_SPECIAL_CHARS = r"[#@*_|~^<>{}\[\]\\\"]"
BRACKETS = r"[\[\](){}]"

# --------------------------------------------------------------------------- addresses
# Whole comma-separated segments that are placeholders injected by S2/S3 (≈44k each of
# NULL / null / <NULL> / N/A per file). Matched case-insensitively on a stripped segment.
# "NA"/"Na" are NOT treated as placeholders (ambiguous; they also occur in clean S1 data).
ADDRESS_PLACEHOLDER_SEGMENT = r"^(?:<\s*null\s*>|null|n/a)$"

LANDMARK = r"\b(?:near|nr|opp|opposite|behind|beside|next to|adjacent to|in front of|landmark)\b"
CARE_OF = r"\b[cswdh]\s?/\s?o\b"
ADDR_SPECIAL_CHARS = r"[#@*_|~^<>{}\[\]\\\"]"

# Postal-code candidates: deliberately conservative, applied to the normalised address.
# Evidence: US 5-digit numbers are almost always house numbers (1.06M of 1.24M are the leading
# token, 0 ZIP+4); India has no bare 6-digit PINs; France has a few standalone 5-digit segments.
# Countries not listed get no candidate.
POSTAL_RULES = {
    "US": r"(?:^|,)\s*(\d{5}(?:-\d{4})?)\s*(?:,|$)",
    "France": r"(?:^|,)\s*(\d{5})\s*(?:,|$)",
    # Indian PINs never start with 0; accepted only after a "pin"/"pincode" keyword or glued to a
    # place name with a hyphen ("vellore-632 009"). A looser rule matched house numbers ("a-001 496").
    "India": r"(?:\bpin(?:\s?code)?\s*(?:no\.?)?\s*[:\-.]?\s*|\p{L}{3,}\s?-\s?)([1-9]\d{2}\s?\d{3})\b",
}

# --------------------------------------------------------------------------- region gazetteer
# address_region: matched ONLY against whole comma segments of `address_segments`, with the
# gazetteer chosen by country_clean (so GA = Georgia never meets GA = Goa). Keys are written
# naturally here; transforms.py canonicalises them with the same norm -> punctuation -> fold
# chain as the address text. Values are ISO-3166-2-style codes. Source: domain knowledge; spelling
# variants and native-script names were read off unlabeled address text (never the ground truth).

US_STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado",
    "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}
# region -> (codes, names). Codes are accepted only as a whole segment, like every other key.
INDIA_STATES = {
    "IN-AP": (["AP"], ["Andhra Pradesh", "ఆంధ్రప్రదేశ్"]),
    "IN-AR": (["AR"], ["Arunachal Pradesh"]),
    "IN-AS": (["AS"], ["Assam"]),
    "IN-BR": (["BR"], ["Bihar", "बिहार"]),
    "IN-CT": (["CG", "CT"], ["Chhattisgarh", "Chattisgarh"]),
    "IN-GA": (["GA"], ["Goa"]),
    "IN-GJ": (["GJ"], ["Gujarat", "ગુજરાત"]),
    "IN-HR": (["HR"], ["Haryana", "हरियाणा"]),
    "IN-HP": (["HP"], ["Himachal Pradesh"]),
    "IN-JH": (["JH"], ["Jharkhand"]),
    "IN-KA": (["KA"], ["Karnataka", "ಕರ್ನಾಟಕ"]),
    "IN-KL": (["KL"], ["Kerala", "Keralam", "കേരളം"]),
    "IN-MP": (["MP"], ["Madhya Pradesh", "मध्य प्रदेश"]),
    "IN-MH": (["MH"], ["Maharashtra", "महाराष्ट्र"]),
    "IN-MN": (["MN"], ["Manipur"]),
    "IN-ML": (["ML"], ["Meghalaya"]),
    "IN-MZ": (["MZ"], ["Mizoram"]),
    "IN-NL": (["NL"], ["Nagaland"]),
    "IN-OR": (["OD", "OR"], ["Odisha", "Orissa", "ଓଡ଼ିଶା"]),
    "IN-PB": (["PB"], ["Punjab", "ਪੰਜਾਬ"]),
    "IN-RJ": (["RJ"], ["Rajasthan", "राजस्थान"]),
    "IN-SK": (["SK"], ["Sikkim"]),
    "IN-TN": (["TN"], ["Tamil Nadu", "Tamilnadu", "தமிழ்நாடு"]),
    "IN-TG": (["TG", "TS"], ["Telangana", "తెలంగాణ"]),
    "IN-TR": (["TR"], ["Tripura"]),
    "IN-UP": (["UP"], ["Uttar Pradesh", "उत्तर प्रदेश"]),
    "IN-UT": (["UK", "UA"], ["Uttarakhand", "Uttaranchal"]),
    "IN-WB": (["WB"], ["West Bengal", "পশ্চিমবঙ্গ"]),
    "IN-AN": (["AN"], ["Andaman and Nicobar Islands", "Andaman & Nicobar Islands"]),
    "IN-CH": (["CH"], ["Chandigarh"]),
    "IN-DH": (["DN", "DD"], ["Dadra and Nagar Haveli and Daman and Diu", "Dadra and Nagar Haveli", "Daman and Diu",
                             "Dadra & Nagar Haveli", "Daman & Diu"]),
    "IN-DL": (["DL"], ["Delhi", "New Delhi", "NCT of Delhi", "National Capital Territory of Delhi", "दिल्ली"]),
    "IN-JK": (["JK"], ["Jammu and Kashmir", "Jammu & Kashmir"]),
    "IN-LA": (["LA"], ["Ladakh"]),
    "IN-LD": (["LD"], ["Lakshadweep"]),
    "IN-PY": (["PY"], ["Puducherry", "Pondicherry"]),
}
# The 13 metropolitan regions; the 96 metropolitan departments map up to their region (hierarchy,
# not a heuristic: S2/S3 often write the department, e.g. "Nord", "Gironde", "Loire-Atlantique").
FRANCE_REGIONS = {
    "FR-ARA": ["Auvergne-Rhône-Alpes", "Ain", "Allier", "Ardèche", "Cantal", "Drôme", "Isère", "Loire", "Haute-Loire",
               "Puy-de-Dôme", "Rhône", "Savoie", "Haute-Savoie"],
    "FR-BFC": ["Bourgogne-Franche-Comté", "Côte-d'Or", "Doubs", "Jura", "Nièvre", "Haute-Saône", "Saône-et-Loire",
               "Yonne", "Territoire de Belfort"],
    "FR-BRE": ["Bretagne", "Côtes-d'Armor", "Finistère", "Ille-et-Vilaine", "Morbihan"],
    "FR-CVL": ["Centre-Val de Loire", "Cher", "Eure-et-Loir", "Indre", "Indre-et-Loire", "Loir-et-Cher", "Loiret"],
    "FR-COR": ["Corse", "Corse-du-Sud", "Haute-Corse"],
    "FR-GES": ["Grand Est", "Ardennes", "Aube", "Marne", "Haute-Marne", "Meurthe-et-Moselle", "Meuse", "Moselle",
               "Bas-Rhin", "Haut-Rhin", "Vosges"],
    "FR-HDF": ["Hauts-de-France", "Aisne", "Nord", "Oise", "Pas-de-Calais", "Somme"],
    "FR-IDF": ["Île-de-France", "Paris", "Seine-et-Marne", "Yvelines", "Essonne", "Hauts-de-Seine", "Seine-Saint-Denis",
               "Val-de-Marne", "Val-d'Oise"],
    "FR-NOR": ["Normandie", "Calvados", "Eure", "Manche", "Orne", "Seine-Maritime"],
    "FR-NAQ": ["Nouvelle-Aquitaine", "Charente", "Charente-Maritime", "Corrèze", "Creuse", "Dordogne", "Gironde",
               "Landes", "Lot-et-Garonne", "Pyrénées-Atlantiques", "Deux-Sèvres", "Vienne", "Haute-Vienne"],
    "FR-OCC": ["Occitanie", "Ariège", "Aude", "Aveyron", "Gard", "Haute-Garonne", "Gers", "Hérault", "Lot", "Lozère",
               "Hautes-Pyrénées", "Pyrénées-Orientales", "Tarn", "Tarn-et-Garonne"],
    "FR-PDL": ["Pays de la Loire", "Loire-Atlantique", "Maine-et-Loire", "Mayenne", "Sarthe", "Vendée"],
    "FR-PAC": ["Provence-Alpes-Côte d'Azur", "Alpes-de-Haute-Provence", "Hautes-Alpes", "Alpes-Maritimes",
               "Bouches-du-Rhône", "Var", "Vaucluse"],
}


def region_gazetteer() -> dict:
    """{country_clean: {"code": {raw key: region}, "name": {raw key: region}}} before canonicalisation."""
    us = {"code": {c: f"US-{c}" for c in US_STATES},
          "name": {n: f"US-{c}" for c, n in US_STATES.items()} | {"Washington DC": "US-DC", "Washington D.C.": "US-DC"}}
    india = {"code": {k: reg for reg, (codes, _) in INDIA_STATES.items() for k in codes},
             "name": {k: reg for reg, (_, names) in INDIA_STATES.items() for k in names}}
    france = {"code": {}, "name": {k: reg for reg, names in FRANCE_REGIONS.items() for k in names}}
    return {"US": us, "India": india, "France": france}


# When whole segments name two different regions, which key type wins (None = leave null).
# US: a whole-segment 2-letter code is always a state, but city names often equal state names
#     ("Washington, DC", "Delaware, OH", "Washington, IN")      -> the code wins.
# India: full state names are never fragments, but 2-letter fragments occur ("an", "ar" next to
#     "Maharashtra")                                             -> the full name wins.
# Conflicting names of the same type stay null (e.g. "Andhra Pradesh" + "Telangana").
REGION_PRECEDENCE = {"US": "code", "India": "name", "France": None}


# --------------------------------------------------------------------------- country
# Canonical labels; keys are lower-cased, whitespace-collapsed raw values. Observed values
# (US / India / France) map to themselves; aliases only guard against unseen spellings.
COUNTRY_CANONICAL = {
    "us": "US", "usa": "US", "united states": "US", "united states of america": "US",
    "india": "India", "in": "India", "ind": "India",
    "france": "France", "fr": "France", "fra": "France",
}
