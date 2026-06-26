import spacy
import re
from unicode_fol_kit import MSFLParser, NamingError, ParsingError
from utils.error_checking.analyser.scope import ScopeAnalyzer
from typing import Any
from difflib import SequenceMatcher
from tqdm import tqdm
from multiprocessing import Pool

spacy.prefer_gpu()


_TIME_ADVERBS = {"now", "today", "yesterday", "tomorrow", "later", "soon", "recently"}
_ADVERBS_OF_DURATION = {"forever", "permanently", "temporarily", "briefly"}
_ADVERBS_OF_FREQUENCY = {"often", "usually", "sometimes", "rarely", "occasionally"}
_ADVERBS_OF_MANNER = {"quickly", "slowly", "carefully", "easily", "hardly"}
_INEXPRESSIBLE_MODALS = {"could", "may", "might", "should", "ought", "would", "shall"}


# Parsing/equivalence is delegated to unicode-fol-kit. Its parser raises
# NamingError (lexer-level) and ParsingError (parser-level), and — unlike the
# previous local grammar — its rendered messages deliberately omit the verbose
# "Expected: <tokens>" list for structural-token errors. The two "after a complete
# sub-expression" categories are therefore told apart from the STRUCTURED
# expected-terminal set (resolved to literal symbols), reproducing the original
# string-matched classification exactly. The two exact sets below were the only
# ones the original mapped to those categories; every other expected set fell
# through to "Unknown error", so EXACT set equality is required (not a subset).
_AMBIGUITY_EXPECTED = frozenset({")", "→", "↔", "∧"})
_WRONG_PREDICATE_EXPECTED = frozenset({"=", "<", ">", "≤", "≥", "≠", "+", "-", "*", "/"})
_MIXING_CHARS = {"∧", "∨", "⊕"}
_PARSE_SENTINEL = "GROVES_PARSE::"


def _terminal_patterns(parser: MSFLParser) -> dict:
    """Map each terminal name to its literal/regex pattern for the given parser."""
    index = {}
    for terminal in parser.parser.terminals:
        pattern = terminal.pattern
        index[terminal.name] = pattern.value if hasattr(pattern, "value") else str(pattern)
    return index


def _expected_literals(exc, patterns: dict) -> frozenset:
    """Resolve an exception's expected/allowed terminal names to literal symbols."""
    names = getattr(exc, "allowed", None) or getattr(exc, "expected", None) or set()
    return frozenset(patterns.get(name, name) for name in names)


def classify_parse_error(exc, patterns: dict) -> str:
    """Map a unicode-fol-kit parse exception to a GROVES error category.

    Faithful re-implementation of the original local classification, reading the
    structured exception (type + message + expected-terminal set) instead of the
    verbose Lark message string:

      * ParsingError (unexpected token / premature EOF) -> "Unknown error"
        (the original mapped "Incomplete formula" to Unknown).
      * a lexer "Invalid <token>" error -> "Unallowed characters used".
      * an unexpected character right after a comma -> "Wrong constant/function
        naming" (a malformed argument).
      * an unexpected junctor (∧/∨/⊕) right after a closing parenthesis, where the
        expected set is EXACTLY {=,<,>,≤,≥,≠,+,-,*,/} -> "Wrong predicate naming"
        (a lowercase identifier parsed as a term where a predicate was intended);
        where it is EXACTLY {),→,↔,∧} -> "Syntactic ambiguity" (two different
        junctors mixed without parentheses). The "after closing parenthesis" and
        junctor-character conditions reproduce the original's requirement that the
        no-mixing hint be present, which is what made its string match succeed.
      * anything else -> "Unknown error".
    """
    if isinstance(exc, ParsingError):
        return "Unknown error"
    message = str(exc)
    if "Invalid" in message:
        return "Unallowed characters used"
    if "after comma" in message:
        return "Wrong constant/function naming"
    char = getattr(exc, "char", None)
    expected = _expected_literals(exc, patterns)
    if char in _MIXING_CHARS and "after closing parenthesis" in message:
        if expected == _AMBIGUITY_EXPECTED:
            return "Syntactic ambiguity"
        if expected == _WRONG_PREDICATE_EXPECTED:
            return "Wrong predicate naming"
    return "Unknown error"

def _load_spacy_model():
    """
    Loads the spacy transformer model for englisch without Dependency Parsing,
    Named Entity Recognition and Text Categorization.
    If the model is not found then the model will be downloaded.
    """
    SPACY_MODEL = "en_core_web_trf"
    try:
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner", "textcat"])
        return nlp
    except OSError:
        spacy.cli.download(SPACY_MODEL)
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner", "textcat"])

    return nlp


def _load_lemmatizer():
    """
    Loads a lightweight spaCy model used exclusively for lemmatizing short
    FOL name fragments (predicate parts, constants). The small model is
    ~40x faster than the transformer for this task and equally accurate on
    isolated words.
    """
    LEMMA_MODEL = "en_core_web_sm"
    try:
        nlp = spacy.load(LEMMA_MODEL, disable=["parser", "ner", "textcat"])
        return nlp
    except OSError:
        spacy.cli.download(LEMMA_MODEL)
        nlp = spacy.load(LEMMA_MODEL, disable=["parser", "ner", "textcat"])
    return nlp

def _get_relevant_word_tags() -> set[str]:
    """Returns the POS-Tags for words relevant to FOL formalization"""
    return {"NOUN", "PROPN", "VERB", "ADJ"}

def is_critical_adverb(lemma: str) -> bool:
    """
    Checks if the lemma of a verb is a critical adverb by a explicit blacklists.
    """
    if lemma in _TIME_ADVERBS or lemma in _ADVERBS_OF_DURATION or lemma in _ADVERBS_OF_FREQUENCY or lemma in _ADVERBS_OF_MANNER:
        return True
    return False

def is_critical_modal_verb(tag: str, lemma: str, text: str) -> bool:
    """
    Checks if a modal verb is inexpressible in FOL.
    Checks both lemma and surface form to handle spaCy lemmatization
    (e.g. could → can, might → may, would → will).
    """
    surface = text.lower()
    return tag == "MD" and (lemma in _INEXPRESSIBLE_MODALS or surface in _INEXPRESSIBLE_MODALS)


def _extract_fol_names_batch(formulas: list[str], nlp_lemma: Any) -> list[set[str]]:
    """Extrahiert FOL-Namen für alle Formulas in einem einzigen nlp.pipe()-Aufruf."""
    all_names = []
    formula_indices = []

    for idx, formula in enumerate(formulas):
        names = []
        for m in re.finditer(r"\b([A-Z][A-Za-z0-9_]*)\s*\(([^)]*)\)", formula):
            raw_pred = m.group(1)
            parts = re.findall(r"[A-Z][a-z]*|[a-z]+|\d+", raw_pred)
            names.extend(p.lower() for p in parts)
            args = [a.strip() for a in m.group(2).split(",") if a.strip()]
            names.extend(a.lower() for a in args)

        all_names.extend(names)
        formula_indices.extend([idx] * len(names))

    result: list[set[str]] = [set() for _ in formulas]
    for doc, idx in zip(nlp_lemma.pipe(all_names, batch_size=256), formula_indices):
        for token in doc:
            result[idx].add(token.lemma_.lower())

    return result


def _fuzzy_match(word1, word2, threshold=0.85) -> bool:
    """Fuzzy matches two words. Returns true if they are a match."""
    return SequenceMatcher(None, word1, word2).ratio() >= threshold

def _get_missing_words(nl_document: spacy.tokens.Doc, formula_words: set[str]) -> str | None:
    """Returns words that are not in the FOL but should be part of the formalization"""
    for word in nl_document:
            if word.pos_ in _get_relevant_word_tags() and not word.is_stop:
                not_found = True
                for w in formula_words:
                    if _fuzzy_match(word.lemma_.lower(), w.lower()) or word.lemma_.lower() in w.lower():
                        not_found = False
                        break
                if not_found:
                    return "SEMANTIC_ERROR: Missing relevant word '{}' in FOL.".format(word.lemma_)
    
    return None


def _process_batch(batch: list[dict]) -> list[dict]:
    """Wird in jedem Worker-Prozess ausgeführt – Parser einmal pro Batch erstellen"""
    parser = MSFLParser()
    patterns = _terminal_patterns(parser)

    for entry in batch:
        errors = []
        formula = entry["FOL"]
        try:
            parsed = parser.parse(formula)
            scope_analyzer = ScopeAnalyzer()
            try:
                scope_analyzer.visit(parsed)
            except Exception as e:
                errors.append(str(e))
        except (NamingError, ParsingError) as e:
            errors.append(_PARSE_SENTINEL + classify_parse_error(e, patterns))

        entry["_parser_errors"] = errors

    return batch


def analyze_errors(data: list[dict]) -> list[dict]:
    nlp = _load_spacy_model()
    nlp_lemma = _load_lemmatizer()
    texts = [entry["NL"] for entry in data]
    formulas = [entry["FOL"] for entry in data]

    docs = list(tqdm(nlp.pipe(texts), desc="spaCy NL pass", total=len(data)))
    formula_words_list = _extract_fol_names_batch(formulas, nlp_lemma)

    for entry, nl_doc, formula_words in tqdm(
        zip(data, docs, formula_words_list, strict=True),
        desc="Analyzing (spaCy)",
        total=len(data)
    ):
        errors = []

        for token in nl_doc:
            if is_critical_adverb(token.lemma_):
                errors.append(f"EXPRESSIVENESS_ERROR: Adverb '{token.lemma_}' is hard to express in FOL.")
                break
            elif is_critical_modal_verb(token.tag_, token.lemma_, token.text):
                errors.append(f"EXPRESSIVENESS_ERROR: Modal verb '{token.lemma_}' is hard to express in FOL.")
                break

        missing_words = _get_missing_words(nl_doc, formula_words)
        if missing_words:
            errors.append(missing_words)

        entry["_spacy_errors"] = errors

    n_workers = 24
    batch_size = max(1, len(data) // n_workers)
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    with Pool(processes=n_workers) as pool:
        processed_batches = list(tqdm(
            pool.imap(_process_batch, batches),
            total=len(batches),
            desc="Parsing (parallel)"
        ))

    data = [entry for batch in processed_batches for entry in batch]

    for entry in data:
        errors = entry.pop("_parser_errors") + entry.pop("_spacy_errors")

        new_errors = []
        for error in errors:
            if error.startswith(_PARSE_SENTINEL):
                # Parser/lexer error: already resolved to a category at parse time
                # (see classify_parse_error), since the structured expected-token
                # set is needed to reproduce the original classification.
                new_errors.append(error[len(_PARSE_SENTINEL):])
            elif error.startswith("SEMANTIC_ERROR: Missing relevant word"):
                new_errors.append("Possible incomplete formalization")
            elif error.startswith("FREE_VARIABLE_ERROR"):
                new_errors.append("Free variables")
            elif error.startswith("EXPRESSIVENESS_ERROR: Adverb"):
                new_errors.append("Hard to express adverb")
            elif error.startswith("EXPRESSIVENESS_ERROR: Modal verb"):
                new_errors.append("Hard to express modal verb")
            else:
                new_errors.append("Unknown error")

        entry["ERRORS"] = new_errors

    return data