#!/usr/bin/env python3
"""
BioSignal Pipeline Graph Extractor – Review Paper Edition
==========================================================
Usage:
  # Process a single review paper with section-aware semantic chunking
  python review_paper_pipeline_extractor.py \\
      --file review_paper.txt --provider ollama --model qwen3:32b

  # Disable section filtering (process full text)
  python review_paper_pipeline_extractor.py \\
      --file review_paper.txt --no-section-filter

  # Multiple review papers merged iteratively
  python review_paper_pipeline_extractor.py \\
      --files paper1.txt paper2.txt --output graph_review.json

  # Custom embedding model and threshold
  python review_paper_pipeline_extractor.py \\
      --file paper.txt --chunk-model all-mpnet-base-v2 --chunk-size 75

Requirements:
  pip install anthropic openai
  pip install langchain-experimental langchain-huggingface sentence-transformers
  pip install json-repair          # optional but recommended
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL_ANTHROPIC = "claude-opus-4-6"
DEFAULT_MODEL_OLLAMA    = "qwen3:32b"
OLLAMA_BASE_URL         = "http://139.30.7.163:11434/v1"

MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY   = 2   # seconds; doubles on each failure
# Maximum tokens for LLM output.  Large merged graphs can exceed 8 k tokens;
# set high enough to avoid response truncation mid-JSON.
MAX_OUTPUT_TOKENS  = 16384

# Semantic chunking defaults — lower threshold than one_shot to create
# finer-grained chunks that match technique subsections in review papers
DEFAULT_CHUNK_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE        = 90   # percentile; lower = more splits
DEFAULT_BATCH_SIZE        = 3    # chunks grouped into one LLM call

# Section-awareness: keywords that identify relevant vs. skip sections
RELEVANT_SECTION_KEYWORDS = {
    "signal processing", "pre-processing", "preprocessing",
    "filtering", "filter", "denoising", "noise removal", "artifact",
    "artefact", "segmentation", "windowing", "epoching",
    "normaliz", "normalis",
    "feature extraction", "feature engineering", "feature selection",
    "dimensionality reduction", "decomposition",
    "classification", "detection", "regression", "learning",
    "machine learning", "deep learning", "neural network",
    "evaluation", "validation", "benchmark",
    "method", "approach", "technique", "algorithm", "pipeline",
    "ecg processing", "eeg processing", "emg processing",
    "data preparation", "data augmentation", "resampling",
}

SKIP_SECTION_KEYWORDS = {
    "introduction", "background", "motivation", "related work",
    "literature review", "state of the art", "overview",
    "conclusion", "conclusions", "summary",
    "future work", "future direction", "open challenge", "open issue",
    "limitation", "discussion", "research direction", "research agenda",
    "standardiz",   # "Standardization" sections are agenda items
    "recommendation", "perspective", "outlook",
    "acknowledgment", "acknowledgement", "author contribution",
    "reference", "bibliography", "appendix",
    "abstract",  # already processed as a distinct text block
    # author biography markers
    "biography", "biographies", "received the", "received his", "received her",
    "he is currently", "she is currently", "is a professor", "is a senior member",
}

# Line prefixes that indicate figure/table captions or journal page headers —
# never valid section headers.
_NON_HEADER_PREFIXES = (
    "fig.", "figure ", "table ", "tab.",
    "ieee ", "acm ", "springer", "elsevier",
    "doi:", "http", "©", "volume", "vol.",
    "received ", "revised ", "accepted ",
)

# Minimum section word count to bother sending to the LLM
MIN_SECTION_WORDS = 80


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """\
You are an expert in biosignal processing and scientific experiment pipeline design.

This text is from a REVIEW or SURVEY paper. Review papers DESCRIBE what other
researchers have done — they do not perform operations themselves. Your task is
to extract the biosignal processing operations that the SURVEYED papers actually
perform, as reported in the review text.

Your role is to extract:
  1. Discrete biosignal processing OPERATIONS (what is done to the signal)
  2. RATIONALS (why each operation is applied — the signal/data condition
     or clinical constraint that necessitates it)
  3. CAUSAL CONNECTIONS between rationals and operations

Principles:
- Extract ONLY operations explicitly stated as USED, APPLIED, PERFORMED, or
  IMPLEMENTED in the surveyed literature.
- Use exact terminology from the source text.
- Every causality "source" MUST exactly match a rational string in the rationals list.
- Every causality "target" MUST exactly match an operation name in the operations list.
- Respond with properly formatted JSON only. Never use markdown code blocks.\
"""


REVIEW_ONE_SHOT_PROMPT = """\
Read the scientific text below (from a REVIEW / SURVEY paper) and extract
a biosignal processing pipeline graph in one step.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 1 — PROVENANCE FILTER  (apply before extracting anything)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Only consider sentences that use PERFORMED language:
  ✓ "was used", "is applied", "employs", "uses", "implements",
    "is performed", "was adopted", "was trained", "extracts",
    "computes", "segments", "filters", "normalises / normalizes"

Discard sentences that use RECOMMENDATION or META language:
  ✗ "should be", "needs to be", "it is important to", "future work",
    "is recommended", "could be explored", "is reviewed", "is discussed",
    "is presented", "are compared", "are surveyed"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 2 — EXTRACT OPERATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
From the provenance-filtered sentences, extract discrete biosignal processing
steps: upstream (data preparation, filtering, segmentation, normalisation,
denoising, quality control) and downstream (feature extraction, dimensionality
reduction, classification, evaluation).

For each candidate operation apply ALL FOUR tests:

  TEST A — SPECIFICITY (most important):
    The operation name MUST identify a specific method, algorithm, or technique.
    ✓ GOOD: "Pan-Tompkins QRS detector", "FIR baseline-wander filter", "SVM classifier",
      "discrete wavelet transform (DWT)", "FFPSO feature optimizer"
    ✗ BAD: "classification algorithms", "regression algorithms", "ML algorithms",
      "machine learning-based prediction", "signal processing techniques"
    If you cannot name the specific algorithm from the text, do not extract it.

  TEST A2 — ATOMICITY (one method per node, no exceptions):
    Each operation node must describe EXACTLY ONE method, algorithm, or technique.
    If the text lists multiple methods separated by commas, "and", "or", "/", or
    parenthetical examples, create one SEPARATE operation entry per method.
    ✗ INVALID (compound): "SVM, KNN, or Decision Trees"
    ✗ INVALID (compound): "DWT, band filter, adaptive filter or their variants"
    ✗ INVALID (compound): "direct suppression, ABS, bandpass filter, BSS, ICA, PCA"
    ✓ VALID: one entry "support vector machine (SVM) classifier" + one entry
             "k-nearest neighbors (KNN) classifier" + one entry "decision tree classifier"
    The name field must never contain a comma-separated list of algorithms.

  TEST B — METHOD not GOAL or STEP-DESCRIPTION:
    The name must identify a METHOD (an algorithm/technique), not a GOAL, PURPOSE,
    or step-description using action verbs.
    ✗ Reject names that start with a goal infinitive: "to detect", "to differentiate",
      "to predict", "to evaluate", "to identify", "algorithms to detect".
    ✗ Reject names that start with an action verb describing the step itself:
      "detect QRS complex" → use "QRS complex detection algorithm" or the specific
      method name (e.g., "Pan-Tompkins algorithm").
      "extract features from atrial activity" → name the specific transform used.
      Other rejected starts: "measure", "compute", "apply", "remove", "segment",
      "calculate", "perform", "use", "derive", "combine", "correct".
    ✓ If you must describe the step, convert to noun form and name the method:
      "detect R-peaks" → "R-peak detector" or the specific detection algorithm.
    ✗ Reject names that describe what classification is FOR rather than which
      classifier is USED.

  TEST C — EXECUTABILITY and SPECIFICITY:
    Ask: "Could this be called as a function on signal data?"
    ✗ EXCLUDE hardware devices or sensors (wearable sensor, ECG electrode, EBG structure)
    ✗ EXCLUDE deployment architecture (edge computing, cloud computing, IoT gateway)
    ✗ EXCLUDE system quality / network measures
      (QoS algorithm, transmission power control, energy harvesting)
    ✗ EXCLUDE single-word generic category labels used alone:
      "feature extraction" alone is NOT valid — name the specific transform or method.
      "feature selection" alone is NOT valid — name the specific selection algorithm.
      "classification" alone is NOT valid — name the specific classifier.
      "preprocessing" alone is NOT valid — name the specific preprocessing step.
      Always prefer the specific method name: "CWT", "ReliefF", "SVM", "bandpass filter".

  TEST D — PERFORMED not RECOMMENDED:
    ✗ Exclude if the text says "should be", "is needed", "there is a need to",
      "future work", "standardisation is needed", "needs to be explored".

PARAMETERS (mandatory):
  Capture EVERY numerical value, algorithm variant, hyperparameter, frequency band,
  window size, overlap, or configuration detail stated in the text.
  Write "not specified" only if the text contains no configuration detail.
  Never leave "parameters" as an empty object {} when the text contains details.

  Always capture these categories when present:
    • frequency_range_hz   — e.g. "5–26", "1–45", "50–60"
    • window_size_ms       — e.g. 500
    • algorithm_variant    — e.g. "LMS", "RLS", "IIR", "FIR", "Cal-Sal order"
    • noise_targets        — e.g. ["EMG", "PLI", "BLW"]
    • SNR_improvement_dB   — e.g. {"PLI": 22.36, "BLW": 11.56}
    • outperforms          — list of methods this outperforms per the text
    • advantages           — computational or accuracy advantages cited
    • limitations          — limitations or failure modes cited
    • n_features           — number of features cited
    • thresholds           — numeric thresholds or percentile values

DISTINCT NAME RULE (critical for deduplication):
  If the SAME base method appears in the text with TWO DIFFERENT configurations
  (e.g. different frequency ranges, different algorithm variants, different
  noise targets), they are NOT the same operation — create a SEPARATE entry for
  each configuration and encode the distinguishing parameter in the name:
    ✓ "bandpass filter (5–26 Hz)"   and   "bandpass filter (1–45 Hz)"
    ✓ "adaptive filter (LMS)"       and   "adaptive filter (RLS)"
    ✓ "wavelet transformation (QRS detection)"  and  "wavelet transformation (BLW removal)"
  Only operations with BOTH the same name AND the same key configuration are
  considered duplicates and should be merged into one entry.

TYPE CLASSIFICATION GUIDE:
  feature_extraction  — methods that CREATE new representations from signal data:
    wavelet transforms, FFT, Hilbert transform, entropy measures, band power, CWT, WTC
  feature_selection   — methods that CHOOSE WHICH existing features to keep/discard:
    ReliefF, Information Gain, LASSO regularization, Recursive Feature Elimination,
    Consistency-Based Filter, INTERACT, Correlation-Based Feature Selection, FFPSO
  dimensionality_reduction — methods that PROJECT features into fewer dimensions:
    PCA, ICA, LDA, UMAP, t-SNE
  filtering           — signal conditioning / noise suppression:
    bandpass filter, LMS, RLS, ANC, BSS, wavelet denoising, notch filter
  segmentation        — splitting signal into windows or events:
    Pan-Tompkins algorithm, QRS detector, sliding window, epoching
  preprocessing       — general signal conditioning before main pipeline:
    baseline removal, resampling, normalization, artifact correction
  classification      — assigning labels to feature vectors:
    SVM, KNN, CNN, LSTM, decision tree, random forest, logistic regression

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 3 — EXTRACT RATIONALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each operation, extract the reason stated or implied in the text for WHY
that operation is applied.

Apply the RATIONAL VALIDITY TEST:
  The rational must complete the sentence:
    "We apply [operation] BECAUSE [rational]"

  ✓ INCLUDE if the rational states a SIGNAL or DATA CONDITION:
    e.g., "wearable ECG contains motion artefacts and baseline wander"
  ✓ INCLUDE if the rational states a CLINICAL or APPLICATION CONSTRAINT:
    e.g., "AF episodes are short and irregular, requiring short windows"
  ✓ INCLUDE if the rational states a MODEL or COMPUTATIONAL REQUIREMENT:
    e.g., "CNN requires fixed-length input segments"

  ✗ EXCLUDE if the rational is CIRCULAR — it merely restates what the
    operation does: e.g., "to filter the signal" for a filtering step,
    or "to remove noise" for a noise removal step.
  ✗ EXCLUDE if the rational is only an OUTCOME or GOAL without a cause:
    e.g., "to improve accuracy", "for better performance"
    → rewrite as the underlying cause, or discard.
  ✗ EXCLUDE if the rational is the ENTIRE SYSTEM GOAL:
    e.g., "for AF diagnosis" (too broad — every operation serves this goal).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 STEP 4 — EXTRACT CAUSALITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each rational, go through EVERY operation and determine whether that
rational justifies applying that operation. A rational may justify multiple
operations. Produce one entry per valid (rational → operation) pair.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 SELF-VALIDATION (apply before writing final JSON)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Every causality "source" must be the exact "rational" string in the rationals list.
• Every causality "target" must be the exact "name" string in the operations list.
• Remove any causality whose source or target does not match exactly.
• Re-check each operation against all four tests (A–D above).
• Re-check each rational against the RATIONAL VALIDITY TEST.
• POLARITY: every causality edge must have polarity "positive".
  Negative edges (cannot apply / should not apply) are NOT allowed.
  If a rational explains why an operation is PROBLEMATIC rather than why it is USED,
  do NOT create an edge to that operation — instead, create an edge to the ALTERNATIVE
  operation that the text recommends as a result of that problem.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Text:
\"\"\"
${text}
\"\"\"

CRITICAL: Respond ONLY with the raw JSON object below.

{
  "operations": [
    {
      "name": "exact operation name from text",
      "type": "preprocessing | filtering | segmentation | normalization | feature_extraction | feature_selection | dimensionality_reduction | classification | evaluation | other",
      "parameters": {
        "parameter_name": "parameter value from text"
      },
      "text_evidence": "relevant quote from the text"
    }
  ],
  "rationals": [
    {
      "rational": "concise WHY statement: signal/data condition or clinical constraint",
      "type": "application-driven | data-driven | clinical-constraint | axiomatic",
      "text_evidence": "relevant quote from the text"
    }
  ],
  "causalities": [
    {
      "source": "EXACT rational string from rationals list",
      "target": "EXACT operation name from operations list",
      "polarity": "positive",
      "edge_type": "can apply | should apply | must apply",
      "text_evidence": "exact quote supporting this connection"
    }
  ]
}\
"""


REVIEW_MERGE_PROMPT = """\
You are extending a biosignal processing pipeline graph with content from a NEW
section of a REVIEW / SURVEY paper.

The graph already contains the operations and rationals listed below.
DO NOT re-output them — output ONLY items that are genuinely new.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ALREADY-KNOWN OPERATIONS (${operation_count} total — do NOT repeat)
 Format: name  [key distinguishing parameters]
 Use this to decide: same name + same config → skip (duplicate).
 Same name + DIFFERENT config → output as a NEW operation with config in name.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${existing_op_names}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ALREADY-KNOWN RATIONAL TEXTS (${rational_count} total — copy EXACTLY if reusing)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${existing_rat_texts}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ALREADY-KNOWN EDGES (${causality_count} total — do NOT repeat)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
${existing_edge_pairs}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXTRACTION RULES (same as first pass)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • PROVENANCE: only extract operations using PERFORMED language
    (used, applied, employs, implements, was adopted, extracts, segments, filters).
    Reject "should be", "is recommended", "future work", "is reviewed", "needs to be".
  • SPECIFICITY (Test A): operation name must identify a specific algorithm or
    technique. Reject generic labels ("classification algorithms", "ML algorithms").
    Also reject single generic category labels used alone: "feature extraction",
    "feature selection", "classification", "preprocessing" are NOT valid names —
    always use the specific method (e.g., "ReliefF", "SVM", "bandpass filter").
  • ATOMICITY (Test A2): each operation node must describe EXACTLY ONE method.
    If the text enumerates multiple methods (X, Y, and Z), create one entry per
    method — never put a comma-separated list in a single operation name.
  • DISTINCT NAME RULE: if the same base method appears with TWO DIFFERENT
    configurations (frequency range, algorithm variant, noise target), create a
    SEPARATE entry for each and encode the distinguishing parameter in the name:
      ✓ "bandpass filter (5–26 Hz)"  and  "bandpass filter (1–45 Hz)"
      ✓ "adaptive filter (LMS)"      and  "adaptive filter (RLS)"
    Only treat operations as duplicates when BOTH the name AND the key config match.
  • PARAMETERS: always capture frequency_range_hz, algorithm_variant, window_size_ms,
    noise_targets, filter_type, outperforms, advantages, limitations, n_features.
  • METHOD not GOAL or STEP-DESCRIPTION (Test B): reject names that describe
    goals ("to detect", "to differentiate") OR action-verb step descriptions
    ("detect QRS complex", "extract features from", "measure X", "compute Y").
    Convert to noun/method form: "detect R-peaks" → "R-peak detector" or specific
    algorithm name.
  • EXECUTABILITY (Test C): must be a signal-processing step; reject hardware,
    deployment architecture, and system quality/network measures (QoS, etc.).
  • RATIONAL validity: must complete "we apply [op] BECAUSE [rational]";
    must state a signal/data condition or clinical constraint; must NOT be
    circular or purely an outcome.
  • POLARITY: always "positive". No negative edges allowed.
    If a rational explains why an operation is PROBLEMATIC, create an edge to
    the ALTERNATIVE the text recommends instead — NOT to the problematic one.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY items not already in the lists above:
  • NEW operations found in the new text (not in already-known names list).
  • NEW rationals found in the new text (not semantically equivalent to already-known rationals).
  • NEW causality edges — including edges that connect an already-known rational
    to a new operation, or a new rational to an already-known operation.
    When referencing an already-known rational in a causality source, copy its
    text EXACTLY as it appears in the already-known rationals list above.
    When referencing an already-known operation in a causality target, copy its
    name EXACTLY as it appears in the already-known operations list above.

MANDATORY EDGE REQUIREMENT:
  Every new operation you output MUST have at least one causality entry whose
  "target" matches that operation's name exactly.
  If you cannot find any rational that justifies a new operation, DO NOT include
  that operation — an unconnected operation is worse than no extraction at all.
  Check: for each entry in "operations", confirm there is a matching entry in
  "causalities" with the same "target". Remove any operation without a match.

If nothing new is found, return empty lists for all three keys.

SELF-VALIDATION (before writing final JSON):
  • Every causality "source" must exactly match either a new rational text in your
    output OR one of the already-known rational texts above.
  • Every causality "target" must exactly match either a new operation name in your
    output OR one of the already-known operation names above.
  • Remove any causality that fails this check.

NEW TEXT:
\"\"\"
${text}
\"\"\"

CRITICAL: Respond ONLY with the raw JSON object below.
Output ONLY new items — do NOT repeat already-known operations, rationals, or edges.

{
  "operations": [
    {
      "name": "exact operation name",
      "type": "preprocessing | filtering | segmentation | normalization | feature_extraction | feature_selection | dimensionality_reduction | classification | evaluation | other",
      "parameters": { "parameter_name": "value" },
      "text_evidence": "quote"
    }
  ],
  "rationals": [
    {
      "rational": "concise WHY statement: signal/data condition or clinical constraint",
      "type": "application-driven | data-driven | clinical-constraint | axiomatic",
      "text_evidence": "quote"
    }
  ],
  "causalities": [
    {
      "source": "EXACT rational string (new or already-known)",
      "target": "EXACT operation name (new or already-known)",
      "polarity": "positive",
      "edge_type": "can apply | should apply | must apply",
      "text_evidence": "supporting quote"
    }
  ]
}\
"""


# ============================================================
# BACKEND ABSTRACTION  (same as one_shot_pipeline_extractor.py)
# ============================================================

class AnthropicBackend:
    def __init__(self, model: str, api_key: str):
        try:
            import anthropic as _anthropic
        except ImportError:
            print("ERROR: anthropic package not found.  Run: pip install anthropic")
            sys.exit(1)
        self.client = _anthropic.Anthropic(api_key=api_key)
        self.model  = model

    def chat(self, system: str, user_message: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


class OllamaBackend:
    def __init__(self, model: str, base_url: str):
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: openai package not found.  Run: pip install openai")
            sys.exit(1)
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.model  = model

    def chat(self, system: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
            ],
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            print(
                "  ⚠  WARNING: model response was cut off (finish_reason='length'). "
                "The JSON may be truncated. Consider reducing --batch-size or "
                "processing fewer sections per call.",
                file=sys.stderr,
            )
        return choice.message.content


# ============================================================
# UTILITIES
# ============================================================

def fill_template(template: str, variables: dict) -> str:
    result = template
    for key, value in variables.items():
        result = result.replace(f"${{{key}}}", str(value))
    return result


def parse_llm_json(raw: str, context: str = "") -> dict:
    """
    Parse JSON from an LLM response.
    Strips markdown code-fence wrappers and Qwen <think> tags if present.
    Falls back to json_repair if available when standard parsing fails.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            from json_repair import repair_json
            repaired = repair_json(text, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except ImportError:
            pass
        raise ValueError(
            f"JSON parse error [{context}]: {exc}\n"
            f"--- Response snippet ---\n{text[:500]}"
        ) from exc


def call_llm(backend, system: str, user_message: str,
             context: str, verbose: bool) -> str:
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            if verbose:
                print(f"    [{context}] calling model (attempt {attempt}/{MAX_RETRY_ATTEMPTS}) …")
            return backend.chat(system, user_message)
        except Exception as exc:
            if attempt == MAX_RETRY_ATTEMPTS:
                raise RuntimeError(
                    f"[{context}] failed after {MAX_RETRY_ATTEMPTS} attempts: {exc}"
                ) from exc
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"    [{context}] error: {exc}  – retrying in {delay}s …")
            time.sleep(delay)


# ============================================================
# SECTION-AWARE TEXT SPLITTING
# ============================================================

# Matches common section header patterns in plain-text papers:
#   "2. Signal Processing"   "3.1 Feature Extraction"   "IV. METHODS"
# Requires the line to start with an uppercase letter (not a digit-only token
# like a page number) or a Roman/Arabic section number followed by a word.
_SECTION_HEADER_RE = re.compile(
    r"^(?:(?:[IVXivx]+\.?|[A-Z]\.|\d+(?:\.\d+)*\.?)\s+)?([A-Z][A-Za-z][^\n]{2,70})$",
    re.MULTILINE,
)


def _header_is_relevant(header_text: str) -> bool | None:
    """
    Return True  → keep section
    Return False → skip section
    Return None  → ambiguous (keep by default)
    """
    lower = header_text.lower()
    for kw in SKIP_SECTION_KEYWORDS:
        if kw in lower:
            return False
    for kw in RELEVANT_SECTION_KEYWORDS:
        if kw in lower:
            return True
    return None   # ambiguous — keep


def split_into_sections(text: str, verbose: bool) -> list[tuple[str, str]]:
    """
    Split text on section headers.
    Returns list of (header_title, body_text) pairs.
    All text before the first detected header is treated as a preamble
    with header "PREAMBLE".
    """
    # Find all potential headers and their positions
    matches = list(_SECTION_HEADER_RE.finditer(text))
    if not matches:
        return [("FULL_TEXT", text)]

    sections = []
    prev_end  = 0
    prev_title = "PREAMBLE"

    for m in matches:
        line = m.group(0).strip()
        # Skip lines that are too long to be headers
        if len(line) > 100:
            continue
        # Skip figure/table captions, journal masthead lines, URLs, etc.
        line_lower = line.lower()
        if any(line_lower.startswith(pfx) for pfx in _NON_HEADER_PREFIXES):
            continue
        # Skip lines that look like page-number + journal name (all-caps blob)
        # e.g. "220 IEEEREVIEWSINBIOMEDICALENGINEERING,VOL.14,2021"
        if re.match(r"^\d+\s+[A-Z]{5,}", line):
            continue

        body = text[prev_end:m.start()].strip()
        if body:
            sections.append((prev_title, body))
        prev_title = line
        prev_end   = m.end()

    # Last section
    tail = text[prev_end:].strip()
    if tail:
        sections.append((prev_title, tail))

    if verbose:
        print(f"\n  [Section split] Detected {len(sections)} sections:")
        for title, body in sections:
            decision = _header_is_relevant(title)
            tag = "KEEP" if decision is not False else "SKIP"
            print(f"    [{tag}] {title[:80]} ({len(body.split())} words)")

    return sections


def filter_sections(sections: list[tuple[str, str]], verbose: bool) -> list[tuple[str, str]]:
    """
    Keep only sections likely to contain pipeline operations.
    Ambiguous sections (None) are kept.

    Phase 20 change: short sections (< MIN_SECTION_WORDS) that are NOT
    explicitly irrelevant are no longer dropped.  Instead they are
    accumulated and merged (prepended) into the next long section so
    their text arrives in the LLM's context alongside surrounding content.
    This prevents important short filter/feature paragraphs created by
    2-column PDF extraction from being silently discarded.

    Truly irrelevant sections (SKIP keyword) are still dropped entirely.
    Truly tiny sections (< TINY_SECTION_WORDS = 5 words) are also dropped.
    """
    TINY_SECTION_WORDS = 5          # below this even merging is pointless

    kept   = []
    pending: list[tuple[str, str]] = []   # short sections waiting to merge

    def _flush_pending(into_body: str) -> str:
        """Prepend accumulated short-section text to the body of a long section."""
        nonlocal pending
        if not pending:
            return into_body
        prefix = "\n\n".join(b for _, b in pending)
        pending = []
        return prefix + "\n\n" + into_body

    for title, body in sections:
        word_count = len(body.split())

        # Always discard explicitly irrelevant sections
        decision = _header_is_relevant(title)
        if decision is False:
            if verbose:
                print(f"  [Section filter] SKIP (irrelevant): {title[:60]}")
            continue

        # Truly tiny — not worth merging
        if word_count < TINY_SECTION_WORDS:
            if verbose:
                print(f"  [Section filter] SKIP (too short, {word_count} words): {title[:60]}")
            continue

        if word_count < MIN_SECTION_WORDS:
            # Short but possibly relevant — accumulate for merging
            if verbose:
                print(f"  [Section filter] MERGE-PENDING ({word_count} words): {title[:60]}")
            pending.append((title, body))
        else:
            # Long section — merge any pending short sections into it
            merged_body = _flush_pending(body)
            kept.append((title, merged_body))

    # Flush any remaining short sections as a combined chunk
    if pending:
        merged_body = "\n\n".join(b for _, b in pending)
        merged_title = "merged short sections"
        if verbose:
            print(f"  [Section filter] MERGE-TRAILING: flushing {len(pending)} short section(s)")
        kept.append((merged_title, merged_body))

    if verbose:
        print(f"  [Section filter] Keeping {len(kept)}/{len(sections)} sections "
              f"(short sections merged, not dropped)")
    return kept


# ============================================================
# SEMANTIC CHUNKING  (within a section body)
# ============================================================

def chunk_text_semantically(text: str, embed_model: str, percentile: int,
                             verbose: bool) -> list[str]:
    """
    Split *text* into semantically coherent chunks.
    percentile: breakpoint threshold passed to SemanticChunker [50–99].
    Higher = fewer, larger chunks.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain.text_splitter import SemanticChunker
            from langchain.embeddings import HuggingFaceEmbeddings
        except ImportError:
            print(
                "ERROR: Semantic chunking requires additional packages.\n"
                "  pip install langchain-experimental langchain-huggingface "
                "sentence-transformers"
            )
            sys.exit(1)

    percentile = max(50, min(99, percentile))
    embeddings  = HuggingFaceEmbeddings(model_name=embed_model)
    chunker     = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile,
    )
    chunks = chunker.split_text(text)
    if verbose:
        for i, c in enumerate(chunks):
            print(f"      sub-chunk {i+1}: {len(c.split())} words")
    return chunks


def batch_chunks(chunks: list[tuple[str, str]],
                 batch_size: int) -> list[tuple[str, str]]:
    """
    Group consecutive chunks into batches of at most batch_size.
    Adjacent texts are joined with a clear section-break marker so the
    LLM knows where one section ends and the next begins.
    Each batch is processed as a single LLM call, reducing total calls
    by a factor of batch_size.

    Example: 60 chunks with batch_size=3 → 20 LLM calls instead of 60.
    """
    if batch_size <= 1:
        return chunks
    batched = []
    for i in range(0, len(chunks), batch_size):
        group = chunks[i : i + batch_size]
        combined_text  = "\n\n\n=== SECTION BREAK ===\n\n\n".join(t for t, _ in group)
        combined_label = f"[batch {i//batch_size + 1}: {group[0][1]} … {group[-1][1]}]"
        batched.append((combined_text, combined_label))
    return batched


def prepare_chunks(raw_texts: list[str], raw_labels: list[str],
                   use_section_filter: bool, chunk: bool,
                   embed_model: str, percentile: int,
                   batch_size: int,
                   verbose: bool) -> list[tuple[str, str]]:
    """
    Convert raw texts into (text, label) pairs for the extraction pipeline.

    If use_section_filter: split each text into sections, discard irrelevant
    ones, then optionally apply semantic chunking within each kept section.
    If not use_section_filter and chunk: apply semantic chunking to full text.
    Otherwise: use text as-is.

    After all splitting, consecutive chunks are grouped into batches of
    batch_size so multiple sections are processed in one LLM call.
    """
    result = []

    for raw_text, raw_label in zip(raw_texts, raw_labels):

        if use_section_filter:
            sections = split_into_sections(raw_text, verbose)
            kept     = filter_sections(sections, verbose)

            for sec_title, sec_body in kept:
                sec_label = f"{raw_label} [{sec_title[:50]}]"
                if chunk:
                    if verbose:
                        print(f"\n  [Chunking section] '{sec_title[:50]}' …")
                    sub_chunks = chunk_text_semantically(
                        sec_body, embed_model, percentile, verbose
                    )
                    for i, c in enumerate(sub_chunks):
                        result.append((c, f"{sec_label} [sub {i+1}/{len(sub_chunks)}]"))
                else:
                    result.append((sec_body, sec_label))

        elif chunk:
            if verbose:
                print(f"\n  [Chunking] '{raw_label}' …")
            sub_chunks = chunk_text_semantically(
                raw_text, embed_model, percentile, verbose
            )
            for i, c in enumerate(sub_chunks):
                result.append((c, f"{raw_label} [chunk {i+1}/{len(sub_chunks)}]"))

        else:
            result.append((raw_text, raw_label))

    return batch_chunks(result, batch_size)


# ============================================================
# EXTRACTION & MERGE
# ============================================================

def one_shot_extract(backend, text: str, verbose: bool) -> tuple:
    """
    First source: extract operations, rationals, and causalities in one call.
    Returns (operations, rationals, causalities) or ([], [], []) on failure.
    """
    if verbose:
        print("\n  [Extract] one-shot extraction …")

    prompt = fill_template(REVIEW_ONE_SHOT_PROMPT, {"text": text})
    try:
        raw  = call_llm(backend, SYSTEM_PROMPT, prompt, "extract", verbose)
        data = parse_llm_json(raw, "extract")
    except (RuntimeError, ValueError) as exc:
        if verbose:
            print(f"    WARNING: extraction failed ({exc.__class__.__name__}) – skipping chunk.")
        return [], [], []

    ops  = data.get("operations",  [])
    rats = data.get("rationals",   [])
    deps = data.get("causalities", [])

    if not ops:
        if verbose:
            print("    → No operations found in this chunk – skipping.")
        return [], [], []

    if verbose:
        print(f"    → {len(ops)} ops / {len(rats)} rationals / {len(deps)} causalities")
        if len(ops) <= 12:
            print(json.dumps(ops, indent=2))

    return ops, rats, deps


def one_shot_merge(backend, text: str, existing_ops: list, existing_rats: list,
                   existing_deps: list, verbose: bool) -> tuple:
    """
    Subsequent sources: extract ONLY new items from the new text, then
    union them with the existing graph on our side.

    The prompt sends compact reference lists (names / texts only) so its size
    stays O(1) regardless of how many nodes are already in the graph.
    """
    if verbose:
        print("\n  [Merge] extracting new items from chunk …")

    # ── Build compact references for the prompt ───────────────────────────
    known_op_names  = {op["name"]      for op in existing_ops  if op.get("name")}
    known_rat_texts = {r["rational"]   for r  in existing_rats if r.get("rational")}
    known_edges     = {(d["source"], d["target"])
                       for d in existing_deps
                       if d.get("source") and d.get("target")}

    # Show name + key distinguishing params so LLM knows which configs are known
    def _key_params_summary(op: dict) -> str:
        """Return a short bracket string of distinguishing params, or empty."""
        params = op.get("parameters") or {}
        if not isinstance(params, dict) or not params:
            return ""
        SHOW_KEYS = ("frequency_range_hz", "algorithm_variant", "noise_targets",
                     "filter_type", "type", "variant", "band", "window_size_ms",
                     "order", "components", "stop_frequency_hz", "purposes")
        parts = []
        for k in SHOW_KEYS:
            v = params.get(k)
            if v is not None:
                if isinstance(v, list):
                    v = "+".join(str(x) for x in v)
                parts.append(str(v))
        return f"  [{', '.join(parts)}]" if parts else ""

    op_name_list  = "\n".join(
        f"  - {op['name']}{_key_params_summary(op)}"
        for op in sorted(existing_ops, key=lambda x: x.get("name",""))
        if op.get("name")
    )
    rat_text_list = "\n".join(f"  - {t}" for t in sorted(known_rat_texts))
    edge_list     = "\n".join(f"  - {s} → {t}" for s, t in sorted(known_edges))

    prompt = fill_template(REVIEW_MERGE_PROMPT, {
        "operation_count":   len(existing_ops),
        "rational_count":    len(existing_rats),
        "causality_count":   len(existing_deps),
        "existing_op_names": op_name_list  or "  (none yet)",
        "existing_rat_texts": rat_text_list or "  (none yet)",
        "existing_edge_pairs": edge_list   or "  (none yet)",
        "text":              text,
    })

    try:
        raw  = call_llm(backend, SYSTEM_PROMPT, prompt, "merge", verbose)
        data = parse_llm_json(raw, "merge")
    except (RuntimeError, ValueError) as exc:
        if verbose:
            print(f"    WARNING: merge failed ({exc.__class__.__name__}) – keeping existing graph.")
        return existing_ops, existing_rats, existing_deps

    new_ops  = data.get("operations",  [])
    new_rats = data.get("rationals",   [])
    new_deps = data.get("causalities", [])

    # ── Union: add new items, skip duplicates ─────────────────────────────
    # Dedup ops by name + param fingerprint: same name + same key config = duplicate;
    # same name + different config = distinct op (LLM should encode config in name,
    # but as a safety net we also check a simple param fingerprint here).
    def _op_fingerprint(op: dict) -> str:
        """Stable fingerprint of an operation's name + key discriminating params."""
        name = op.get("name", "").lower().strip()
        params = op.get("parameters") or {}
        if not isinstance(params, dict):
            return name
        DISCRIM = ("frequency_range_hz", "algorithm_variant", "filter_type",
                   "noise_targets", "stop_frequency_hz", "window_size_ms",
                   "type", "variant", "band", "order")
        fp_parts = []
        for k in DISCRIM:
            v = params.get(k)
            if v is not None:
                if isinstance(v, list):
                    v = tuple(sorted(str(x).lower() for x in v))
                fp_parts.append((k, str(v).lower()))
        return name + "|" + str(sorted(fp_parts))

    known_op_fingerprints = {_op_fingerprint(op) for op in existing_ops if op.get("name")}

    added_ops = []
    for op in new_ops:
        if not op.get("name"):
            continue
        fp = _op_fingerprint(op)
        if op["name"] in known_op_names and fp in known_op_fingerprints:
            continue  # true duplicate: same name AND same config
        added_ops.append(op)
        known_op_fingerprints.add(fp)

    added_rats = [r  for r  in new_rats if r.get("rational")  not in known_rat_texts]

    # All edge targets/sources must resolve against the combined node set
    merged_op_names  = known_op_names  | {op["name"]    for op in added_ops  if op.get("name")}
    merged_rat_texts = known_rat_texts | {r["rational"] for r  in added_rats if r.get("rational")}

    added_deps = [
        d for d in new_deps
        if d.get("source") in merged_rat_texts
        and d.get("target") in merged_op_names
        and (d["source"], d["target"]) not in known_edges
    ]

    merged_ops  = existing_ops  + added_ops
    merged_rats = existing_rats + added_rats
    merged_deps = existing_deps + added_deps

    if verbose:
        print(f"    +{len(added_ops)} ops, +{len(added_rats)} rationals, "
              f"+{len(added_deps)} edges  "
              f"(totals: {len(merged_ops)} ops / {len(merged_rats)} rats / {len(merged_deps)} edges)")

    return merged_ops, merged_rats, merged_deps


def process_chunk(backend, text: str, existing_graph: dict | None,
                  verbose: bool) -> tuple:
    """
    Dispatch to extract (first chunk) or merge (subsequent chunks).
    Returns (operations, rationals, causalities).
    """
    if existing_graph is None:
        return one_shot_extract(backend, text, verbose)

    ex_ops, ex_rats, ex_deps = _strip_graph_ids(existing_graph)
    ops, rats, deps = one_shot_merge(backend, text, ex_ops, ex_rats, ex_deps, verbose)
    if not ops:
        return ex_ops, ex_rats, ex_deps
    return ops, rats, deps


# ============================================================
# GRAPH BUILDER  (same logic as one_shot_pipeline_extractor.py)
# ============================================================

_POSITIVE_EDGE_TYPES = {"can apply", "should apply", "must apply"}
_NEGATIVE_EDGE_TYPES = {"cannot apply", "should not apply", "must not apply"}


def _deduplicate_rationals(rationals: list) -> list:
    """
    Merge rational nodes with identical 'rational' text.
    Combines text_evidence with ' | ADDITIONAL: ' separator.
    """
    seen: dict[str, dict] = {}
    for r in rationals:
        key = r.get("rational", "").strip()
        if not key:
            continue
        if key in seen:
            existing_ev = seen[key].get("text_evidence", "")
            new_ev      = r.get("text_evidence", "")
            if new_ev and new_ev not in existing_ev:
                seen[key]["text_evidence"] = existing_ev + " | ADDITIONAL: " + new_ev
        else:
            seen[key] = dict(r)
    return list(seen.values())


def _fix_edge_polarity_consistency(dependencies: list) -> list:
    """
    Phase 19: negative edges are banned — rationals only explain why to USE an op.
    Drop any edge with polarity="negative" or a negative edge_type, and warn.
    Also normalise polarity to "positive" on any remaining edge.
    """
    fixed = []
    dropped = 0
    for dep in dependencies:
        polarity  = dep.get("polarity", "positive")
        edge_type = dep.get("edge_type", "can apply")
        if polarity == "negative" or edge_type in _NEGATIVE_EDGE_TYPES:
            dropped += 1
            continue   # discard — rational should point to an alternative instead
        dep = dict(dep)
        dep["polarity"] = "positive"
        if edge_type not in _POSITIVE_EDGE_TYPES:
            dep["edge_type"] = "can apply"
        fixed.append(dep)
    if dropped:
        print(f"  ⚠  Dropped {dropped} negative edge(s) (polarity=negative / cannot-apply). "
              f"Rationals must only motivate usage, not exclusion.", file=sys.stderr)
    return fixed


def _warn_compound_nodes(operations: list, verbose: bool = True) -> None:
    """
    Detect operation nodes whose name contains a comma-separated list of methods
    or obvious enumerations (X, Y, and Z; or A / B / C).  These violate the
    ATOMICITY rule (Test A2) and indicate the LLM ignored the prompt constraint.
    Prints a WARNING for each compound node found so they surface in the run log.
    """
    # Heuristic: name contains ≥2 commas, or matches " and " / " or " between
    # what look like named methods (after a comma), or contains " / " separator.
    _ENUM_PATTERN = re.compile(
        r"(?:,\s*(?:and\s+|or\s+)?[A-Za-z(])"  # comma-separated list
        r"|(?:\s+/\s+[A-Za-z(])"                 # slash-separated
        r"|(?:,.*,)",                             # two or more commas
        re.IGNORECASE,
    )
    compound = [op for op in operations if _ENUM_PATTERN.search(op.get("name", ""))]
    if compound and verbose:
        print(f"\n  ⚠  ATOMICITY VIOLATIONS — {len(compound)} compound operation node(s) detected:")
        for op in compound:
            print(f"       • [{op.get('type','?')}] {op['name']}")
        print("     These should be split into one node per method.\n")


def build_graph(operations: list, rationals: list, dependencies: list,
                sources: list, model: str, provider: str) -> dict:
    # Post-processing fixes before graph construction
    _warn_compound_nodes(operations, verbose=True)
    rationals    = _deduplicate_rationals(rationals)
    dependencies = _fix_edge_polarity_consistency(dependencies)

    dependencies = [
        dep for dep in dependencies
        if dep.get("source") and dep.get("target")
    ]

    known_op_names  = {op["name"]    for op in operations if op.get("name")}
    known_rat_names = {r["rational"] for r  in rationals  if r.get("rational")}

    valid_deps = [
        dep for dep in dependencies
        if dep.get("source", "") in known_rat_names
        and dep.get("target", "") in known_op_names
    ]

    # Keep ALL operations — an op with no rational is still a valid extracted
    # pipeline step; we simply don't know its WHY yet.  Dropping it silently
    # caused 36 ops to be lost in batch 11 (Phase 16 fix).
    # Only drop rationals that have no outgoing edge — they carry no information.
    rats_covered = {dep["source"] for dep in valid_deps}
    operations = [op for op in operations if op.get("name")]
    rationals  = [r  for r  in rationals  if r.get("rational") and r["rational"] in rats_covered]

    # Count how many ops lack any edge (informational warning only)
    ops_covered = {dep["target"] for dep in valid_deps}
    unlinked = [op for op in operations if op["name"] not in ops_covered]
    if unlinked:
        print(f"\n  ℹ  {len(unlinked)} operation(s) have no rational edge (kept):")
        for op in unlinked:
            print(f"       • [{op.get('type','?')}] {op['name']}")

    op_nodes = [
        {"id": op["name"], "node_type": "operation",
         **{k: v for k, v in op.items() if k != "name"}}
        for op in operations
    ]
    rat_nodes = [
        {"id": r["rational"], "node_type": "rational",
         **{k: v for k, v in r.items() if k != "rational"}}
        for r in rationals
    ]

    return {
        "graph": {
            "nodes": op_nodes + rat_nodes,
            "edges": [
                {"id": f"{dep['source']}→{dep['target']}", **dep}
                for dep in valid_deps
            ],
            "metadata": {
                "operation_count": len(operations),
                "rational_count":  len(rationals),
                "node_count":      len(operations) + len(rationals),
                "edge_count":      len(valid_deps),
                "source_labels":   sources,
                "model_used":      model,
                "provider":        provider,
                "extracted_at":    datetime.now().isoformat(),
                "extractor":       "review_paper_pipeline_extractor",
            },
        }
    }


def _strip_graph_ids(graph: dict) -> tuple:
    operations, rationals = [], []
    for n in graph["graph"]["nodes"]:
        node      = {k: v for k, v in n.items() if k != "id"}
        node_type = node.pop("node_type", "operation")
        if node_type == "rational":
            node["rational"] = n["id"]
            rationals.append(node)
        else:
            node["name"] = n["id"]
            operations.append(node)

    edges = [
        {k: v for k, v in e.items() if k != "id"}
        for e in graph["graph"]["edges"]
        if e.get("source") and e.get("target")
    ]
    return operations, rationals, edges


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Review-paper biosignal pipeline extractor — "
                    "provenance-filtered, section-aware, validity-tested.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input ──────────────────────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",  help="Inline text to analyse")
    group.add_argument("--file",  help="Path to a single .txt file")
    group.add_argument("--files", nargs="+", help="Multiple .txt files (merged iteratively)")

    # ── Backend ────────────────────────────────────────────────────────────
    parser.add_argument("--provider", choices=["anthropic", "ollama"], default="ollama")
    parser.add_argument("--model",
                        help=f"Model name (Anthropic default: {DEFAULT_MODEL_ANTHROPIC}, "
                             f"Ollama default: {DEFAULT_MODEL_OLLAMA})")
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL)

    # ── Section-aware filtering ────────────────────────────────────────────
    parser.add_argument("--no-section-filter", action="store_true",
                        help="Disable section detection and filtering "
                             "(process full text as-is)")

    # ── Semantic chunking ──────────────────────────────────────────────────
    parser.add_argument("--chunk", action="store_true",
                        help="Apply semantic chunking within each section "
                             "(or within the full text if --no-section-filter).")
    parser.add_argument("--chunk-model", default=DEFAULT_CHUNK_EMBED_MODEL, metavar="MODEL")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        metavar="PERCENTILE",
                        help=f"SemanticChunker breakpoint percentile [50–99]. "
                             f"Lower = finer chunks. "
                             f"(default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        metavar="N",
                        help=f"Group N chunks into one LLM call to reduce API calls. "
                             f"Use 1 to disable batching. "
                             f"(default: {DEFAULT_BATCH_SIZE})")

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument("--output", default="review_graph.json")
    parser.add_argument("--quiet", action="store_true")

    args    = parser.parse_args()
    verbose = not args.quiet
    out_path = args.output

    model = args.model or (
        DEFAULT_MODEL_ANTHROPIC if args.provider == "anthropic"
        else DEFAULT_MODEL_OLLAMA
    )

    # ── Backend ────────────────────────────────────────────────────────────
    if args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
            sys.exit(1)
        backend = AnthropicBackend(model=model, api_key=api_key)
    else:
        backend = OllamaBackend(model=model, base_url=args.base_url)

    use_section_filter = not args.no_section_filter

    if verbose:
        print(f"Provider        : {args.provider}")
        print(f"Model           : {model}")
        print(f"Mode            : review-paper one-shot")
        print(f"Section filter  : {'ON' if use_section_filter else 'OFF'}")
        print(f"Semantic chunk  : {'ON  | percentile: ' + str(args.chunk_size) + '  | embed: ' + args.chunk_model if args.chunk else 'OFF'}")

    # ── Collect raw texts ──────────────────────────────────────────────────
    def _strip_md_headings(text: str) -> str:
        """Strip Markdown heading markers (##, ###, etc.) so the section
        detector can match its Roman/Arabic prefix patterns."""
        return re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    if args.text:
        raw_texts, raw_labels = [_strip_md_headings(args.text)], ["<inline text>"]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            raw_texts, raw_labels = [_strip_md_headings(f.read())], [args.file]
    else:
        raw_texts, raw_labels = [], []
        for path in args.files:
            with open(path, encoding="utf-8") as f:
                raw_texts.append(_strip_md_headings(f.read()))
            raw_labels.append(path)

    # ── Prepare chunks ──────────────────────────────────────────────────────
    chunks = prepare_chunks(
        raw_texts, raw_labels,
        use_section_filter=use_section_filter,
        chunk=args.chunk,
        embed_model=args.chunk_model,
        percentile=args.chunk_size,
        batch_size=args.batch_size,
        verbose=verbose,
    )

    if not chunks:
        print("No content chunks produced. Check input files or section filter settings.",
              file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"\n  Batch size       : {args.batch_size}")
        print(f"  Chunks to process: {len(chunks)}\n")

    # ── Extraction loop (one chunk / batch at a time) ────────────────────────
    # Intermediate saves go to <out_path>.progress — never to out_path itself.
    # This ensures a failed or interrupted run never destroys a previous good
    # final graph.  out_path is only written once: at successful completion.
    progress_path = out_path + ".progress"

    all_ops:   list = []
    all_rats:  list = []
    all_deps:  list = []
    all_labels: list = []
    batch_stats: list = []          # (label, d_ops, d_rats, d_deps) per batch
    current_graph = None          # None on first iteration → one_shot_extract

    for i, (text, label) in enumerate(chunks):
        if verbose:
            print(f"  [{i+1:2}/{len(chunks)}] {label[:70]}")

        prev_n_ops  = len(all_ops)
        prev_n_rats = len(all_rats)
        prev_n_deps = len(all_deps)

        try:
            all_ops, all_rats, all_deps = process_chunk(
                backend, text, current_graph, verbose
            )
        except Exception as exc:
            print(f"\n  ✗ Batch {i+1} failed: {exc}", file=sys.stderr)
            print(f"    Progress saved to {progress_path}  "
                  f"({len(all_ops)} ops / {len(all_rats)} rats / {len(all_deps)} edges)",
                  file=sys.stderr)
            batch_stats.append((label, 0, 0, 0, "FAILED"))
            break   # exit loop; preserve whatever we have

        all_labels.append(label)

        d_ops  = len(all_ops)  - prev_n_ops
        d_rats = len(all_rats) - prev_n_rats
        d_deps = len(all_deps) - prev_n_deps
        status = "⚠ DEAD" if (d_ops == 0 and d_rats == 0 and d_deps == 0) else "ok"
        batch_stats.append((label, d_ops, d_rats, d_deps, status))

        if verbose:
            print(f"       +{d_ops} ops / +{d_rats} rats / +{d_deps} edges  "
                  f"(totals: {len(all_ops)} / {len(all_rats)} / {len(all_deps)})"
                  + (f"  ← {status}" if status != "ok" else ""))

        # Rebuild intermediate graph dict so next process_chunk can see it
        current_graph = {
            "graph": {
                "nodes": (
                    [{"id": op["name"], "node_type": "operation",
                      **{k: v for k, v in op.items() if k != "name"}}
                     for op in all_ops]
                    + [{"id": r["rational"], "node_type": "rational",
                        **{k: v for k, v in r.items() if k != "rational"}}
                       for r in all_rats]
                ),
                "edges": [
                    {"id": f"{d['source']}→{d['target']}", **d}
                    for d in all_deps
                ],
            }
        }

        # Atomic save to .progress (NOT out_path) after each batch
        _tmp = progress_path + ".tmp"
        with open(_tmp, "w", encoding="utf-8") as f:
            json.dump(current_graph, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(_tmp, progress_path)

    # ── Build final graph (post-processing + metadata) ──────────────────────
    # ── Batch contribution summary ────────────────────────────────────────────
    if verbose and batch_stats:
        print("\n  ┌─ Batch contribution summary " + "─" * 42)
        dead = [s for s in batch_stats if s[4] != "ok"]
        for idx, (lbl, d_o, d_r, d_d, st) in enumerate(batch_stats, 1):
            flag = "  ← ⚠ DEAD" if st == "⚠ DEAD" else ("  ← FAILED" if st == "FAILED" else "")
            print(f"  │ Batch {idx:2}: +{d_o:3} ops  +{d_r:3} rats  +{d_d:3} edges{flag}")
        print(f"  └─ Total: {len(all_ops)} ops / {len(all_rats)} rats / {len(all_deps)} edges"
              + (f"  ({len(dead)} dead batch(es))" if dead else ""))
        if dead:
            print("  ⚠  Dead batches produced nothing -- check if those sections are filtered out"
                  " or if the LLM hit its output limit.", file=sys.stderr)

    try:
        final_graph = build_graph(
            all_ops, all_rats, all_deps,
            all_labels, model, args.provider
        )
    except Exception as exc:
        print(f"\n  ✗ build_graph failed: {exc}", file=sys.stderr)
        print(f"    Raw progress preserved in {progress_path}", file=sys.stderr)
        sys.exit(1)

    # Only now write to out_path
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(final_graph, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)

    # Remove progress file on clean completion
    if os.path.exists(progress_path):
        os.remove(progress_path)

    if verbose:
        n_nodes = len(final_graph["graph"]["nodes"])
        n_edges = len(final_graph["graph"]["edges"])
        n_ops   = final_graph["graph"]["metadata"]["operation_count"]
        n_rats  = final_graph["graph"]["metadata"]["rational_count"]
        print(f"\n  ✓ Graph saved to {out_path}")
        print(f"    {n_ops} operations | {n_rats} rationals | {n_edges} edges | {n_nodes} total nodes")

if __name__ == "__main__":
    main()
