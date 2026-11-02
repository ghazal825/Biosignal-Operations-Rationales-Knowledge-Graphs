#!/usr/bin/env python3
"""
BioSignal Pipeline Graph Extractor  
==========================================
Usage:
  # Anthropic
  export ANTHROPIC_API_KEY="sk-..."
  python biosignal_pipeline_extractor_v2.py --text "..." --provider anthropic

  # Ollama (local)
  python biosignal_pipeline_extractor_v2.py --text "..." --provider ollama --model qwen3.6:35b

  # Multiple files merged iteratively
  python biosignal_pipeline_extractor_v2.py --provider ollama --model qwen3.6:35b \\
      --files paper1.txt paper2.txt --output graph.json

Requirements:
  pip install anthropic openai
  # For Ollama: https://ollama.com  (ollama serve must be running)
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
DEFAULT_MODEL_OLLAMA    = "qwen3.6:35b"
OLLAMA_BASE_URL         = "http://139.30.7.163:11434/v1"

MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY   = 2   # seconds; doubles on each failure

# Label used when an operation has no rational in the text
DEFAULT_RATIONAL_LABEL = "No rationale provided in the text"


# ============================================================
# PROMPTS  (no examples – pure instructions only)
# ============================================================

SYSTEM_PROMPT = """\
You are an expert in biosignal processing and scientific experiment pipeline design.

Your role is to analyse scientific text and extract Biosignal processing
operations and the rationals behind applying them.

You will receive the text and will be prompted in up to 3 stages:

Pre-step. Text-Evidence Extraction: Scan the full text and return verbatim
  passages that are relevant to processing operations, their rationales, and
  the connections between them.  These passages will be used in the next stage.

Stage 1 – Simultaneous Node + Edge Extraction: Using the pre-extracted text
  evidences, extract in a SINGLE response:
    • All processing operations (blue nodes)
    • All rationals / reasons (green nodes)
    • All causalities that connect a rational to an operation (directed edges)

Stage 2 – Merge Variants: When given a second or third text source, merge the
  newly extracted operations, rationals, and causalities with the existing
  ones, preserving all previously captured information.

Stage 3 – Validation: Check for name mismatches between the operations and
  causalities, and fix them by renaming or deleting inconsistent entries.

Principles:
- Extract ONLY operations and connections that are explicitly mentioned or
  strongly implied in the text.
- Use the exact terminology as it appears in the source text.
- Respond with properly formatted JSON only.
- Never wrap your response in markdown code blocks or add any other text.\
"""

# ── Pre-step – Text evidence extraction ─────────────────────────────────────
TEXT_EVIDENCE_PROMPT = """\
Scan the scientific text below and extract ALL verbatim passages that are
relevant to building a biosignal processing pipeline graph.  Focus on:

1. Operation evidences  – any passage that names, describes, or parametrises
   a biosignal processing step (filtering, normalisation, segmentation,
   feature extraction, classification, statistical tests, plotting, etc.).

2. Rationale evidences  – any passage that gives a reason, motivation, or
   justification for applying (or avoiding) a processing operation.

3. Connection evidences  – any passage that explicitly or implicitly links a
   reason/condition to a specific processing operation, i.e. shows that the
   reason CAUSES or justifies the use of the operation.

Text:
\"\"\"
${text}
\"\"\"

CRITICAL: Respond ONLY with the raw JSON object below. No markdown, no extra text.

{
  "operation_evidences": [
    {
      "passage": "verbatim quote from the text describing the operation",
      "relevance": "which operation / aspect this passage covers"
    }
  ],
  "rationale_evidences": [
    {
      "passage": "verbatim quote from the text describing a reason for applying an operation",
      "relevance": "which rationale or motivation this passage describes"
    }
  ],
  "connection_evidences": [
    {
      "passage": "verbatim quote that links a reason to an operation",
      "relevance": "which rational→operation connection this passage supports"
    }
  ]
}\
"""

# ── Stage 1 – Simultaneous node + edge extraction ────────────────────────────
COMBINED_EXTRACTION_PROMPT = """\
Using the PRE-EXTRACTED TEXT EVIDENCES and the ORIGINAL TEXT below, extract
in a SINGLE response:
  • All discrete biosignal processing operations (blue nodes)
  • All rationals / reasons behind applying them (green nodes)
  • All causalities linking a rational to an operation (directed edges)

Original Text:
\"\"\"
${text}
\"\"\"

PRE-EXTRACTED TEXT EVIDENCES:
${text_evidences}

OPERATIONS (blue nodes)
  – Every single transformation applied to the signal or data derived from it,
    including upstream processing (data curation, formatting, grouping,
    selection, segmentation, normalisation, pre-processing, denoising, quality
    control, etc.) and downstream processing (exact method names, feature
    extraction, statistics, learning, aggregation, reporting, plotting, etc.).
  – In "parameters" include ALL details that configure and uniquely identify
    the operation.
  – If the text describes a compound procedure or multi-step method as a single
    described step, include all sub-steps in the name — do not truncate at the
    first recognisable method name.
  – Fill in "type", "signal_type", and "parameters" only when the text
    provides that information; otherwise use empty strings / objects.

Operation type definitions (for the "type" field of operations):
  – "preprocessing": Operations that prepare or clean the raw signal before
    analysis, including data formatting, quality control, denoising, and
    artefact removal.
  – "filtering": Operations that attenuate or isolate specific frequency
    components or signal characteristics.
  – "segmentation": Operations that divide the signal into windows, epochs,
    or segments.
  – "normalization": Operations that rescale or standardise the signal or
    derived features.
  – "feature extraction": Operations that compute descriptors, statistics, or
    representations from the signal or its segments.
  – "dimensionality reduction": Operations that reduce the number of variables
    or components, including decomposition and selection methods.
  – "classification / regression": Operations that assign labels or predict
    values from features.
  – "evaluation": Operations that measure model or pipeline performance.
  – "other": Any operation that does not fit the categories above.

RATIONALS (green nodes)
  – Reasons, motivations, or justifications for applying (or avoiding) an
    operation, as expressed in the text.
  – The "rational" field must be an exact or near-verbatim phrase from the
    text — do not paraphrase or shorten.
  – Each rational must have its own entry even if it applies to multiple
    operations.
  – Pay particular attention to purpose clauses and goal statements that are
    embedded inside operation descriptions, such as "for the removal of X",
    "to suppress Y", "in order to Z", "to improve W". These must be extracted
    as separate rational nodes even when they appear inline as part of an
    operation sentence.

Rational type definitions (for the "type" field of rationals):
  – "application-driven": The reason is based on the specific application
    domain, clinical context, or end-use goal.
  – "data-driven": The reason is based on observed or reported properties of
    the data, such as noise characteristics, signal quality, or empirical
    findings.
  – "axiomatic": The reason is based on a well-known or universally accepted
    property of the signal or method, stated as an established fact without
    reference to specific empirical evidence.

CAUSALITIES (edges)
  – Directed edges: rational → operation.
  – "source" must be the EXACT rational string.
  – "target" must be the EXACT operation name.
  – Include only causalities supported by the text.
  – For each operation, explicitly check whether any rational provides a
    reason to apply it — including implicit links where the rational's goal
    is clearly achieved by the operation. Do not leave an operation without
    a causality if a relevant rational can be identified.

Edge type definitions (for the "type of an edge" field):
  Positive edges (polarity = "positive") — rational supports applying the operation:
  – "can apply": The rational indicates the operation is a valid option in
    the described context, but not strictly required.
  – "should apply": The rational provides a strong recommendation or
    best-practice justification for applying the operation.
  – "must apply": The rational indicates a strict requirement — omitting the
    operation would violate a fundamental constraint.
  Negative edges (polarity = "negative") — rational argues against applying the operation:
  – "cannot apply": The rational indicates the operation is not a valid
    option in the described context.
  – "should not apply": The rational provides a strong recommendation against
    applying the operation.
  – "must not apply": The rational indicates the operation is strictly
    forbidden — applying it would violate a fundamental constraint.

Requirements:
  – Extract ONLY items explicitly mentioned or strongly implied in the text.
  – Use the exact terminology from the source text.
  – Causality sources and targets must exactly match the names in the
    operations and rationals lists you produce in the SAME response.

CRITICAL: Respond ONLY with the raw JSON object below. No markdown, no extra text.

{
  "operations": [
    {
      "name": "exact operation name from text",
      "type": "preprocessing | filtering | segmentation | normalization | feature extraction | dimensionality reduction | classification / regression | evaluation | other",
      "parameters": {
        "parameter_name": "parameter value from text",
        "another_parameter": "another value"
      },
      "text_evidence": "relevant quote from the text"
    }
  ],
  "rationals": [
    {
      "rational": "exact or near-verbatim phrase from the text expressing the reason — do not paraphrase or shorten",
      "type": "application-driven | data-driven | axiomatic",
      "text_evidence": "relevant quote from the text"
    }
  ],
  "causalities": [
    {
      "source": "EXACT rational string from the rationals list above",
      "target": "EXACT operation name from the operations list above",
      "polarity": "positive | negative",
      "type of an edge": "can apply | should apply | must apply | cannot apply | should not apply | must not apply",
      "text_evidence": "exact quote supporting this connection"
    }
  ]
}\
"""

# ── Merge – Combined node + edge merge ───────────────────────────────────────
COMBINED_MERGE_PROMPT = """\
PRESERVATION REQUIREMENTS:
  ALL ${operation_count} EXISTING OPERATIONS must be kept.
  ALL ${rational_count}  EXISTING RATIONALS  must be kept.
  ALL ${causality_count} EXISTING CAUSALITIES must be kept.

EXISTING OPERATIONS (${operation_count} total):
${existing_operations}

EXISTING RATIONALS (${rational_count} total):
${existing_rationals}

EXISTING CAUSALITIES (${causality_count} total):
${existing_causalities}

CRITICAL MERGE STRATEGY:
  STEP 1 – Use ALL existing operations, rationals, and causalities as your
            foundation.  Do not remove any.
  STEP 2 – Read the new text evidences and original text for items not yet
            captured.
  STEP 3 – For each candidate operation:
      MERGE  – if it refers to the same processing step as an existing one —
               whether under a different name, abbreviation, or equivalent
               description (operations that accomplish the same goal with the
               same method should be merged even if their wording differs
               substantially) — merge them into one entry; keep the most
               precise name, combine text_evidence with " | ADDITIONAL: "
               separator, and merge the parameter dicts.
      ADD    – if it is a genuinely different processing step, add it as a
               new entry.
  STEP 4 – For each candidate rational:
      MERGE  – if it expresses the same reason as an existing rational —
               whether worded differently or at a different level of
               specificity — merge text_evidence.
      ADD    – if genuinely new, add it.
  STEP 5 – For each candidate causality:
      UPDATE – if the same source→target pair already exists, add new
               text_evidence.
      ADD    – if it is a genuinely new source→target connection, add it.

FINAL OUTPUT must contain:
  – At minimum ${operation_count} operations (all existing, preserved or enhanced)
  – At minimum ${rational_count} rationals (all existing, preserved or enhanced)
  – At minimum ${causality_count} causalities (all existing, preserved)
  – Any genuinely new items from the new text

"""

# ── Validation – fix name mismatches ────────────────────────────────────────
VALIDATION_PROMPT = """\
Fix name-matching issues in the causality list.
- Each "source" must exactly match a name from the AVAILABLE RATIONALS list.
- Each "target" must exactly match a name from the AVAILABLE OPERATIONS list.

AVAILABLE RATIONALS:
${rationals_list}

AVAILABLE OPERATIONS:
${operation_list}

MISMATCHED ENDPOINTS (appear in causalities but do not match either list):
${validation_issues}

For each mismatched endpoint:
  1. RENAME it to the closest matching name from the appropriate list
     (rationals list for sources, operations list for targets).
  2. Specify DELETION if no reasonable match exists.

CRITICAL: Respond ONLY with the raw JSON object below. No markdown, no extra text.

{
  "renamings": [
    { "old_name": "mismatched name", "new_name": "exact name from the appropriate list" }
  ],
  "deletions": [
    "mismatched name whose causality should be removed"
  ]
}\
"""


# ============================================================
# BACKEND ABSTRACTION
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
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text


class OllamaBackend:
    """
    Uses the OpenAI-compatible REST endpoint that Ollama exposes at
    http://localhost:11434/v1  (no API key required).
    """
    def __init__(self, model: str, base_url: str):
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: openai package not found.  Run: pip install openai")
            sys.exit(1)
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.model  = model

    def chat(self, system: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": user_message},
            ],
        )
        return response.choices[0].message.content


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
    Strips markdown code-fence wrappers if the model added them.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
    # Qwen thinking tags  <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
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


def _save_debug(debug_dir: str, filename: str, data) -> None:
    """Write a stage's output to a JSON file inside debug_dir."""
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"    [debug] saved → {path}")


def _normalize_dep(dep: dict) -> dict:
    """
    Normalise causality field names that the LLM may output with
    the prompt's typos ('poslarity', 'type of an edge') to the
    canonical names used by the graph builder and visualiser.
    """
    d = dict(dep)
    # polarity
    if "polarity" not in d and "poslarity" in d:
        d["polarity"] = d.pop("poslarity")
    # edge_type
    if "edge_type" not in d and "type of an edge" in d:
        d["edge_type"] = d.pop("type of an edge")
    return d


def _format_text_evidences(evidences: dict) -> str:
    """
    Serialise the text-evidence dict as a compact, readable block
    that can be dropped into the combined extraction prompt.
    """
    lines = []
    for category, items in evidences.items():
        label = category.replace("_", " ").upper()
        lines.append(f"[{label}]")
        for item in items:
            passage   = item.get("passage",   "").strip()
            relevance = item.get("relevance", "").strip()
            lines.append(f'  • "{passage}"')
            if relevance:
                lines.append(f'    → {relevance}')
        lines.append("")
    return "\n".join(lines)


# ============================================================
# PIPELINE STAGES
# ============================================================

def _op_list(operations: list) -> str:
    return "\n".join(f'{i+1}. "{op["name"]}"' for i, op in enumerate(operations))


def _rat_list(rationals: list) -> str:
    return "\n".join(f'{i+1}. "{r["rational"]}"' for i, r in enumerate(rationals))


# ── Pre-step: Text evidence extraction ──────────────────────────────────────

def extract_text_evidences(backend, text: str, verbose: bool) -> dict:
    """
    Stage 0 – scan the raw text and return categorised verbatim passages.
    The returned dict has keys:
      "operation_evidences", "rationale_evidences", "connection_evidences"
    each containing a list of {"passage": ..., "relevance": ...} dicts.
    """
    if verbose:
        print("\n  [Pre-step] Extracting text evidences …")
    prompt = fill_template(TEXT_EVIDENCE_PROMPT, {"text": text})
    raw    = call_llm(backend, SYSTEM_PROMPT, prompt, "text-evidence", verbose)
    data   = parse_llm_json(raw, "text-evidence")

    n_op   = len(data.get("operation_evidences",  []))
    n_rat  = len(data.get("rationale_evidences",  []))
    n_con  = len(data.get("connection_evidences", []))
    if verbose:
        print(f"    → {n_op} operation, {n_rat} rationale, "
              f"{n_con} connection evidence(s) retrieved")
    return data


# ── Stage 1: Simultaneous node + edge extraction ─────────────────────────────

def extract_nodes_and_edges(backend, text: str,
                            text_evidences: dict, verbose: bool) -> tuple:
    """
    Stage 1 – extract operations, rationals, AND causalities in one call.
    Returns (operations, rationals, dependencies).
    """
    if verbose:
        print("\n  [Stage 1] Simultaneous extraction of nodes and edges …")
    prompt = fill_template(COMBINED_EXTRACTION_PROMPT, {
        "text":           text,
        "text_evidences": _format_text_evidences(text_evidences),
    })
    raw  = call_llm(backend, SYSTEM_PROMPT, prompt, "combined-extraction", verbose)
    data = parse_llm_json(raw, "combined-extraction")

    ops  = data.get("operations",  [])
    rats = data.get("rationals",   [])
    deps = [_normalize_dep(d) for d in data.get("causalities", [])]

    if not ops:
        raise ValueError("No operations extracted – check the input text.")
    if verbose:
        print(f"    → {len(ops)} operation(s), {len(rats)} rational(s), "
              f"{len(deps)} causality(ies) extracted")
    return ops, rats, deps


# ── Stage 2: Combined merge ──────────────────────────────────────────────────

def merge_nodes_and_edges(backend, text: str, text_evidences: dict,
                          existing_ops: list, existing_rats: list,
                          existing_deps: list, verbose: bool) -> tuple:
    """
    Stage 2 – merge new text into the existing graph in a single call.
    Returns (operations, rationals, dependencies).
    """
    if verbose:
        print("\n  [Merge] Merging nodes and edges …")

    merge_header = fill_template(COMBINED_MERGE_PROMPT, {
        "operation_count":    len(existing_ops),
        "rational_count":     len(existing_rats),
        "causality_count":    len(existing_deps),
        "existing_operations":  json.dumps(existing_ops,  indent=2),
        "existing_rationals":   json.dumps(existing_rats,  indent=2),
        "existing_causalities": json.dumps(existing_deps, indent=2),
    })
    base   = fill_template(COMBINED_EXTRACTION_PROMPT, {
        "text":           text,
        "text_evidences": _format_text_evidences(text_evidences),
    })
    prompt = merge_header + "\n\n" + base
    raw    = call_llm(backend, SYSTEM_PROMPT, prompt, "combined-merge", verbose)
    data   = parse_llm_json(raw, "combined-merge")

    ops  = data.get("operations",  [])
    rats = data.get("rationals",   [])
    deps = [_normalize_dep(d) for d in data.get("causalities", [])]

    if len(ops) < len(existing_ops):
        raise ValueError(
            f"Merge shrank operation list ({len(existing_ops)} → {len(ops)})."
        )
    if len(rats) < len(existing_rats):
        raise ValueError(
            f"Merge shrank rational list ({len(existing_rats)} → {len(rats)})."
        )
    if verbose:
        print(f"    → ops: {len(existing_ops)} → {len(ops)}, "
              f"rats: {len(existing_rats)} → {len(rats)}, "
              f"causalities: {len(existing_deps)} → {len(deps)}")
    return ops, rats, deps


# ── Stage 3: Validation ──────────────────────────────────────────────────────

def validate_graph(backend, operations: list, rationals: list,
                   dependencies: list, verbose: bool) -> tuple:
    """
    Check that every causality source matches a rational name and every
    target matches an operation name.  Calls the LLM to fix mismatches.
    Returns (operations, rationals, fixed_dependencies).
    """
    if verbose:
        print("\n  [Validation] Checking name consistency …")

    op_names  = {op["name"]    for op in operations}
    rat_names = {r["rational"] for r in rationals}
    mismatches = set()
    for dep in dependencies:
        src = dep.get("source", "")
        tgt = dep.get("target", "")
        if src and src not in rat_names:
            mismatches.add(src)
        if tgt and tgt not in op_names:
            mismatches.add(tgt)

    if not mismatches:
        if verbose:
            print("    → No mismatches – graph is consistent.")
        return operations, rationals, dependencies

    if verbose:
        print(f"    → {len(mismatches)} mismatch(es): {sorted(mismatches)}")

    rat_list_str = "\n".join(
        f'{i+1}. "{n}"' for i, n in enumerate(sorted(rat_names))
    )
    op_list_str = "\n".join(
        f'{i+1}. "{n}"' for i, n in enumerate(sorted(op_names))
    )
    issues_str = "\n".join(f'- "{n}"' for n in sorted(mismatches))
    prompt = fill_template(VALIDATION_PROMPT, {
        "rationals_list":    rat_list_str,
        "operation_list":    op_list_str,
        "validation_issues": issues_str,
    })

    try:
        raw   = call_llm(backend, SYSTEM_PROMPT, prompt, "validation", verbose)
        fixes = parse_llm_json(raw, "validation")
    except Exception as exc:
        print(f"    WARNING: validation call failed ({exc}). Skipping.")
        return operations, rationals, dependencies

    renamings = {r["old_name"]: r["new_name"] for r in fixes.get("renamings", [])}
    deletions = set(fixes.get("deletions", []))

    fixed_deps = []
    for dep in dependencies:
        src = dep.get("source", "")
        tgt = dep.get("target", "")
        if src in deletions or tgt in deletions:
            continue
        dep = {**dep,
               "source": renamings.get(src, src),
               "target": renamings.get(tgt, tgt)}
        fixed_deps.append(dep)

    if verbose:
        removed = len(dependencies) - len(fixed_deps)
        print(f"    → {len(renamings)} rename(s), {removed} causality(ies) removed")
    return operations, rationals, fixed_deps


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_graph(operations: list, rationals: list, dependencies: list,
                sources: list, model: str, provider: str) -> dict:
    """
    Build the final graph dict.

    Operations with no incoming rational edge are dropped.
    Rationals with no outgoing edge to an operation are dropped.
    Only nodes that participate in at least one causality are kept.
    """
    # ── Drop operations with no incoming rational edge ───────────────────────
    ops_covered  = {dep["target"] for dep in dependencies}
    operations   = [op for op in operations if op["name"] in ops_covered]

    # ── Drop rationals with no outgoing edge to an operation ─────────────────
    rats_covered = {dep["source"] for dep in dependencies}
    rationals    = [r for r in rationals if r["rational"] in rats_covered]

    # ── Build node lists ─────────────────────────────────────────────────────
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
                for dep in dependencies
            ],
            "metadata": {
                "operation_count": len(operations),
                "rational_count":  len(rationals),
                "node_count":      len(operations) + len(rationals),
                "edge_count":      len(dependencies),
                "source_labels":   sources,
                "model_used":      model,
                "provider":        provider,
                "extracted_at":    datetime.now().isoformat(),
            },
        }
    }


def _strip_graph_ids(graph: dict) -> tuple:
    """
    Remove synthetic 'id' and 'node_type' keys before passing nodes/edges
    back to the LLM.  Returns (operations, rationals, edges).
    """
    operations, rationals = [], []
    for n in graph["graph"]["nodes"]:
        node      = {k: v for k, v in n.items() if k != "id"}
        node_type = node.pop("node_type", "operation")
        if node_type == "rational":
            # Skip the auto-generated fallback rational – it's not real text evidence
            if n["id"] == DEFAULT_RATIONAL_LABEL:
                continue
            node["rational"] = n["id"]
            rationals.append(node)
        else:
            node["name"] = n["id"]
            operations.append(node)

    edges = [
        {k: v for k, v in e.items() if k != "id"}
        for e in graph["graph"]["edges"]
        # Also strip edges that come from the fallback rational
        if e.get("source") != DEFAULT_RATIONAL_LABEL
    ]
    return operations, rationals, edges


def process_text(backend, text: str, existing_graph: dict | None,
                 verbose: bool, debug_dir: str | None = None,
                 source_idx: int = 1) -> tuple:
    """
    Run the full pipeline for a single text source.

    Flow:
      Pre-step → extract_text_evidences
      If first source  → extract_nodes_and_edges (Stage 1)
      If merge         → merge_nodes_and_edges   (Stage 2)
      Always           → validate_graph          (Stage 3)

    Returns (operations, rationals, dependencies).
    """
    # Pre-step: always extract text evidences first
    text_evidences = extract_text_evidences(backend, text, verbose)
    if debug_dir:
        _save_debug(debug_dir, f"source{source_idx}_prestep_evidences.json", text_evidences)

    if existing_graph is None:
        ops, rats, deps = extract_nodes_and_edges(
            backend, text, text_evidences, verbose
        )
        if debug_dir:
            _save_debug(debug_dir, f"source{source_idx}_stage1_operations.json",  ops)
            _save_debug(debug_dir, f"source{source_idx}_stage1_rationals.json",   rats)
            _save_debug(debug_dir, f"source{source_idx}_stage1_causalities.json", deps)
    else:
        ex_ops, ex_rats, ex_deps = _strip_graph_ids(existing_graph)
        ops, rats, deps = merge_nodes_and_edges(
            backend, text, text_evidences, ex_ops, ex_rats, ex_deps, verbose
        )
        if debug_dir:
            _save_debug(debug_dir, f"source{source_idx}_merge_operations.json",  ops)
            _save_debug(debug_dir, f"source{source_idx}_merge_rationals.json",   rats)
            _save_debug(debug_dir, f"source{source_idx}_merge_causalities.json", deps)

    ops, rats, deps = validate_graph(backend, ops, rats, deps, verbose)
    if debug_dir:
        _save_debug(debug_dir, f"source{source_idx}_validation_result.json", {
            "operations":  ops,
            "rationals":   rats,
            "causalities": deps,
        })
    return ops, rats, deps


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract a biosignal processing pipeline graph from scientific text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input ──────────────────────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text",  help="Inline text to analyse")
    group.add_argument("--file",  help="Path to a single .txt file")
    group.add_argument("--files", nargs="+",
                       help="Paths to multiple .txt files (merged iteratively)")

    # ── Backend ────────────────────────────────────────────────────────────
    parser.add_argument("--provider", choices=["anthropic", "ollama"],
                        default="ollama",
                        help="LLM backend (default: ollama)")
    parser.add_argument("--model",
                        help=(f"Model name.  "
                              f"Anthropic default: {DEFAULT_MODEL_ANTHROPIC}  "
                              f"Ollama default: {DEFAULT_MODEL_OLLAMA}"))
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL,
                        help=f"Ollama server URL (default: {OLLAMA_BASE_URL})")

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument("--output", default="pipeline_graph.json",
                        help="Output JSON file (default: pipeline_graph.json)")
    parser.add_argument("--debug-dir", default=None, metavar="DIR",
                        help="If set, write each stage's JSON output to this folder")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress messages")

    args    = parser.parse_args()
    verbose = not args.quiet

    # ── Resolve model name ─────────────────────────────────────────────────
    model = args.model or (
        DEFAULT_MODEL_ANTHROPIC if args.provider == "anthropic"
        else DEFAULT_MODEL_OLLAMA
    )

    # ── Build backend ──────────────────────────────────────────────────────
    if args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
            sys.exit(1)
        backend = AnthropicBackend(model=model, api_key=api_key)
    else:
        backend = OllamaBackend(model=model, base_url=args.base_url)

    if verbose:
        print(f"Provider : {args.provider}")
        print(f"Model    : {model}")

    # ── Collect input texts ────────────────────────────────────────────────
    if args.text:
        texts, labels = [args.text], ["<inline text>"]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            texts, labels = [f.read()], [args.file]
    else:
        texts, labels = [], []
        for path in args.files:
            with open(path, encoding="utf-8") as f:
                texts.append(f.read())
            labels.append(path)

    # ── Run pipeline ───────────────────────────────────────────────────────
    current_graph = None
    for idx, (text, label) in enumerate(zip(texts, labels)):
        if verbose:
            bar = "=" * 60
            print(f"\n{bar}\n  Source {idx+1}/{len(texts)}: {label}\n{bar}")
        ops, rats, deps = process_text(backend, text, current_graph, verbose,
                                        debug_dir=args.debug_dir,
                                        source_idx=idx + 1)
        current_graph   = build_graph(ops, rats, deps, labels[:idx+1], model, args.provider)
        if args.debug_dir:
            _save_debug(args.debug_dir, f"source{idx+1}_final_graph.json", current_graph)

    # ── Write output ───────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(current_graph, f, indent=2, ensure_ascii=False)

    if verbose:
        m = current_graph["graph"]["metadata"]
        print(f"\n{'='*60}")
        print(f"  Done!  Graph written to: {args.output}")
        print(f"  Nodes (operations) : {m['operation_count']}")
        print(f"  Nodes (rationals)  : {m['rational_count']}")
        print(f"  Edges (causalities): {m['edge_count']}")
        print(f"{'='*60}\n")

    print(json.dumps(current_graph, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
