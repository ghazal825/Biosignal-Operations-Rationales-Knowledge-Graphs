#!/usr/bin/env python3
"""
BioSignal Pipeline Graph Extractor – One-Shot Edition
=======================================================
Usage:
  # Anthropic
  export ANTHROPIC_API_KEY="sk-..."
  python one_shot_pipeline_extractor.py --text "..." --provider anthropic

  # Ollama (local)
  python one_shot_pipeline_extractor.py --text "..." --provider ollama --model qwen3.6:35b

  # Multiple files merged iteratively
  python one_shot_pipeline_extractor.py --provider ollama --model qwen3.6:35b \\
      --files paper1.txt paper2.txt --output graph.json

  # Semantic chunking of a single long paper
  python one_shot_pipeline_extractor.py --provider ollama --model qwen3.6:35b \\
      --file paper.txt --chunk --output graph.json

  # Semantic chunking with a custom embedding model and target chunk size
  python one_shot_pipeline_extractor.py --provider ollama --model qwen3.6:35b \\
      --file paper.txt --chunk --chunk-model all-mpnet-base-v2 --chunk-size 500

Requirements:
  pip install anthropic openai
  pip install langchain-text-splitters langchain-community sentence-transformers
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

# Semantic chunking defaults
DEFAULT_CHUNK_EMBED_MODEL = "all-MiniLM-L6-v2"  # fast, good quality
DEFAULT_CHUNK_SIZE        = 300                  # approximate sentences per chunk boundary


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT = """\
You are an expert in biosignal processing and scientific experiment pipeline design.

Your role is to analyse scientific text and extract biosignal processing operations,
the rationals (reasons) behind applying them, and the causal connections between
rationals and operations — all in a single response.

Principles:
- Extract ONLY operations and connections that are explicitly mentioned or
  strongly implied in the text.
- Use the exact terminology as it appears in the source text.
- Every causality "source" MUST exactly match a rational string in the rationals list.
- Every causality "target" MUST exactly match an operation name in the operations list.
- Respond with properly formatted JSON only.
- Never wrap your response in markdown code blocks or add any other text.\
"""


# ── One-shot extraction (first / only text source) ─────────────────────────
ONE_SHOT_PROMPT = """\
Read the scientific text below and produce a complete pipeline graph in one step.

Your output must contain THREE sections merged into one JSON object:

1. OPERATIONS
   Extract every discrete biosignal processing step mentioned, from upstream
   (data curation, formatting, grouping, selection, segmentation, normalisation,
   pre-processing, denoising, quality control, etc.) to downstream
   (exact method names with parameters, feature extraction, statistics, learning,
   aggregation, reporting, plotting, etc.).
   In "parameters" include ALL details that configure the operation and make it unique.

2. RATIONALS
   For each operation, extract the specific reason stated or implied in the text for
   why that operation is applied (signal quality goals, noise/artifact sources,
   clinical/diagnostic goals, computational constraints, signal properties,
   dataset consistency requirements, etc.).
   Extract one rational per distinct reason.

3. CAUSALITIES
   For each rational, go through EVERY operation and determine whether that rational
   justifies applying that operation (based on the text).
   A rational may justify multiple operations.
   Produce one entry per valid (rational → operation) pair.

SELF-VALIDATION (apply before writing your final JSON):
   • Every causality "source" must be the exact "rational" string of an entry in your
     rationals list.
   • Every causality "target" must be the exact "name" string of an entry in your
     operations list.
   • Remove any causality whose source or target does not match exactly.
   • Operations with no incoming causality edge are kept in the list but will be
     filtered at build time.

Requirements:
- Use exact names/wording as they appear in the text.
- Fill in "type", "signal_type", and "parameters" only when the text provides that
  information; otherwise use empty strings or empty objects.
- For causality "polarity" and "edge_type": determine whether a rational says that,
  in the described circumstances, an operation can, cannot, should, should not, must
  or must not be applied.

Text:
\"\"\"
${text}
\"\"\"

CRITICAL: Respond ONLY with the raw JSON object below. No markdown, no extra text.

{
  "operations": [
    {
      "name": "exact operation name from text",
      "type": "label the signal processing method into a category",
      "parameters": {
        "parameter_name": "parameter value from text"
      },
      "text_evidence": "relevant quote from the text"
    }
  ],
  "rationals": [
    {
      "rational": "concise statement of WHY a processing step is applied",
      "type": "application-driven | data-driven | axiomatic",
      "text_evidence": "relevant quote from the text"
    }
  ],
  "causalities": [
    {
      "source": "EXACT rational string from the rationals list above",
      "target": "EXACT operation name from the operations list above",
      "polarity": "positive | negative",
      "edge_type": "can apply | should apply | must apply",
      "text_evidence": "exact quote supporting this connection"
    }
  ]
}\
"""


# ── One-shot merge (subsequent text sources) ────────────────────────────────
ONE_SHOT_MERGE_PROMPT = """\
You already have an extracted graph (operations, rationals, causalities).
Now extend it with information from a NEW text source, and produce the complete
updated graph in a single response.

PRESERVATION REQUIREMENT:
  ALL ${operation_count} existing operations and ALL ${causality_count} existing
  causalities MUST be kept in the final output.

EXISTING OPERATIONS (${operation_count} total – keep all):
${existing_operations}

EXISTING RATIONALS (${rational_count} total – keep all):
${existing_rationals}

EXISTING CAUSALITIES (${causality_count} total – keep all):
${existing_causalities}

MERGE STRATEGY:
  STEP 1 – Use ALL existing entries as your foundation. Do not remove any.
  STEP 2 – Read the new text for additional information not yet captured.
  STEP 3 – For each candidate operation/rational/causality decide:
    MERGE: if it refers to the same item as an existing entry under a different
           name or abbreviation → merge into one entry (keep the most precise name,
           combine text_evidence with " | ADDITIONAL: " separator, merge parameters).
    ADD:   if it is genuinely new → add as a new entry.
    UPDATE: for causalities, update text_evidence if the same source→target pair
            appears with new supporting evidence.

SELF-VALIDATION (apply before writing your final JSON):
  • Every causality "source" must exactly match a "rational" string in your
    final rationals list.
  • Every causality "target" must exactly match an "operation" name in your
    final operations list.
  • Remove any causality whose source or target does not match exactly.

FINAL OUTPUT must contain:
  - All ${operation_count} existing operations (preserved or enhanced) + any new ones
  - All ${rational_count} existing rationals (preserved or enhanced) + any new ones
  - All ${causality_count} existing causalities (preserved or updated) + any new ones

NEW TEXT:
\"\"\"
${text}
\"\"\"

CRITICAL: Respond ONLY with the raw JSON object below. No markdown, no extra text.

{
  "operations": [
    {
      "name": "exact operation name",
      "type": "category label",
      "parameters": {
        "parameter_name": "value"
      },
      "text_evidence": "quote"
    }
  ],
  "rationals": [
    {
      "rational": "concise reason statement",
      "type": "application-driven | data-driven | axiomatic",
      "text_evidence": "quote"
    }
  ],
  "causalities": [
    {
      "source": "EXACT rational string",
      "target": "EXACT operation name",
      "polarity": "positive | negative",
      "edge_type": "can apply | should apply | must apply",
      "text_evidence": "supporting quote"
    }
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
            max_tokens=8192,
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
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.model  = model

    def chat(self, system: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
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
        # Try json_repair as a fallback (pip install json-repair)
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
# SEMANTIC CHUNKING
# ============================================================

def chunk_text_semantically(text: str, embed_model: str, chunk_size: int,
                             verbose: bool) -> list[str]:
    """
    Split *text* into semantically coherent chunks using LangChain's
    SemanticChunker backed by a sentence-transformers embedding model.

    Parameters
    ----------
    text        : full document text
    embed_model : HuggingFace sentence-transformers model name
                  (e.g. "all-MiniLM-L6-v2", "all-mpnet-base-v2")
    chunk_size  : controls the percentile breakpoint threshold passed to
                  SemanticChunker (higher → fewer, larger chunks)
    verbose     : print progress

    Returns
    -------
    List of text chunks (strings), in document order.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        # Fallback import paths for older langchain versions
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

    if verbose:
        print(f"\n  [Chunking] Loading embedding model '{embed_model}' …")

    embeddings = HuggingFaceEmbeddings(model_name=embed_model)

    # breakpoint_threshold_type="percentile" splits where the cosine distance
    # between consecutive sentence embeddings exceeds the given percentile.
    # chunk_size maps to the percentile (clamped 50–99).
    percentile = max(50, min(99, chunk_size))
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile,
    )

    chunks = chunker.split_text(text)

    if verbose:
        print(f"  [Chunking] Split into {len(chunks)} semantic chunks "
              f"(percentile threshold: {percentile})")
        for i, c in enumerate(chunks):
            word_count = len(c.split())
            print(f"    Chunk {i+1:>3}: {word_count:>5} words")

    return chunks


# ============================================================
# ONE-SHOT PIPELINE STAGE
# ============================================================

def one_shot_extract(backend, text: str, verbose: bool) -> tuple:
    """
    First source: extract operations, rationals, and causalities in one LLM call.
    Returns (operations, rationals, causalities).
    """
    if verbose:
        print("\n  [One-Shot] Extracting operations, rationals, and causalities …")

    prompt = fill_template(ONE_SHOT_PROMPT, {"text": text})
    try:
        raw  = call_llm(backend, SYSTEM_PROMPT, prompt, "one-shot-extract", verbose)
        data = parse_llm_json(raw, "one-shot-extract")
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
        print(f"    → {len(ops)} operations, {len(rats)} rationals, {len(deps)} causalities")
        print("\n  --- Operations ---")
        print(json.dumps(ops, indent=2))
        print("\n  --- Rationals ---")
        print(json.dumps(rats, indent=2))
        print("\n  --- Causalities ---")
        print(json.dumps(deps, indent=2))

    return ops, rats, deps


def one_shot_merge(backend, text: str, existing_ops: list, existing_rats: list,
                   existing_deps: list, verbose: bool) -> tuple:
    """
    Subsequent sources: merge the new text into the existing graph in one LLM call.
    Returns (operations, rationals, causalities).
    """
    if verbose:
        print("\n  [One-Shot Merge] Merging new source into existing graph …")

    prompt = fill_template(ONE_SHOT_MERGE_PROMPT, {
        "operation_count":   len(existing_ops),
        "rational_count":    len(existing_rats),
        "causality_count":   len(existing_deps),
        "existing_operations": json.dumps(existing_ops, indent=2),
        "existing_rationals":  json.dumps(existing_rats, indent=2),
        "existing_causalities": json.dumps(existing_deps, indent=2),
        "text": text,
    })
    try:
        raw  = call_llm(backend, SYSTEM_PROMPT, prompt, "one-shot-merge", verbose)
        data = parse_llm_json(raw, "one-shot-merge")
    except (RuntimeError, ValueError) as exc:
        if verbose:
            print(f"    WARNING: merge failed ({exc.__class__.__name__}) – keeping existing graph.")
        return existing_ops, existing_rats, existing_deps

    ops  = data.get("operations",  existing_ops)
    rats = data.get("rationals",   existing_rats)
    deps = data.get("causalities", existing_deps)

    # Guard against the model returning a shrunken list — fall back to existing
    if len(ops) < len(existing_ops):
        if verbose:
            print(f"    WARNING: merge shrank ops ({len(existing_ops)} → {len(ops)}) – keeping existing.")
        ops = existing_ops
    if len(deps) < len(existing_deps):
        if verbose:
            print(f"    WARNING: merge shrank deps ({len(existing_deps)} → {len(deps)}) – keeping existing.")
        deps = existing_deps

    if verbose:
        print(f"    → {len(existing_ops)} → {len(ops)} operations, "
              f"{len(existing_rats)} → {len(rats)} rationals, "
              f"{len(existing_deps)} → {len(deps)} causalities after merge")
        print("\n  --- Operations ---")
        print(json.dumps(ops, indent=2))
        print("\n  --- Rationals ---")
        print(json.dumps(rats, indent=2))
        print("\n  --- Causalities ---")
        print(json.dumps(deps, indent=2))

    return ops, rats, deps


def process_text(backend, text: str, existing_graph: dict | None,
                 verbose: bool) -> tuple:
    """
    Returns (operations, rationals, causalities), or ([], [], []) if the chunk
    contains no extractable content.
    Dispatches to one_shot_extract for the first source, one_shot_merge for the rest.
    """
    if existing_graph is None:
        return one_shot_extract(backend, text, verbose)

    # If the chunk yields nothing, return the existing graph unchanged
    ex_ops, ex_rats, ex_deps = _strip_graph_ids(existing_graph)
    ops, rats, deps = one_shot_merge(backend, text, ex_ops, ex_rats, ex_deps, verbose)
    if not ops:
        return ex_ops, ex_rats, ex_deps
    return ops, rats, deps


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_graph(operations: list, rationals: list, dependencies: list,
                sources: list, model: str, provider: str) -> dict:
    """
    Assemble the final graph dict.
    Operations with no incoming edge are dropped; rationals with no outgoing
    edge are dropped.
    """
    # Drop malformed causality entries missing required keys
    dependencies = [
        dep for dep in dependencies
        if dep.get("source") and dep.get("target")
    ]

    # Build name sets from the actual node lists
    known_op_names  = {op["name"]     for op in operations if op.get("name")}
    known_rat_names = {r["rational"]  for r  in rationals  if r.get("rational")}

    # ── Step 1: drop edges whose endpoints don't exist in the node lists ──────
    valid_deps = []
    for dep in dependencies:
        src, tgt = dep.get("source", ""), dep.get("target", "")
        if src in known_rat_names and tgt in known_op_names:
            valid_deps.append(dep)
    dependencies = valid_deps

    # ── Step 2: prune orphan nodes (no edge touches them) ────────────────────
    ops_covered  = {dep["target"] for dep in dependencies}
    rats_covered = {dep["source"] for dep in dependencies}

    operations = [op for op in operations if op.get("name")    and op["name"]    in ops_covered]
    rationals  = [r  for r  in rationals  if r.get("rational") and r["rational"] in rats_covered]

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
                "extractor":       "one_shot_pipeline_extractor",
            },
        }
    }


def _strip_graph_ids(graph: dict) -> tuple:
    """
    Recover plain operation/rational/causality lists from a previously built graph.
    Removes synthetic 'id' and 'node_type' keys before passing data back to the LLM.
    Returns (operations, rationals, edges).
    """
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
        description="One-shot biosignal pipeline graph extractor – all stages in one LLM call.",
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

    # ── Semantic chunking ──────────────────────────────────────────────────
    parser.add_argument("--chunk", action="store_true",
                        help="Split each input text into semantic mini-batches "
                             "before extraction (uses sentence-transformers).")
    parser.add_argument("--chunk-model", default=DEFAULT_CHUNK_EMBED_MODEL,
                        metavar="MODEL",
                        help=(f"Sentence-transformers model for semantic chunking "
                              f"(default: {DEFAULT_CHUNK_EMBED_MODEL})"))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        metavar="PERCENTILE",
                        help=(f"Breakpoint percentile threshold for SemanticChunker "
                              f"[50–99]. Higher = fewer, larger chunks. "
                              f"(default: {DEFAULT_CHUNK_SIZE})"))

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument("--output", default="pipeline_graph.json",
                        help="Output JSON file (default: pipeline_graph.json)")
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
        print(f"Mode     : one-shot (single LLM call per source)")
        if args.chunk:
            print(f"Chunking : enabled  |  embed model: {args.chunk_model}  "
                  f"|  percentile threshold: {args.chunk_size}")

    # ── Collect input texts ────────────────────────────────────────────────
    if args.text:
        raw_texts, raw_labels = [args.text], ["<inline text>"]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            raw_texts, raw_labels = [f.read()], [args.file]
    else:
        raw_texts, raw_labels = [], []
        for path in args.files:
            with open(path, encoding="utf-8") as f:
                raw_texts.append(f.read())
            raw_labels.append(path)

    # ── Optionally expand each text into semantic chunks ───────────────────
    if args.chunk:
        texts, labels = [], []
        for raw_text, raw_label in zip(raw_texts, raw_labels):
            chunks = chunk_text_semantically(
                raw_text, args.chunk_model, args.chunk_size, verbose
            )
            for chunk_idx, chunk in enumerate(chunks):
                texts.append(chunk)
                labels.append(f"{raw_label} [chunk {chunk_idx + 1}/{len(chunks)}]")
        if verbose:
            print(f"\n  Total sources after chunking: {len(texts)}")
    else:
        texts, labels = raw_texts, raw_labels

    # ── Run pipeline ───────────────────────────────────────────────────────
    current_graph  = None
    skipped_chunks = 0
    for idx, (text, label) in enumerate(zip(texts, labels)):
        if verbose:
            bar = "=" * 60
            print(f"\n{bar}\n  Source {idx+1}/{len(texts)}: {label}\n{bar}")
        ops, rats, deps = process_text(backend, text, current_graph, verbose)
        if not ops:
            skipped_chunks += 1
            if verbose:
                print(f"    (chunk skipped – graph unchanged)")
            continue
        current_graph = build_graph(ops, rats, deps, labels[:idx+1], model, args.provider)

    if current_graph is None:
        print("ERROR: No operations were extracted from any chunk. "
              "Check your input text or try a lower --chunk-size.")
        sys.exit(1)

    if verbose and skipped_chunks:
        print(f"\n  Skipped {skipped_chunks} chunk(s) with no extractable operations.")

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
