#!/usr/bin/env python3
"""
graph_eval.py -- evaluategraphfunction: score a generated pipeline graph
against a ground-truth graph.
==============================================================================
USAGE
-----
    from graph_eval import evaluate_graph_function

    result = evaluate_graph_function(generated_graph, ground_truth_graph)
    print(result["node_f1"], result["edge_f1"])

Requirements: pip install networkx scipy sentence-transformers
"""

from __future__ import annotations

import time
from difflib import SequenceMatcher
from typing import Optional

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError(
        "graph_eval.py requires numpy and scipy. Install with: "
        "pip install numpy scipy"
    ) from exc

try:
    import networkx as nx
except ImportError as exc:
    raise ImportError(
        "graph_eval.py requires networkx for the GED metric. Install with: "
        "pip install networkx"
    ) from exc


DEFAULT_NODE_MATCH_THRESHOLD_RATIONAL = 0.67
DEFAULT_NODE_MATCH_THRESHOLD_OPERATION = 0.75

DEFAULT_GED_TIMEOUT_SEC = 15
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
MIN_KEY_SIMILARITY_FOR_VALUE_MATCH = 0.3

# ============================================================
# TEXT SIMILARITY
# ============================================================

def _text_similarity(a: str, b: str) -> float:
    """SequenceMatcher ratio in [0, 1]. Symmetric, deterministic."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


_EMBEDDING_MODEL = None
_EMBEDDING_CACHE: dict = {}


def _get_embedding_model():

    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL


def _embed(text: str) -> "np.ndarray":

    if text not in _EMBEDDING_CACHE:
        model = _get_embedding_model()
        _EMBEDDING_CACHE[text] = model.encode(
            text, normalize_embeddings=True, show_progress_bar=False,
        )
    return _EMBEDDING_CACHE[text]


def _embedding_similarity(a: str, b: str) -> float:

    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    cos = float(np.dot(_embed(a), _embed(b)))
    return max(0.0, min(1.0, cos))


def _parse_param(param: str) -> tuple:
    """Split a 'key: value' string into (key, value), both stripped. A
    string with no colon is treated as a bare key with an empty value."""
    if ":" in param:
        key, _, value = param.partition(":")
        return key.strip(), value.strip()
    return param.strip(), ""


def _split_top_level_commas(s: str) -> list:

    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _value_token_set(value: str) -> set:

    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    tokens = set()
    for raw in _split_top_level_commas(value):
        t = raw.strip()
        if not t:
            continue
        if t.startswith("[") and t.endswith("]"):
            inner = ",".join(p.strip() for p in t[1:-1].split(","))
            t = f"[{inner}]"
        tokens.add(t.lower())
    return tokens


def _flatten_params(params: Optional[list]) -> list:

    flat = []
    for p in (params or []):
        if not isinstance(p, str) or not p.strip():
            continue
        key, value = _parse_param(p)
        tokens = _value_token_set(value)
        if not tokens:
            flat.append((key, ""))
        else:
            for token in sorted(tokens):  # sorted: deterministic iteration order
                flat.append((key, token))
    return flat


def _param_overlap_score(gen_params: Optional[list], gt_params: Optional[list]) -> float:

    gen_flat = _flatten_params(gen_params)
    gt_flat = _flatten_params(gt_params)
    if not gen_flat and not gt_flat:
        return 1.0
    if not gen_flat or not gt_flat:
        return 0.0

    claimed_gt = set()
    total = 0.0
    for gen_key, gen_value in gen_flat:
        best_score, best_j = 0.0, None
        for j, (gt_key, gt_value) in enumerate(gt_flat):
            if j in claimed_gt:
                continue
            if gen_value.lower() != gt_value.lower():
                continue  # exact value-token match is a hard requirement, not fuzzy
            key_sim = _embedding_similarity(gen_key, gt_key)
            if key_sim < MIN_KEY_SIMILARITY_FOR_VALUE_MATCH:
                continue  # value coincidence alone isn't enough -- the keys must be plausibly the same concept
            if key_sim > best_score:
                best_score, best_j = key_sim, j
        if best_j is not None:
            claimed_gt.add(best_j)
            total += best_score
    return 2 * total / (len(gen_flat) + len(gt_flat))


def _node_similarity(node_a: dict, node_b: dict) -> float:

    name_sim = _embedding_similarity(
        (node_a.get("name") or "").strip(),
        (node_b.get("name") or "").strip(),
    )
    evidence_sim = _text_similarity(
        (node_a.get("text_evidence") or "").strip().lower(),
        (node_b.get("text_evidence") or "").strip().lower(),
    )
    if node_a.get("node_type") == "operation":
        gen_params = node_a.get("parameters")
        gt_params = node_b.get("parameters")
        if not gen_params or not gt_params:
            if not gen_params and not gt_params:
                return (name_sim + evidence_sim + 1.0) / 3.0
    
            return (name_sim + evidence_sim) / 2.0
        param_sim = _param_overlap_score(gen_params, gt_params)
        return (name_sim + evidence_sim + param_sim) / 3.0
    return (name_sim + evidence_sim) / 2.0


# ============================================================
# NODE ALIGNMENT (the "diff matrix")
# ============================================================

def build_node_similarity_matrix(gen_nodes: list, gt_nodes: list) -> "np.ndarray":
    """
    Returns an (len(gen_nodes) x len(gt_nodes)) matrix of similarity scores
    in [0, 1]. Cross node_type pairs (operation vs rational) are forced to
    0 -- an operation node should never be matched to a rational node
    regardless of text overlap. Same-type pairs are scored by
    _node_similarity, which uses a different formula per type (see there).
    """
    n, m = len(gen_nodes), len(gt_nodes)
    matrix = np.zeros((n, m), dtype=float)
    for i, gn in enumerate(gen_nodes):
        for j, gtn in enumerate(gt_nodes):
            if gn.get("node_type") != gtn.get("node_type"):
                continue
            matrix[i, j] = _node_similarity(gn, gtn)
    return matrix


def align_nodes(gen_nodes: list, gt_nodes: list,
                 threshold_rational: float = DEFAULT_NODE_MATCH_THRESHOLD_RATIONAL,
                 threshold_operation: float = DEFAULT_NODE_MATCH_THRESHOLD_OPERATION) -> dict:

    matrix = build_node_similarity_matrix(gen_nodes, gt_nodes)
    gen_ids = [n["id"] for n in gen_nodes]
    gt_ids = [n["id"] for n in gt_nodes]
    gen_types = [n.get("node_type") for n in gen_nodes]

    alignment, matched_gen_idx, matched_gt_idx = _hungarian_align(
        matrix, gen_ids, gt_ids, gen_types, threshold_rational, threshold_operation,
    )

    unmatched_gen_ids = [gen_ids[i] for i in range(len(gen_ids)) if i not in matched_gen_idx]
    unmatched_gt_ids = [gt_ids[j] for j in range(len(gt_ids)) if j not in matched_gt_idx]
    gen_id_to_gt_id = {a["gen_id"]: a["gt_id"] for a in alignment}

    return {
        "matrix": matrix.tolist(),
        "alignment": alignment,
        "unmatched_gen_ids": unmatched_gen_ids,
        "unmatched_gt_ids": unmatched_gt_ids,
        "gen_id_to_gt_id": gen_id_to_gt_id,
    }


def _hungarian_align(matrix: "np.ndarray", gen_ids: list, gt_ids: list, gen_types: list,
                      threshold_rational: float, threshold_operation: float) -> tuple:
    alignment = []
    matched_gen_idx, matched_gt_idx = set(), set()

    if matrix.size > 0:
        thresholds = np.array([
            threshold_operation if t == "operation" else threshold_rational
            for t in gen_types
        ])
        thresholded_matrix = np.where(matrix >= thresholds[:, None], matrix, 0.0)

        cost = 1.0 - thresholded_matrix
        row_idx, col_idx = linear_sum_assignment(cost)
        for i, j in zip(row_idx, col_idx):
            sim = float(matrix[i, j])
            threshold = threshold_operation if gen_types[i] == "operation" else threshold_rational
            if sim >= threshold:
                alignment.append({
                    "gen_index": int(i),
                    "gt_index": int(j),
                    "gen_id": gen_ids[i],
                    "gt_id": gt_ids[j],
                    "similarity": round(sim, 4),
                })
                matched_gen_idx.add(i)
                matched_gt_idx.add(j)

    return alignment, matched_gen_idx, matched_gt_idx


def _prf1(true_positive: float, n_predicted: int, n_actual: int) -> dict:
    precision = true_positive / n_predicted if n_predicted else (1.0 if n_actual == 0 else 0.0)
    recall = true_positive / n_actual if n_actual else (1.0 if n_predicted == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ============================================================
# EDGE COMPARISON (via the node alignment)
# ============================================================

EDGE_EVIDENCE_WEIGHT = 1 / 3
EDGE_STRUCTURE_WEIGHT = 2 / 3
EDGE_TYPE_PATTERN_WEIGHT = 0.4   # fraction of EDGE_STRUCTURE_WEIGHT
EDGE_POLARITY_WEIGHT = 0.4       # fraction of EDGE_STRUCTURE_WEIGHT
EDGE_STRENGTH_WEIGHT = 0.2       # fraction of EDGE_STRUCTURE_WEIGHT


EDGE_STRENGTH_MODE = "label"


def _classify_strength(edge_type: str) -> str:

    text = (edge_type or "").lower()
    if "must" in text:
        return "must"
    if "should" in text:
        return "should"
    return "can"


def _normalise_label(edge_type) -> str:
    return " ".join(str(edge_type or "").lower().split())


def _strength_match(gen_edge_type, gt_edge_type) -> float:
    """The strength term of the edge score: 1.0 or 0.0 (see EDGE_STRENGTH_MODE)."""
    if EDGE_STRENGTH_MODE == "keyword":
        return 1.0 if _classify_strength(gen_edge_type) == _classify_strength(gt_edge_type) else 0.0
    a, b = _normalise_label(gen_edge_type), _normalise_label(gt_edge_type)
    return 1.0 if a and a == b else 0.0


def build_edge_similarity_matrix(gen_edges: list, gt_edges: list,
                                  gen_id_to_gt_id: dict,
                                  gen_nodes_by_id: dict, gt_nodes_by_id: dict) -> "np.ndarray":

    n, m = len(gen_edges), len(gt_edges)
    matrix = np.zeros((n, m), dtype=float)
    for i, ge in enumerate(gen_edges):
        mapped_source = gen_id_to_gt_id.get(ge.get("source"))
        mapped_target = gen_id_to_gt_id.get(ge.get("target"))
        if mapped_source is None or mapped_target is None:
            continue  # cascade: an unmatched endpoint means this edge can never score

        gen_source_node = gen_nodes_by_id.get(ge.get("source")) or {}
        gen_target_node = gen_nodes_by_id.get(ge.get("target")) or {}
        gen_type_pattern = (gen_source_node.get("node_type"), gen_target_node.get("node_type"))
        gen_evidence = (ge.get("text_evidence") or "").strip().lower()

        for j, gte in enumerate(gt_edges):
            if mapped_source != gte.get("source") or mapped_target != gte.get("target"):
                continue  # not the same specific node pair -- no partial credit

            gt_source_node = gt_nodes_by_id.get(gte.get("source")) or {}
            gt_target_node = gt_nodes_by_id.get(gte.get("target")) or {}
            gt_type_pattern = (gt_source_node.get("node_type"), gt_target_node.get("node_type"))

            evidence_sim = _text_similarity(
                gen_evidence, (gte.get("text_evidence") or "").strip().lower(),
            )
            type_match = 1.0 if gen_type_pattern == gt_type_pattern else 0.0
            polarity_match = 1.0 if ge.get("polarity") == gte.get("polarity") else 0.0
            strength_match = _strength_match(ge.get("edge_type"), gte.get("edge_type"))

            structure_score = (
                EDGE_TYPE_PATTERN_WEIGHT * type_match
                + EDGE_POLARITY_WEIGHT * polarity_match
                + EDGE_STRENGTH_WEIGHT * strength_match
            )
            matrix[i, j] = (
                EDGE_EVIDENCE_WEIGHT * evidence_sim + EDGE_STRUCTURE_WEIGHT * structure_score
            )
    return matrix


def evaluate_edges(gen_edges: list, gt_edges: list, gen_id_to_gt_id: dict,
                    gen_nodes_by_id: dict, gt_nodes_by_id: dict) -> dict:

    matrix = build_edge_similarity_matrix(
        gen_edges, gt_edges, gen_id_to_gt_id, gen_nodes_by_id, gt_nodes_by_id,
    )
    matched_gt = set()
    total_score = 0.0
    matched_pairs = []

    for i, ge in enumerate(gen_edges):
        best_j, best_score = None, 0.0
        for j in range(len(gt_edges)):
            if j in matched_gt:
                continue
            if matrix[i, j] > best_score:
                best_score, best_j = float(matrix[i, j]), j
        if best_j is not None and best_score > 0:
            matched_gt.add(best_j)
            total_score += best_score
            matched_pairs.append({
                "gen_edge_id": ge.get("id"),
                "gt_edge_id": gt_edges[best_j].get("id"),
                "score": round(best_score, 4),
            })

    result = _prf1(total_score, len(gen_edges), len(gt_edges))
    result["matched_pairs"] = matched_pairs
    result["mean_edge_score_on_matches"] = (
        round(sum(p["score"] for p in matched_pairs) / len(matched_pairs), 4)
        if matched_pairs else None
    )
    result["matrix"] = matrix.tolist()
    return result


# ============================================================
# APPROXIMATE GRAPH EDIT DISTANCE (secondary global scalar)
# ============================================================

def _to_networkx(graph: dict, prefix: str) -> "nx.DiGraph":
    g = nx.DiGraph()
    for node in graph.get("nodes", []):
        g.add_node(
            f"{prefix}{node['id']}",
            name=node.get("name", ""),
            node_type=node.get("node_type", ""),
            text_evidence=node.get("text_evidence", ""),
            parameters=node.get("parameters", []),
        )
    for edge in graph.get("edges", []):
        g.add_edge(
            f"{prefix}{edge['source']}",
            f"{prefix}{edge['target']}",
            polarity=edge.get("polarity", ""),
            edge_type=edge.get("edge_type", ""),
        )
    return g


def _ged_worker(generated: dict, ground_truth: dict, threshold_rational: float,
                 threshold_operation: float, queue) -> None:

    try:
        g1 = _to_networkx(generated, "g_")
        g2 = _to_networkx(ground_truth, "t_")

        def node_match(n1, n2):
            if n1.get("node_type") != n2.get("node_type"):
                return False
            threshold = threshold_operation if n1.get("node_type") == "operation" else threshold_rational
            return _node_similarity(n1, n2) >= threshold

        def edge_match(e1, e2):
            return e1.get("polarity") == e2.get("polarity")

        best = None
        is_exact = False
        for value in nx.optimize_graph_edit_distance(
            g1, g2, node_match=node_match, edge_match=edge_match,
        ):
            best = value
            # best-effort: if we get a cheap early value, report it; the
            # parent process enforces the actual wall-clock deadline since
            # this loop itself may never yield in time (see below)
        else:
            is_exact = True
        queue.put(("ok", {"ged": best, "ged_is_exact": is_exact}))
    except Exception as exc:  # pragma: no cover -- defensive, reported to parent
        queue.put(("error", str(exc)))


def approx_graph_edit_distance(generated: dict, ground_truth: dict,
                                threshold_rational: float = DEFAULT_NODE_MATCH_THRESHOLD_RATIONAL,
                                threshold_operation: float = DEFAULT_NODE_MATCH_THRESHOLD_OPERATION,
                                timeout_sec: float = DEFAULT_GED_TIMEOUT_SEC) -> dict:

    import multiprocessing as mp

    start = time.perf_counter()
    ctx = mp.get_context("spawn")  # explicit: consistent behavior on Windows and Unix
    queue = ctx.Queue()
    proc = ctx.Process(target=_ged_worker,
                        args=(generated, ground_truth, threshold_rational, threshold_operation, queue))
    proc.start()
    proc.join(timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)  # give the OS a moment to actually reclaim it
        return {
            "ged": None, "ged_is_exact": False,
            "ged_skipped_reason": f"exceeded {timeout_sec}s hard timeout -- graph "
                                   f"too large for a practical GED search "
                                   f"({len(generated.get('nodes', []))} vs "
                                   f"{len(ground_truth.get('nodes', []))} nodes)",
            "ged_elapsed_sec": round(time.perf_counter() - start, 2),
        }

    if not queue.empty():
        status, payload = queue.get()
        if status == "ok":
            payload["ged_elapsed_sec"] = round(time.perf_counter() - start, 2)
            return payload
        return {"ged": None, "ged_is_exact": False, "ged_error": payload,
                "ged_elapsed_sec": round(time.perf_counter() - start, 2)}

    return {
        "ged": None, "ged_is_exact": False,
        "ged_error": "worker process exited without producing a result "
                      "(crashed or was killed externally)",
        "ged_elapsed_sec": round(time.perf_counter() - start, 2),
    }


# ============================================================
# TOP-LEVEL evaluategraphfunction
# ============================================================

def evaluate_graph_function(generated: dict, ground_truth: dict,
                             node_match_threshold_rational: float = DEFAULT_NODE_MATCH_THRESHOLD_RATIONAL,
                             node_match_threshold_operation: float = DEFAULT_NODE_MATCH_THRESHOLD_OPERATION,
                             ged_timeout_sec: float = DEFAULT_GED_TIMEOUT_SEC,
                             compute_ged: bool = False) -> dict:

    gen_nodes = generated.get("nodes", [])
    gt_nodes = ground_truth.get("nodes", [])
    gen_edges = generated.get("edges", [])
    gt_edges = ground_truth.get("edges", [])

    node_alignment = align_nodes(
        gen_nodes, gt_nodes,
        threshold_rational=node_match_threshold_rational,
        threshold_operation=node_match_threshold_operation,
    )
    node_scores = _prf1(len(node_alignment["alignment"]), len(gen_nodes), len(gt_nodes))

    edge_scores = evaluate_edges(
        gen_edges, gt_edges, node_alignment["gen_id_to_gt_id"],
        gen_nodes_by_id={n["id"]: n for n in gen_nodes},
        gt_nodes_by_id={n["id"]: n for n in gt_nodes},
    )

    result = {
        "node_precision": node_scores["precision"],
        "node_recall": node_scores["recall"],
        "node_f1": node_scores["f1"],
        "edge_precision": edge_scores["precision"],
        "edge_recall": edge_scores["recall"],
        "edge_f1": edge_scores["f1"],
        "mean_edge_score_on_matches": edge_scores["mean_edge_score_on_matches"],
        "n_generated_nodes": len(gen_nodes),
        "n_ground_truth_nodes": len(gt_nodes),
        "n_generated_edges": len(gen_edges),
        "n_ground_truth_edges": len(gt_edges),
        "node_alignment": node_alignment,
        "edge_alignment": {
            "matched_pairs": edge_scores["matched_pairs"],
            "matrix": edge_scores["matrix"],
        },
        "node_match_threshold_rational": node_match_threshold_rational,
        "node_match_threshold_operation": node_match_threshold_operation,
    }

    if compute_ged:
        result.update(approx_graph_edit_distance(
            generated, ground_truth,
            threshold_rational=node_match_threshold_rational,
            threshold_operation=node_match_threshold_operation,
            timeout_sec=ged_timeout_sec,
        ))

    return result


# ============================================================
# SYNTACTIC / STRUCTURAL VALIDATION
# ============================================================

# Closed edge_type vocabulary per relationship shape (source_node_type,
# target_node_type) -> {"positive": {...}, "negative": {...}}. Mirrors
# icl_pipeline_extractor.py's prompt instructions -- this is the single
# source of truth for VALIDATING a graph; if the prompt's allowed values
# ever change, this table must be updated to match, and vice versa.
EDGE_TYPE_VOCABULARY = {
    ("rational", "operation"): {
        "positive": {"can apply", "should apply", "must apply"},
        "negative": {"can not apply", "should not apply", "must not apply"},
    },
    ("operation", "operation"): {
        "positive": {"can be followed by", "must be followed by", "should be followed by"},
        "negative": {"can not be followed by", "should not be followed by", "must not be followed by"},
    },
    ("operation", "rational"): {
        "positive": {"mandatorily implies", "can lead to"},
        "negative": set(),  # no negative form defined for this shape
    },
    ("rational", "rational"): {
        "positive": {"may lead to", "causes", "in conjunction with"},
        "negative": {"does not cause"},
    },
}


def validate_graph_syntax(graph: dict) -> dict:
 
    violations = []

    def _violation(severity, check, message, **ids):
        violations.append({"severity": severity, "check": check, "message": message, **ids})

    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    node_ids_seen = set()
    node_type_by_id = {}
    for n in nodes:
        nid = n.get("id")
        if not isinstance(nid, int):
            _violation("error", "node_id_not_int", f"node id {nid!r} is not an integer", node_id=nid)
            continue
        if nid in node_ids_seen:
            _violation("error", "duplicate_node_id", f"node id {nid} appears more than once", node_id=nid)
        node_ids_seen.add(nid)
        node_type_by_id[nid] = n.get("node_type")
        if n.get("node_type") not in ("operation", "rational"):
            _violation("error", "invalid_node_type",
                       f'node {nid} has node_type={n.get("node_type")!r}, expected "operation" or "rational"',
                       node_id=nid)

    edge_ids_seen = set()
    for e in edges:
        eid = e.get("id")
        if not isinstance(eid, int):
            _violation("error", "edge_id_not_int", f"edge id {eid!r} is not an integer", edge_id=eid)
        elif eid in edge_ids_seen:
            _violation("error", "duplicate_edge_id", f"edge id {eid} appears more than once", edge_id=eid)
        edge_ids_seen.add(eid)

        source, target = e.get("source"), e.get("target")
        source_ok = source in node_type_by_id
        target_ok = target in node_type_by_id
        if not source_ok:
            _violation("error", "dangling_edge_source",
                       f"edge {eid} references source={source!r}, not a node id defined in this graph",
                       edge_id=eid)
        if not target_ok:
            _violation("error", "dangling_edge_target",
                       f"edge {eid} references target={target!r}, not a node id defined in this graph",
                       edge_id=eid)
        if source_ok and target_ok and source == target:
            _violation("error", "self_loop", f"edge {eid} has source == target == {source}", edge_id=eid)

        polarity = e.get("polarity")
        if polarity not in ("positive", "negative"):
            _violation("error", "invalid_polarity",
                       f'edge {eid} has polarity={polarity!r}, expected "positive" or "negative"', edge_id=eid)

        if source_ok and target_ok:
            shape = (node_type_by_id[source], node_type_by_id[target])
            vocab = EDGE_TYPE_VOCABULARY.get(shape)
            if vocab is None:
                continue  # invalid node_type on source/target already reported above
            edge_type = (e.get("edge_type") or "").strip().lower()
            allowed_all = vocab["positive"] | vocab["negative"]
            if edge_type not in allowed_all:
                _violation("warning", "edge_type_outside_vocabulary",
                           f'edge {eid} ({shape[0]}->{shape[1]}) uses edge_type={e.get("edge_type")!r}, '
                           f"not in the closed vocabulary for this shape: {sorted(allowed_all)}",
                           edge_id=eid)
            elif polarity in ("positive", "negative") and edge_type not in vocab[polarity]:
                other = "negative" if polarity == "positive" else "positive"
                _violation("error", "polarity_edge_type_mismatch",
                           f'edge {eid} declares polarity={polarity!r} but edge_type={e.get("edge_type")!r} '
                           f"belongs to the {other} vocabulary for {shape[0]}->{shape[1]}",
                           edge_id=eid)


    adjacency = {nid: set() for nid in node_type_by_id}
    for e in edges:
        source, target = e.get("source"), e.get("target")
        if source in node_type_by_id and target in node_type_by_id and source != target:
            adjacency[source].add(target)
            adjacency[target].add(source)

    visited = set()
    for start in node_type_by_id:
        if start in visited:
            continue
        component = []
        stack = [start]
        visited.add(start)
        while stack:
            cur = stack.pop()
            component.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        if not any(node_type_by_id[nid] == "operation" for nid in component):
            component_sorted = sorted(component)
            _violation("error", "isolated_rational_island",
                       f"{len(component_sorted)} rational node(s) {component_sorted} form a "
                       "connected component with no operation node reachable -- this "
                       "rationale is never anchored to any operation",
                       node_ids=component_sorted)

    is_valid = not any(v["severity"] == "error" for v in violations)
    return {"is_valid": is_valid, "violations": violations}


if __name__ == "__main__":
    # Minimal self-test with synthetic graphs -- run with:
    #   python graph_eval.py
    import json as _json

    identical = {
        "nodes": [
            {"id": 1, "name": "Low-pass filter", "node_type": "operation",
             "parameters": [], "text_evidence": "a low pass filter was applied"},
            {"id": 2, "name": "remove high frequency noise", "node_type": "rational",
             "parameters": [], "text_evidence": "to remove high frequency noise"},
        ],
        "edges": [
            {"id": 1, "source": 2, "target": 1, "polarity": "positive",
             "edge_type": "can apply", "text_evidence": "applied to remove noise"},
        ],
    }
    near_duplicate = {
        "nodes": [
            {"id": 10, "name": "Lowpass filtering", "node_type": "operation",
             "parameters": [], "text_evidence": "a low pass filter was applied"},
            {"id": 20, "name": "eliminate high-frequency noise", "node_type": "rational",
             "parameters": [], "text_evidence": "to remove high frequency noise"},
            {"id": 30, "name": "unrelated extra node", "node_type": "operation",
             "parameters": [], "text_evidence": "something not in ground truth"},
        ],
        "edges": [
            {"id": 1, "source": 20, "target": 10, "polarity": "positive",
             "edge_type": "should apply", "text_evidence": "applied to remove noise"},
        ],
    }

    print("=== identical vs identical (should be perfect scores) ===")
    print(_json.dumps(
        {k: v for k, v in evaluate_graph_function(identical, identical).items()
         if k not in ("node_alignment", "edge_alignment")},
        indent=2,
    ))

    print("\n=== near_duplicate vs identical (partial match expected) ===")
    print(_json.dumps(
        {k: v for k, v in evaluate_graph_function(near_duplicate, identical).items()
         if k not in ("node_alignment", "edge_alignment")},
        indent=2,
    ))

    print(
        "\n=== operation-node parameter-weighting regression test: same name + "
        "evidence, different parameters -- must score 2/3, not 1.0 ==="
    )
    same_name_evidence_diff_params = _node_similarity(
        {"name": "Low-pass filter", "node_type": "operation",
         "text_evidence": "a low pass filter was applied", "parameters": ["cutoff_hz: 50"]},
        {"name": "Low-pass filter", "node_type": "operation",
         "text_evidence": "a low pass filter was applied", "parameters": ["cutoff_hz: 100"]},
    )
    print(f"same name+evidence, different params: similarity={same_name_evidence_diff_params:.4f}")
    assert abs(same_name_evidence_diff_params - 2 / 3) < 1e-9, (
        f"expected exactly 2/3 (name=1.0, evidence=1.0, params=0.0, averaged), "
        f"got {same_name_evidence_diff_params}"
    )
    both_empty_params = _node_similarity(
        {"name": "SDNN", "node_type": "operation", "text_evidence": "SDNN", "parameters": []},
        {"name": "SDNN", "node_type": "operation", "text_evidence": "SDNN", "parameters": []},
    )
    assert both_empty_params == 1.0, (
        f"expected 1.0 when both sides have zero parameters, got {both_empty_params}"
    )
    print("PASSED: operation-node parameters now contribute 1/3 of the score, "
          "separately from name and text_evidence.")

    print(
        "\n=== operation-node ASYMMETRIC-empty-parameters regression test: "
        "one side extracted zero parameters, the other extracted some -- "
        "real pair from 01_batch: gen \"HRV triangular index\" carries "
        "\"recommended_recording_duration_min: >=20\" while ground truth's "
        "\"HRV Triangular Index (HTI)\" has no parameters at all. Missing "
        "information on one side must NOT be scored as a confirmed "
        "contradiction (a hard params=0.0 in the 3-way average) -- it must "
        "fall back to the 2-way name+evidence average instead ==="
    )
    asymmetric_params = _node_similarity(
        {"name": "HRV triangular index", "node_type": "operation",
         "text_evidence": "recordings of at least 20 min are recommended",
         "parameters": ["recommended_recording_duration_min: >=20"]},
        {"name": "HRV triangular index", "node_type": "operation",
         "text_evidence": "recordings of at least 20 min are recommended",
         "parameters": []},
    )
    assert asymmetric_params == 1.0, (
        f"name and evidence are identical on both sides here, so with params "
        f"excluded from the average (not zeroed) this must be a perfect 1.0, "
        f"got {asymmetric_params}"
    )
    asymmetric_params_none_side = _node_similarity(
        {"name": "Lorenz plot analysis", "node_type": "operation",
         "text_evidence": "geometrical properties of the NN scatter plots",
         "parameters": ["recommended_recording_duration_min: >=20"]},
        {"name": "Lorenz plot", "node_type": "operation",
         "text_evidence": "other geometrical properties of the NN scatter plots",
         "parameters": None},
    )
    assert asymmetric_params_none_side > 0.7, (
        f"a near-identical name+evidence pair with one side's parameters "
        f"missing (None, not just []) must score well above what a hard "
        f"params=0.0 penalty would allow (~0.6), got {asymmetric_params_none_side}"
    )
    print(f"one side empty ([]), other has a real param: similarity={asymmetric_params:.4f}")
    print(f"one side empty (None), other has a real param: similarity={asymmetric_params_none_side:.4f}")
    print("PASSED: an asymmetric parameter list (info missing on one side, not "
          "confirmed to disagree) no longer drags the score down via a hard "
          "params=0.0 in the 3-way average -- it falls back to name+evidence.")

    print(
        "\n=== embedding regression test 1: name PARAPHRASE (word order swap) "
        "must score much higher via embeddings than plain SequenceMatcher -- "
        "real pair from 02_batch: gen \"QRS complex detection\" vs ground "
        "truth \"detection of QRS complex\" ==="
    )
    paraphrase_name_sim = _embedding_similarity("QRS complex detection", "detection of QRS complex")
    old_string_sim = _text_similarity("qrs complex detection", "detection of qrs complex")
    print(f"embedding name_sim={paraphrase_name_sim:.4f}  vs  old SequenceMatcher={old_string_sim:.4f}")
    assert paraphrase_name_sim > old_string_sim + 0.2, (
        f"expected embeddings to clearly beat SequenceMatcher on a genuine "
        f"paraphrase, got embedding={paraphrase_name_sim} vs string={old_string_sim}"
    )
    print("PASSED: embeddings recognize the word-order paraphrase; SequenceMatcher"
          " alone scored it well below match threshold.")

    print(
        "\n=== embedding regression test 2: parameter KEY-spelling variant, "
        "SAME value -- must now match (was a false-negative under brute-force "
        "exact-string comparison) ==="
    )
    key_spelling_score = _param_overlap_score(
        ["cutoff_frequency_hz: 50"], ["cut_off_frequency_hz: 50"],
    )
    print(f"key spelling variant, same value: param_score={key_spelling_score:.4f}")
    assert key_spelling_score > 0.8, (
        f"expected a high score -- same value, key differs only by an "
        f"underscore -- got {key_spelling_score}"
    )
    print("PASSED: key-naming variants with identical values are now recognized.")

    print(
        "\n=== embedding regression test 3: parameter VALUE must still be "
        "EXACT -- a key-embedding match does NOT rescue a different value "
        "(49 Hz != 50 Hz, deliberately not fuzzy) ==="
    )
    wrong_value_score = _param_overlap_score(
        ["cutoff_frequency_hz: 49"], ["cutoff_frequency_hz: 50"],
    )
    print(f"identical key, different value: param_score={wrong_value_score:.4f}")
    assert wrong_value_score == 0.0, (
        f"expected 0.0 -- identical/near-identical key naming must NOT paper "
        f"over a different value, got {wrong_value_score}"
    )
    print("PASSED: parameter values remain a hard exact-match requirement.")

    print(
        "\n=== embedding regression test 4: value-list REORDERING must not "
        "count as a mismatch -- real pair from 05_batch: gen \"cutoff_frequency_"
        "hz: 11, 90, 30, 35, 50, 100, 70\" vs ground truth \"cut_off_frequency_"
        "hz: 11, 30, 35, 50, 70, 90, 100\" (same 7 numbers, different order, "
        "different key spelling) ==="
    )
    reordered_list_score = _param_overlap_score(
        ["cutoff_frequency_hz: 11, 90, 30, 35, 50, 100, 70"],
        ["cut_off_frequency_hz: 11, 30, 35, 50, 70, 90, 100"],
    )
    print(f"same values reordered + key spelling variant: param_score={reordered_list_score:.4f}")
    assert reordered_list_score > 0.8, (
        f"expected a high score -- same 7 values just reordered, key differs "
        f"only by spelling -- got {reordered_list_score}"
    )
    print("PASSED: reordered value lists no longer score as a total mismatch.")

    print(
        "\n=== embedding regression test 5: a PARTIALLY correct value list "
        "(6 of 7 numbers right) must score partial credit, not 0 and not "
        "full credit ==="
    )
    partial_list_score = _param_overlap_score(
        ["cutoff_frequency_hz: 11, 90, 30, 35, 50, 100, 70"],
        ["cutoff_frequency_hz: 11, 90, 30, 35, 50, 100, 999"],  # 1 of 7 wrong
    )
    print(f"6 of 7 values correct: param_score={partial_list_score:.4f}")
    assert 0.0 < partial_list_score < 0.95, (
        f"expected partial credit strictly between 0 and ~1 for 6/7 correct "
        f"values, got {partial_list_score}"
    )
    print("PASSED: partially-correct value extraction earns partial credit, "
          "not an all-or-nothing score.")

    print(
        "\n=== embedding regression test 6: a value list bundled under ONE key "
        "must still be recognized against the SAME values split across MULTIPLE "
        "keys -- real pair from 05_batch: gen \"filter_count: 2, window_width_ms: "
        "200, 600\" vs ground truth \"small_width_ms: 200, large_width_ms: 600, "
        "orders: 2, 50\" (two median-filter widths, grouped differently) ==="
    )
    key_splitting_before_fix_score = 0.2196  # the old per-key-entry design's score on this exact pair
    key_splitting_score = _param_overlap_score(
        ["filter_count: 2", "window_width_ms: 200, 600"],
        ["small_width_ms: 200", "large_width_ms: 600", "orders: 2, 50"],
    )
    print(f"1-key-bundled vs 2-keys-split, same underlying values: "
          f"param_score={key_splitting_score:.4f}  (was {key_splitting_before_fix_score} before flattening)")
    assert key_splitting_score > key_splitting_before_fix_score * 1.5, (
        f"expected flattening to substantially beat the old per-key-entry score "
        f"({key_splitting_before_fix_score}) by bridging both widths, got {key_splitting_score}"
    )
    same_key_multi_value_still_works = _param_overlap_score(
        ["cutoff_hz: 50, 100"], ["cutoff_hz: 50, 100"],
    )
    assert same_key_multi_value_still_works == 1.0, (
        f"flattening must not break the common case of a matching multi-value "
        f"list under the SAME key on both sides, got {same_key_multi_value_still_works}"
    )
    print("PASSED: flattening to individual value tokens bridges a value list "
          "grouped differently across keys, without disturbing the common "
          "same-key case.")

    print(
        "\n=== embedding regression test 7: a value-token match must not "
        "earn credit from pure NUMERIC COINCIDENCE -- real pair from "
        "05_batch: gen \"filter_count: 2\" vs ground truth \"orders: 2, 50\" "
        "share only the digit \"2\"; filter COUNT and filter ORDER are "
        "unrelated quantities (key_sim ~0.08, well under the 0.3 floor) ==="
    )
    coincidence_score = _param_overlap_score(["filter_count: 2"], ["orders: 2, 50"])
    print(f"filter_count vs orders, coincidental shared digit: param_score={coincidence_score:.4f}")
    assert coincidence_score == 0.0, (
        f"expected 0.0 -- a shared digit alone must not earn credit when "
        f"the keys are unrelated, got {coincidence_score}"
    )
    legitimate_variant_score = _param_overlap_score(
        ["successive_difference_threshold_ms: 20"], ["threshold_ms: 20"],
    )
    print(f"legitimate key-naming variant (same underlying parameter): "
          f"param_score={legitimate_variant_score:.4f}")
    assert legitimate_variant_score > 0.5, (
        f"expected the gate to still ALLOW a genuine key-naming variant "
        f"through (key_sim ~0.68, above the 0.3 floor), got {legitimate_variant_score}"
    )
    print("PASSED: numeric coincidence between unrelated keys is refused, "
          "while genuine key-naming variants still pass through.")

    print(
        "\n=== parameter-notation regression test: the current set/range "
        "syntax (\"{...}\" sets, \"[a, b]\" ranges) must tokenize correctly "
        "AND the legacy bare-comma / hyphenated-range notation still in "
        "ground truth must keep working unchanged ==="
    )
    assert _value_token_set("11, 30, 35") == {"11", "30", "35"}, (
        "legacy bare comma-list must still split into individual tokens"
    )
    assert _value_token_set("0.5-40") == {"0.5-40"}, (
        "legacy hyphenated range must still be ONE atomic token, unchanged"
    )
    assert _value_token_set("{0.5, 1, 2.2, 5}") == {"0.5", "1", "2.2", "5"}, (
        "a \"{...}\" set must split its contents exactly like a bare comma-list"
    )
    assert _value_token_set("[0.5, 40]") == {"[0.5,40]"}, (
        "a \"[a, b]\" range must be ONE atomic token, not split on its internal comma"
    )
    assert _value_token_set("{[0.5, 40], [1, 40]}") == {"[0.5,40]", "[1,40]"}, (
        "a set of alternative ranges must split into one token PER range, "
        "each range still atomic"
    )
    set_vs_set_score = _param_overlap_score(
        ["cut_off_frequency_hz: {0.5, 1, 2.2, 5}"], ["cut_off_frequency_hz: {0.5, 1, 2.2, 5}"],
    )
    assert set_vs_set_score == 1.0, (
        f"identical \"{{...}}\" sets under a matching key must score a perfect "
        f"1.0, got {set_vs_set_score}"
    )
    range_vs_split_endpoints_score = _param_overlap_score(
        ["frequency_range_hz: [0.5, 40]"], ["frequency_range_hz: 0.5, 40"],
    )
    assert range_vs_split_endpoints_score == 0.0, (
        f"a single [0.5, 40] RANGE must NOT match a value list that reports "
        f"0.5 and 40 as two SEPARATE discrete values -- these are different "
        f"claims (one continuous interval vs. two specific points), got "
        f"{range_vs_split_endpoints_score}"
    )
    print(f"set-vs-set (identical): param_score={set_vs_set_score:.4f}")
    print(f"range-as-one-token vs two-split-endpoints (must differ): "
          f"param_score={range_vs_split_endpoints_score:.4f}")
    print("PASSED: current set/range notation tokenizes correctly, a range is never "
          "confused with its two endpoints reported separately, and legacy "
          "bare-comma/hyphenated-range values already in ground truth are unaffected.")

    print(
        "\n=== node regression test: the optimizer must not sacrifice a "
        "real, threshold-clearing match to marginally help a doomed pair "
        "that gets rejected by the threshold anyway (a real pathology "
        "confirmed in 01_batch: gen[19] scored 0.69 against its true "
        "match but the OLD raw-sum optimizer assigned it elsewhere to "
        "let a spurious node inch its own doomed score up) ==="
    )
    # Controlled synthetic matrix reproducing the exact shape of the bug:
    # raw-sum-optimal assignment (A->gtY, B->gtX) beats the correct one
    # (A->gtX, B->gtY) on RAW similarity (1.25 vs 0.78), even though it
    # yields ZERO accepted matches instead of one. Exercised directly
    # against _hungarian_align (not real node dicts/embeddings) so the
    # exact similarity values are controlled precisely.
    rescue_matrix = np.array([
        [0.68, 0.60],   # gen A: 0.68 vs gt X (clears 0.67), 0.60 vs gt Y (doesn't)
        [0.65, 0.10],   # gen B: 0.65 vs gt X (doesn't clear), 0.10 vs gt Y
    ])
    rescue_alignment, _, _ = _hungarian_align(
        rescue_matrix, gen_ids=["A", "B"], gt_ids=["X", "Y"], gen_types=["rational", "rational"],
        threshold_rational=0.67, threshold_operation=0.75,
    )
    print(f"accepted matches: {rescue_alignment}")
    assert len(rescue_alignment) == 1, (
        f"expected exactly 1 accepted match (A->X, the only pair clearing "
        f"threshold) -- if this is 0, the optimizer sacrificed the real "
        f"match to chase B's higher-but-still-doomed raw score "
        f"(raw sum 1.25 for A->Y+B->X vs 0.78 for A->X+B->Y), got "
        f"{rescue_alignment}"
    )
    assert rescue_alignment[0]["gen_id"] == "A" and rescue_alignment[0]["gt_id"] == "X", (
        f"expected A matched to X specifically, got {rescue_alignment}"
    )
    print("PASSED: the optimizer correctly keeps A's real match instead of "
          "chasing B's higher raw (but still sub-threshold) score.")

    print(
        "\n=== edge regression test 1: right shape, WRONG polarity -- must be "
        "GRADED (nonzero), not a hard 0 like before ==="
    )
    wrong_polarity_score = build_edge_similarity_matrix(
        [{"id": 1, "source": 1, "target": 2, "polarity": "negative",
          "edge_type": "can apply", "text_evidence": "x"}],
        [{"id": 1, "source": 1, "target": 2, "polarity": "positive",
          "edge_type": "can apply", "text_evidence": "x"}],
        {1: 1, 2: 2},
        {1: {"node_type": "rational"}, 2: {"node_type": "operation"}},
        {1: {"node_type": "rational"}, 2: {"node_type": "operation"}},
    )[0, 0]
    print(f"right nodes+type+strength, wrong polarity: score={wrong_polarity_score:.4f}")
    assert abs(wrong_polarity_score - (1 / 3 + 2 / 3 * 0.6)) < 1e-9, (
        f"expected 1/3 (evidence) + 2/3*0.6 (type+strength, no polarity) = 0.7333, "
        f"got {wrong_polarity_score}"
    )
    assert wrong_polarity_score > 0, "wrong polarity must no longer be a hard 0"
    print("PASSED: wrong polarity is graded, not gated.")

    print(
        "\n=== edge regression test 2: DIFFERENT specific node pair, same "
        "type/polarity/strength/evidence -- must still score 0 (exact node "
        "identity is a hard requirement, deliberately not relaxed) ==="
    )
    different_pair_score = build_edge_similarity_matrix(
        [{"id": 1, "source": 1, "target": 2, "polarity": "positive",
          "edge_type": "can apply", "text_evidence": "same wording"}],
        [{"id": 1, "source": 99, "target": 98, "polarity": "positive",
          "edge_type": "can apply", "text_evidence": "same wording"}],
        {1: 10, 2: 20},  # gen's nodes mapped to gt ids 10/20 -- NOT 99/98
        {1: {"node_type": "rational"}, 2: {"node_type": "operation"}},
        {99: {"node_type": "rational"}, 98: {"node_type": "operation"},
         10: {"node_type": "rational"}, 20: {"node_type": "operation"}},
    )[0, 0]
    print(f"different specific node pair, same shape+wording: score={different_pair_score:.4f}")
    assert different_pair_score == 0.0, (
        f"expected 0.0 -- identical type pattern, polarity, strength, and "
        f"evidence text is NOT enough; the ground-truth edge must connect "
        f"the exact same mapped node pair, got {different_pair_score}"
    )
    print("PASSED: identical shape+wording does NOT rescue a different specific node pair.")

    print(
        "\n=== edge regression test 3: the strength term is 1 for identical labels, 0 otherwise ==="
    )
    assert EDGE_STRENGTH_MODE == "label", EDGE_STRENGTH_MODE
    assert _strength_match("can apply", "can apply") == 1.0
    assert _strength_match("Can  Apply ", "can apply") == 1.0            # case and spacing are normalised
    assert _strength_match("can apply", "should apply") == 0.0
    assert _strength_match("can apply", "can not apply") == 0.0           # the negated label is a different label
    assert _strength_match("causes", "may lead to") == 0.0                # the keyword rule could not tell these apart
    assert _strength_match("mandatorily implies", "can lead to") == 0.0   # nor these
    assert _strength_match("", "") == 0.0 and _strength_match(None, "can apply") == 0.0
    print("PASSED: the strength term compares edge labels, like polarity.")

    print(
        "\n=== edge regression test 3b: the legacy keyword rule is still available (can/should/must) ==="
    )
    assert _classify_strength("can apply") == "can"
    assert _classify_strength("should apply") == "should"
    assert _classify_strength("must apply") == "must"
    assert _classify_strength("should not apply") == "should"
    assert _classify_strength("must not apply") == "must"
    assert _classify_strength("mandatorily implies") == "can"  # known imprecision of the legacy rule
    assert _classify_strength("causes") == "can"
    _saved_mode = EDGE_STRENGTH_MODE
    globals()["EDGE_STRENGTH_MODE"] = "keyword"
    try:
        assert _strength_match("causes", "may lead to") == 1.0
        assert _strength_match("can apply", "should apply") == 0.0
    finally:
        globals()["EDGE_STRENGTH_MODE"] = _saved_mode
    print("PASSED: the legacy keyword rule reproduces the earlier behaviour.")

    print(
        "\n=== edge regression test 4: cascade still applies -- an unmatched "
        "endpoint is still an automatic 0, unchanged ==="
    )
    cascade_score = build_edge_similarity_matrix(
        [{"id": 1, "source": 1, "target": 2, "polarity": "positive",
          "edge_type": "can apply", "text_evidence": "x"}],
        [{"id": 1, "source": 5, "target": 6, "polarity": "positive",
          "edge_type": "can apply", "text_evidence": "x"}],
        {1: None, 2: 6},  # source (node 1) was never matched during node alignment
        {1: {"node_type": "rational"}, 2: {"node_type": "operation"}},
        {5: {"node_type": "rational"}, 6: {"node_type": "operation"}},
    )[0, 0]
    assert cascade_score == 0.0, f"expected 0.0 for an unmatched endpoint, got {cascade_score}"
    print("PASSED: an orphaned endpoint is still an automatic 0.")

    print(
        "\n=== syntax validation test 1: a clean, well-formed graph must "
        "report zero violations ==="
    )
    clean_graph = {
        "nodes": [
            {"id": 1, "name": "Low-pass filter", "node_type": "operation", "parameters": [], "text_evidence": "x"},
            {"id": 2, "name": "remove noise", "node_type": "rational", "text_evidence": "y"},
        ],
        "edges": [
            {"id": 1, "source": 2, "target": 1, "polarity": "positive", "edge_type": "can apply", "text_evidence": "z"},
        ],
    }
    clean_report = validate_graph_syntax(clean_graph)
    assert clean_report["is_valid"] is True
    assert clean_report["violations"] == []
    print("PASSED: a syntactically clean graph reports is_valid=True with no violations.")

    print(
        "\n=== syntax validation test 2: dangling reference, self-loop, and "
        "duplicate id are all real ERRORS ==="
    )
    broken_graph = {
        "nodes": [
            {"id": 1, "name": "A", "node_type": "operation", "text_evidence": "x"},
            {"id": 1, "name": "A duplicate", "node_type": "rational", "text_evidence": "x"},
        ],
        "edges": [
            {"id": 1, "source": 1, "target": 99, "polarity": "positive", "edge_type": "can apply", "text_evidence": "x"},
            {"id": 2, "source": 1, "target": 1, "polarity": "positive", "edge_type": "can apply", "text_evidence": "x"},
        ],
    }
    broken_report = validate_graph_syntax(broken_graph)
    checks_found = {v["check"] for v in broken_report["violations"]}
    assert broken_report["is_valid"] is False
    assert "duplicate_node_id" in checks_found
    assert "dangling_edge_target" in checks_found
    assert "self_loop" in checks_found
    print(f"violations found: {sorted(checks_found)}")
    print("PASSED: dangling references, self-loops, and duplicate ids are all caught as errors.")

    print(
        "\n=== syntax validation test 3: edge_type outside the closed "
        "vocabulary for its shape is a WARNING, not a hard error -- real "
        "case from 03_batch's manual QC walkthrough: a rational->rational "
        "edge using \"should apply\" (rational->operation vocabulary) "
        "instead of \"causes\" ==="
    )
    misuse_graph = {
        "nodes": [
            {"id": 1, "name": "unavoidable muscle activity", "node_type": "rational", "text_evidence": "x"},
            {"id": 2, "name": "EMG contamination of ECG", "node_type": "rational", "text_evidence": "y"},
            {"id": 3, "name": "adaptive filter", "node_type": "operation", "text_evidence": "w"},
        ],
        "edges": [
            {"id": 1, "source": 1, "target": 2, "polarity": "positive", "edge_type": "should apply", "text_evidence": "z"},
            # node 3 anchors this component to an operation, isolating the
            # vocabulary-warning behavior from the separate island check
            {"id": 2, "source": 2, "target": 3, "polarity": "positive", "edge_type": "can apply", "text_evidence": "v"},
        ],
    }
    misuse_report = validate_graph_syntax(misuse_graph)
    misuse_checks = {v["check"] for v in misuse_report["violations"]}
    assert misuse_report["is_valid"] is True, "an off-vocabulary edge_type is a warning, not an error -- graph stays valid"
    assert "edge_type_outside_vocabulary" in misuse_checks
    print("PASSED: an off-vocabulary edge_type for its shape is flagged as a warning, structurally valid otherwise.")

    print(
        "\n=== syntax validation test 4: polarity contradicting its "
        "edge_type's actual vocabulary is a real ERROR -- \"must apply\" "
        "(positive-only vocabulary) declared with polarity=\"negative\" ==="
    )
    contradiction_graph = {
        "nodes": [
            {"id": 1, "name": "large number of NNIs", "node_type": "rational", "text_evidence": "x"},
            {"id": 2, "name": "HRV triangular index", "node_type": "operation", "text_evidence": "y"},
        ],
        "edges": [
            {"id": 1, "source": 1, "target": 2, "polarity": "negative", "edge_type": "must apply", "text_evidence": "z"},
        ],
    }
    contradiction_report = validate_graph_syntax(contradiction_graph)
    contradiction_checks = {v["check"] for v in contradiction_report["violations"]}
    assert contradiction_report["is_valid"] is False
    assert "polarity_edge_type_mismatch" in contradiction_checks
    print("PASSED: a polarity/edge_type contradiction is caught as an error.")

    print(
        "\n=== syntax validation test 5: an ISLAND of rational nodes with "
        "no operation reachable at all is a real ERROR -- rationale never "
        "tied to any operation ==="
    )
    island_graph = {
        "nodes": [
            {"id": 1, "name": "noise appears", "node_type": "rational", "text_evidence": "x"},
            {"id": 2, "name": "reduces diagnostic accuracy", "node_type": "rational", "text_evidence": "y"},
            {"id": 3, "name": "Low-pass filter", "node_type": "operation", "text_evidence": "z"},
            {"id": 4, "name": "remove noise", "node_type": "rational", "text_evidence": "w"},
        ],
        "edges": [
            # 1 -> 2: a rational->rational island, NEVER connected to node 3
            {"id": 1, "source": 1, "target": 2, "polarity": "positive", "edge_type": "causes", "text_evidence": "a"},
            # 4 -> 3: correctly anchored to an operation
            {"id": 2, "source": 4, "target": 3, "polarity": "positive", "edge_type": "can apply", "text_evidence": "b"},
        ],
    }
    island_report = validate_graph_syntax(island_graph)
    island_violations = [v for v in island_report["violations"] if v["check"] == "isolated_rational_island"]
    assert island_report["is_valid"] is False
    assert len(island_violations) == 1, f"expected exactly 1 island violation, got {len(island_violations)}"
    assert set(island_violations[0]["node_ids"]) == {1, 2}, (
        f"expected the island to be exactly nodes {{1, 2}}, got {island_violations[0]['node_ids']}"
    )
    print(f"island violation: {island_violations[0]['message']}")
    print("PASSED: a rational-only connected component with no operation reachable is caught as an error, "
          "while the properly-anchored rational (node 4) is not flagged.")

    print(
        "\n=== syntax validation test 6: a single UNCONNECTED rational "
        "node (no edges at all) is the size-1 case of the same island "
        "problem ==="
    )
    lonely_graph = {
        "nodes": [
            {"id": 1, "name": "background context, never used", "node_type": "rational", "text_evidence": "x"},
            {"id": 2, "name": "SDNN", "node_type": "operation", "text_evidence": "y"},
        ],
        "edges": [],
    }
    lonely_report = validate_graph_syntax(lonely_graph)
    lonely_islands = [v for v in lonely_report["violations"] if v["check"] == "isolated_rational_island"]
    assert lonely_report["is_valid"] is False
    assert len(lonely_islands) == 1 and lonely_islands[0]["node_ids"] == [1]
    print("PASSED: a completely disconnected rational node is caught as a size-1 island.")
