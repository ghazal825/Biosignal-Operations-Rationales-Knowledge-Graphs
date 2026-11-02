#!/usr/bin/env python3
"""
cross_validate.py -- leave-one-out cross-validation over the ICL worked
examples, sweeping reasoning effort, with full HTTP logging and cost
tracking.
==============================================================================
USAGE
-----
    # Pilot: 1 fold x 1 reasoning effort, to sanity-check the harness
    python cross_validate.py --pilot

    # Full sweep, default model (gpt-6-astra)
    python cross_validate.py

    # Full sweep, explicit model override
    python cross_validate.py --model gpt-5.6-sol

Requirements: pip install openai networkx scipy numpy
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from icl_pipeline_extractor import (  # noqa: E402
    OpenAIBackend, call_llm, build_messages, parse_llm_json,
    read_text_file, read_json_file, discover_examples_in_dir,
    summarize_graph, DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_TEMPERATURE,
)
from graph_eval import (  # noqa: E402
    evaluate_graph_function, DEFAULT_NODE_MATCH_THRESHOLD_RATIONAL, DEFAULT_NODE_MATCH_THRESHOLD_OPERATION,
)


# ============================================================
# PRICING (USD per 1,000,000 tokens) -- confirmed from OpenAI's own
# GPT-5.6 launch announcement (openai.com/index/previewing-gpt-5-6-sol,
# June 2026): Sol $5/$30, Terra $2.50/$15, Luna $1/$6 (input/output).
# GPT-6 Astra (announced 2026-09-03, developers.openai.com/api/docs/
# models/gpt-6-astra): $10/$50 standard rate (cached input $1, batch/
# flex processing at half price, Fast mode at double -- none of those
# tiers are modeled here, this is the standard-rate estimate only).
# Verify against platform.openai.com/pricing before relying on this for
# anything beyond a rough estimate -- pricing can change.
# ============================================================

PRICING_PER_1M_TOKENS = {
    "gpt-5.6-sol": {"input": 5.00, "output": 30.00},
    "gpt-5.6-terra": {"input": 2.50, "output": 15.00},
    "gpt-5.6-luna": {"input": 1.00, "output": 6.00},
    "gpt-6-astra": {"input": 10.00, "output": 50.00},
}

DEFAULT_REASONING_EFFORTS = [
    # "medium",  # commented out for now -- running one effort only per request
    # "high",    # commented out for now -- running one effort only per request
    # gpt-6-astra rejects "max" on Chat Completions (supported: low/medium/high/xhigh),
    # so "xhigh" is the highest effort this backend can request.
    "xhigh",
]


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int,
                       verbose: bool = False) -> float:

    if verbose:
        print(f"      [estimate_cost_usd] model={model}  prompt_tokens={prompt_tokens}  "
              f"completion_tokens={completion_tokens}")
    pricing = PRICING_PER_1M_TOKENS.get(model)
    if pricing is None:
        if verbose:
            print(f"      [estimate_cost_usd] -> model not in pricing table, cost=None")
        return None
    cost = (prompt_tokens / 1_000_000) * pricing["input"] + \
           (completion_tokens / 1_000_000) * pricing["output"]
    if verbose:
        print(f"      [estimate_cost_usd] -> ${cost:.5f}")
    return cost


# ============================================================
# CROSS-VALIDATION LOOP
# ============================================================

def run_cross_validation(examples_dir: str, model: str, reasoning_efforts: list,
                          output_dir: str, api_key: str, max_tokens: int,
                          temperature: float, node_match_threshold_rational: float,
                          node_match_threshold_operation: float,
                          ged_timeout_sec: float, compute_ged: bool,
                          pilot: bool, verbose: bool,
                          system_prompt: str = None,
                          parameter_pattern: bool = True) -> list:
    if verbose:
        print(f"[run_cross_validation] discovering example pairs in '{examples_dir}' ...")
    all_pairs = discover_examples_in_dir(examples_dir, verbose=verbose)
    if len(all_pairs) < 2:
        print(f"ERROR: need at least 2 example pairs in '{examples_dir}' to "
              f"hold one out and still have examples left; found {len(all_pairs)}.",
              file=sys.stderr)
        sys.exit(1)

    if pilot:
        all_pairs = all_pairs[:1]
        reasoning_efforts = reasoning_efforts[:1]
        if verbose:
            print(f"PILOT MODE: 1 fold x 1 reasoning effort only "
                  f"({Path(all_pairs[0][0]).stem}, effort={reasoning_efforts[0]})")

    if verbose:
        print(f"[run_cross_validation] {len(all_pairs)} fold(s) x "
              f"{len(reasoning_efforts)} reasoning effort(s) "
              f"= {len(all_pairs) * len(reasoning_efforts)} total run(s)")

    http_log_dir = Path(output_dir) / "http_logs"
    graphs_dir = Path(output_dir) / "generated_graphs"
    http_log_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[run_cross_validation] output dirs ready: {http_log_dir}, {graphs_dir}")

    rows = []
    full_pairs = discover_examples_in_dir(examples_dir, verbose=False)  # full set, for building "other 3"

    for fold_idx, held_out in enumerate(all_pairs, 1):
        held_out_text_path, held_out_json_path = held_out
        fold_stem = Path(held_out_text_path).stem
        other_pairs = [p for p in full_pairs if p != held_out]

        if verbose:
            print(f"\n[run_cross_validation] --- fold {fold_idx}/{len(all_pairs)}: "
                  f"loading held-out pair for '{fold_stem}' ---")
        ground_truth = read_json_file(held_out_json_path, verbose=verbose)
        query_text = read_text_file(held_out_text_path, verbose=verbose)

        if verbose:
            print(f"\n=== Fold {fold_idx}/{len(all_pairs)}: held out '{fold_stem}' "
                  f"({summarize_graph(ground_truth)}) -- "
                  f"{len(other_pairs)} example(s) in context ===")

        messages = build_messages(other_pairs, query_text, verbose=False,
                                  system_prompt=system_prompt)

        for effort in reasoning_efforts:
            if verbose:
                print(f"  -- reasoning_effort={effort} --")

            # "default" = send no reasoning_effort at all (the model's own default)
            backend = OpenAIBackend(model=model, api_key=api_key, strict_schema=True,
                                     reasoning_effort=None if effort == "default" else effort,
                                     verbose=verbose, parameter_pattern=parameter_pattern)
            capture = {}
            row = {
                "fold": fold_stem, "reasoning_effort": effort, "model": model,
                "status": "ok",
            }

            wall_start = time.perf_counter()
            try:
                raw = call_llm(backend, messages, max_tokens, temperature,
                                context=f"{fold_stem}/{effort}", verbose=verbose,
                                capture=capture)
                generated = parse_llm_json(raw, context=f"{fold_stem}/{effort}")
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                print(f"    FAILED: {exc}", file=sys.stderr)
                rows.append(row)
                # still save whatever was captured (e.g. a failed request with
                # no response) so the failure itself is inspectable later
                log_path = http_log_dir / f"{fold_stem}__{effort}.json"
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(capture, f, indent=2, ensure_ascii=False, default=str)
                continue
            wall_elapsed = time.perf_counter() - wall_start
            if verbose:
                print(f"    [run_cross_validation] LLM call done in {wall_elapsed:.2f}s "
                      f"-> generated {summarize_graph(generated)}")

            # -- save raw HTTP request/response --
            log_path = http_log_dir / f"{fold_stem}__{effort}.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(capture, f, indent=2, ensure_ascii=False, default=str)
            if verbose:
                print(f"    [run_cross_validation] saved HTTP log -> {log_path}")

            # -- save generated graph --
            graph_path = graphs_dir / f"{fold_stem}__{effort}_graph.json"
            with open(graph_path, "w", encoding="utf-8") as f:
                json.dump(generated, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"    [run_cross_validation] saved generated graph -> {graph_path}")

            # -- evaluate against held-out ground truth --
            if verbose:
                print(f"    [run_cross_validation] scoring against ground truth "
                      f"(node_match_threshold_rational={node_match_threshold_rational}, "
                      f"node_match_threshold_operation={node_match_threshold_operation}, "
                      f"compute_ged={compute_ged}) ...")
            if verbose and compute_ged:
                print(f"    (computing GED, hard-capped at {ged_timeout_sec}s "
                      f"in a subprocess -- may report 'skipped' for larger graphs)")
            metrics = evaluate_graph_function(
                generated, ground_truth,
                node_match_threshold_rational=node_match_threshold_rational,
                node_match_threshold_operation=node_match_threshold_operation,
                ged_timeout_sec=ged_timeout_sec,
                compute_ged=compute_ged,
            )
            if verbose:
                print(f"    [run_cross_validation] -> node_f1={metrics['node_f1']:.4f}  "
                      f"edge_f1={metrics['edge_f1']:.4f}")

            usage = capture.get("response", {}).get("usage", {}) or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cost = estimate_cost_usd(model, prompt_tokens, completion_tokens, verbose=verbose)

            row.update({
                "node_precision": metrics["node_precision"],
                "node_recall": metrics["node_recall"],
                "node_f1": metrics["node_f1"],
                "edge_precision": metrics["edge_precision"],
                "edge_recall": metrics["edge_recall"],
                "edge_f1": metrics["edge_f1"],
                "mean_edge_score_on_matches": metrics.get("mean_edge_score_on_matches"),
                "ged": metrics.get("ged"),
                "ged_is_exact": metrics.get("ged_is_exact"),
                "n_generated_nodes": metrics["n_generated_nodes"],
                "n_ground_truth_nodes": metrics["n_ground_truth_nodes"],
                "n_generated_edges": metrics["n_generated_edges"],
                "n_ground_truth_edges": metrics["n_ground_truth_edges"],
                "latency_sec": round(capture.get("latency_sec", wall_elapsed), 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": round(cost, 5) if cost is not None else None,
            })
            rows.append(row)

            if verbose:
                print(f"    node F1={row['node_f1']}  edge F1={row['edge_f1']}  "
                      f"GED={row['ged']}  "
                      f"latency={row['latency_sec']}s  cost=${row['cost_usd']}")

    if verbose:
        n_ok = sum(1 for r in rows if r.get("status") == "ok")
        n_err = len(rows) - n_ok
        print(f"\n[run_cross_validation] -> done: {len(rows)} row(s) total "
              f"({n_ok} ok, {n_err} error)")
    return rows


def save_summary(rows: list, output_dir: str, verbose: bool = False):
    if verbose:
        print(f"[save_summary] writing {len(rows)} row(s) to '{output_dir}' ...")
    csv_path = Path(output_dir) / "cross_validation_summary.csv"
    json_path = Path(output_dir) / "cross_validation_summary.json"

    fieldnames = [
        "fold", "reasoning_effort", "model", "status", "error",
        "node_precision", "node_recall", "node_f1",
        "edge_precision", "edge_recall", "edge_f1", "mean_edge_score_on_matches",
        "ged", "ged_is_exact",
        "n_generated_nodes", "n_ground_truth_nodes",
        "n_generated_edges", "n_ground_truth_edges",
        "latency_sec", "prompt_tokens", "completion_tokens", "cost_usd",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    if verbose:
        print(f"[save_summary] -> wrote {csv_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[save_summary] -> wrote {json_path}")

    return csv_path, json_path


def print_table(rows: list, verbose: bool = False):
    if verbose:
        print(f"[print_table] rendering summary table for {len(rows)} row(s) ...")
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        print("\nNo successful runs to summarize.")
        return
    header = (f"{'fold':<40} {'effort':<8} {'node F1':>8} {'edge F1':>8} "
              f"{'GED':>6} {'sec':>7} {'cost$':>8}")
    print("\n" + header)
    print("-" * len(header))
    for r in ok_rows:
        ged_display = r.get("ged")
        ged_display = f"{ged_display:.1f}" if ged_display is not None else "n/a"
        print(f"{r['fold']:<40} {r['reasoning_effort']:<8} {r['node_f1']:>8.3f} "
              f"{r['edge_f1']:>8.3f} "
              f"{ged_display:>6} {r['latency_sec']:>7.2f} "
              f"{('%.5f' % r['cost_usd']) if r['cost_usd'] is not None else 'n/a':>8}")
    n_errors = len(rows) - len(ok_rows)
    if n_errors:
        print(f"\n{n_errors} run(s) failed -- see cross_validation_summary.json for details.")
    total_cost = sum(r["cost_usd"] for r in ok_rows if r.get("cost_usd") is not None)
    print(f"\nTotal estimated cost across {len(ok_rows)} successful call(s): ${total_cost:.5f}")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Leave-one-out cross-validation over ICL worked examples, "
                    "sweeping reasoning effort, with HTTP logging and cost tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--examples-dir", default="icl_examples",
                        help="Folder of worked example pairs (default: icl_examples/).")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--reasoning-efforts", nargs="+",
                        default=DEFAULT_REASONING_EFFORTS,
                        choices=["default", "none", "low", "medium", "high", "xhigh", "max"],
                        help=f"Default: {DEFAULT_REASONING_EFFORTS}. 'default' sends no "
                             f"reasoning_effort (the model's own default effort).")
    parser.add_argument("--system-prompt-file", default=None,
                        help="Read the system prompt from this UTF-8 file instead of "
                             "the built-in SYSTEM_PROMPT (for prompt ablations). The "
                             "file's text is used verbatim. Use a separate --output-dir "
                             "so the run does not overwrite other results.")
    parser.add_argument("--no-parameter-pattern", action="store_true",
                        help="Do not require parameter strings to match 'key: value' in the "
                             "output schema (the schema of runs A-G of the thesis; use it to "
                             "reproduce them). Default: the pattern is enforced.")
    parser.add_argument("--output-dir", default="cross_validation_results")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--node-match-threshold-rational", type=float, default=DEFAULT_NODE_MATCH_THRESHOLD_RATIONAL)
    parser.add_argument("--node-match-threshold-operation", type=float, default=DEFAULT_NODE_MATCH_THRESHOLD_OPERATION)
    parser.add_argument("--compute-ged", action="store_true",
                        help="Also attempt an approximate Graph Edit Distance per "
                             "run, in a killable subprocess hard-capped at "
                             "--ged-timeout-sec. OFF by default: in practice this "
                             "search was found to be intractable even for this "
                             "schema's typical graph sizes (it ran over an hour on "
                             "a 27-node graph without finishing), so expect "
                             "ged=None (skipped) most of the time even when "
                             "enabled. The node/edge F1 scores are the metric "
                             "actually used for cross-validation scoring.")
    parser.add_argument("--ged-timeout-sec", type=float, default=15.0)
    parser.add_argument("--pilot", action="store_true",
                        help="Run only the first fold x first reasoning effort "
                             "(1 API call total) to sanity-check the harness "
                             "before committing to the full sweep.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: no OpenAI API key found. Set OPENAI_API_KEY or pass --api-key.",
              file=sys.stderr)
        sys.exit(1)

    system_prompt = None
    if args.system_prompt_file:
        with open(args.system_prompt_file, encoding="utf-8", newline="") as f:
            system_prompt = f.read()

    if verbose:
        print(f"Model              : {args.model}")
        print(f"Reasoning efforts  : {args.reasoning_efforts}")
        print(f"System prompt      : "
              + (f"{args.system_prompt_file} ({len(system_prompt)} chars)"
                 if system_prompt is not None else "built-in SYSTEM_PROMPT"))
        print(f"Examples dir       : {args.examples_dir}")
        print(f"Output dir         : {args.output_dir}")
        print(f"Node match threshold (rational): {args.node_match_threshold_rational}")
        print(f"Node match threshold (operation): {args.node_match_threshold_operation}")
        print(f"Compute GED        : {args.compute_ged}"
              + ("" if args.compute_ged else "  (off by default -- see --help)"))
        if args.pilot:
            print("Mode               : PILOT (1 call only)")

    if verbose:
        print("[main] starting cross-validation run ...")

    rows = run_cross_validation(
        examples_dir=args.examples_dir, model=args.model,
        reasoning_efforts=args.reasoning_efforts, output_dir=args.output_dir,
        api_key=api_key, max_tokens=args.max_tokens, temperature=args.temperature,
        node_match_threshold_rational=args.node_match_threshold_rational,
        node_match_threshold_operation=args.node_match_threshold_operation,
        ged_timeout_sec=args.ged_timeout_sec, compute_ged=args.compute_ged,
        pilot=args.pilot, verbose=verbose, system_prompt=system_prompt,
        parameter_pattern=not args.no_parameter_pattern,
    )

    if verbose:
        print("[main] cross-validation run complete, saving summary ...")
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path, json_path = save_summary(rows, args.output_dir, verbose=verbose)
    print_table(rows, verbose=verbose)
    print(f"\nSummary saved: {csv_path}")
    print(f"Full detail saved: {json_path}")
    print(f"HTTP request/response logs: {Path(args.output_dir) / 'http_logs'}")
    print(f"Generated graphs: {Path(args.output_dir) / 'generated_graphs'}")
    if verbose:
        print("[main] -> done")


if __name__ == "__main__":
    main()
