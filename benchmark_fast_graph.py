#!/usr/bin/env python3
"""
Rebuild and benchmark the EVIRAG fast claim-graph path.
"""

import argparse
import time

from config import CLAIM_GRAPH_CONFIG
from evirag_system import EVIRAGConfig, EVIRAGSystem


def main():
    parser = argparse.ArgumentParser(description="Benchmark EVIRAG fast claim graph mode")
    parser.add_argument(
        "--query",
        default="Does homework improve academic achievement?",
        help="Query to benchmark",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector index and claim graph",
    )
    parser.add_argument(
        "--verify",
        choices=["auto", "none", "local", "cloud"],
        default=CLAIM_GRAPH_CONFIG.get("verify_edges", "auto"),
        help="Offline edge verification backend",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Override verification budget for the chosen backend",
    )
    args = parser.parse_args()

    CLAIM_GRAPH_CONFIG["verify_edges"] = args.verify
    if args.budget is not None:
        if args.verify == "cloud":
            CLAIM_GRAPH_CONFIG["verification_budget_cloud"] = args.budget
        elif args.verify == "local":
            CLAIM_GRAPH_CONFIG["verification_budget_local"] = args.budget

    config = EVIRAGConfig(
        mode="evirag",
        backend="local",
        depth_vs_speed="fast",
        use_visual_grounding=False,
        use_causal_attribution=False,
        rebuild_index=args.rebuild,
    )

    system = EVIRAGSystem(config)

    t0 = time.time()
    system.initialize_corpus(rebuild=args.rebuild)
    init_elapsed = time.time() - t0

    t1 = time.time()
    result1 = system.query(args.query)
    first_elapsed = time.time() - t1

    t2 = time.time()
    result2 = system.query(args.query)
    second_elapsed = time.time() - t2

    status = system.get_system_status()

    print("\n=== FAST GRAPH BENCHMARK ===")
    print(
        {
            "initialize_seconds": round(init_elapsed, 3),
            "first_query_seconds": round(first_elapsed, 3),
            "second_query_seconds": round(second_elapsed, 3),
            "query": args.query,
            "graph_ready": status["corpus"]["claim_graph_ready"],
            "num_claims": status["corpus"]["num_claims"],
        }
    )
    print("\n=== FIRST ANSWER ===")
    print(result1["answer"]["overall_confidence"], result1["answer"]["confidence_score"])
    dominant = result1["answer"].get("dominant_view") or {}
    print(dominant.get("summary", ""))
    print("\n=== STATS ===")
    print(result1["statistics"])
    if result2.get("epistemic_divergence"):
        print("\n=== EPISTEMIC DIVERGENCE ===")
        print(result2["epistemic_divergence"])


if __name__ == "__main__":
    main()
