#!/usr/bin/env python3
"""
EVIRAG Run Script
Quick start script for EVIRAG system
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from evirag_system import EVIRAGSystem, EVIRAGConfig


def cli_demo():
    """Run a CLI demo of EVIRAG"""
    print("="*70)
    print(" EVIRAG: Evidence-Centric, Disagreement-Aware RAG")
    print("="*70)
    
    # Initialize system
    print("\nInitializing system...")
    config = EVIRAGConfig(
        mode="evirag",
        use_visual_grounding=True,
        depth_vs_speed="balanced",
        rebuild_index=False
    )
    
    system = EVIRAGSystem(config)
    
    # Initialize corpus
    print("Processing corpus...")
    system.initialize_corpus()
    
    # Interactive query loop
    print("\n" + "="*70)
    print("System ready! Enter your scientific questions.")
    print("Type 'exit' to quit, 'mode' to change execution mode")
    print("="*70 + "\n")
    
    while True:
        try:
            query = input("\n📚 Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'exit':
                print("Goodbye!")
                break
            
            if query.lower() == 'mode':
                print("\nAvailable modes:")
                print("1. vanilla_rag - Standard RAG")
                print("2. evirag - Full pipeline (default)")
                
                mode = input("Select mode (1-2): ").strip()
                mode_map = {'1': 'vanilla_rag', '2': 'evirag'}
                config.mode = mode_map.get(mode, 'evirag')
                print(f"✓ Mode set to: {config.mode}")
                continue
            
            # Process query
            print(f"\n🔍 Processing in {config.mode} mode...\n")
            result = system.query(query, config)
            
            # Display results
            print_results(result)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def print_results(result):
    """Print query results to console"""
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    
    mode = result.get("mode", "unknown")
    print(f"\nMode: {mode.upper()}")
    
    if mode == "vanilla_rag":
        print(f"\nAnswer:\n{result.get('answer', '')}")
        print(f"\nSources: {len(result.get('sources', []))}")
        
    else:
        answer = result.get("answer", {})
        
        # Confidence
        confidence = answer.get("overall_confidence", "unknown")
        score = answer.get("confidence_score", 0.0)
        print(f"\n{'='*70}")
        print(f" CONFIDENCE: {confidence.upper()} ({score:.3f})")
        print(f"{'='*70}")
        print(answer.get("confidence_reasoning", ""))
        
        # Dominant view
        dominant = answer.get("dominant_view")
        if dominant:
            print(f"\n{'='*70}")
            print(" DOMINANT VIEW")
            print(f"{'='*70}")
            print(f"\n{dominant.get('summary', '')}")
            print(f"\nSupporting Claims: {len(dominant.get('supporting_claim_ids', []))}")
            print(f"Sources: {dominant.get('source_count', 0)}")
            
            weaknesses = dominant.get('weaknesses', [])
            if weaknesses:
                print("\nLimitations:")
                for w in weaknesses:
                    print(f"  • {w}")
        
        # Alternative views
        alt_views = answer.get("alternative_views", [])
        if alt_views:
            print(f"\n{'='*70}")
            print(f" ALTERNATIVE VIEWS ({len(alt_views)})")
            print(f"{'='*70}")
            for i, view in enumerate(alt_views, 1):
                print(f"\n{i}. {view.get('name', 'Unknown')}")
                print(f"   {view.get('summary', '')[:200]}...")
        
        # Statistics
        if "statistics" in result:
            stats = result["statistics"]
            print(f"\n{'='*70}")
            print(" STATISTICS")
            print(f"{'='*70}")
            print(f"Total Claims: {stats.get('total_claims', 0)}")
            print(f"Total Relationships: {stats.get('total_relationships', 0)}")
            print(f"Agents Used: {stats.get('agents_used', 0)}")
        
        # Metrics
        if "metrics" in result:
            metrics = result["metrics"]
            print(f"\n{'='*70}")
            print(" DISAGREEMENT METRICS")
            print(f"{'='*70}")
            print(f"Disagreement Density: {metrics.get('disagreement_density', 0):.3f}")
            print(f"Conflict Ratio: {metrics.get('conflict_ratio', 0):.3f}")
            print(f"Claim Entropy: {metrics.get('claim_entropy', 0):.3f}")
    
    print("\n" + "="*70 + "\n")


def run_streamlit():
    """Launch Streamlit UI"""
    import subprocess

    ui_path = Path(__file__).parent / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])


def run_api():
    """Launch FastAPI service."""
    from fastapi_service import run_server
    run_server()


def run_full():
    """Launch FastAPI + Streamlit together and wait for both."""
    import subprocess
    import time
    import os

    project_root = Path(__file__).parent
    ui_path = project_root / "streamlit_app.py"
    env = {**os.environ, "EVIRAG_API_URL": "http://localhost:8000"}

    print("Starting EVIRAG backend (FastAPI) on http://localhost:8000 ...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "fastapi_service:app",
         "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(project_root),
        env=env,
    )

    # Give the API a few seconds to bind before launching the UI
    time.sleep(3)

    print("Starting EVIRAG frontend (Streamlit) on http://localhost:8501 ...")
    ui_proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ui_path),
         "--server.port", "8501",
         "--server.address", "0.0.0.0"],
        cwd=str(project_root),
        env=env,
    )

    print("\nBoth services running. Press Ctrl+C to stop.\n")
    try:
        api_proc.wait()
        ui_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        ui_proc.terminate()
        api_proc.terminate()
        ui_proc.wait()
        api_proc.wait()
        print("Stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="EVIRAG: Evidence-Centric, Disagreement-Aware RAG"
    )

    parser.add_argument(
        "--mode",
        choices=["cli", "ui", "api", "full"],
        default="cli",
        help="Run mode: cli, ui (Streamlit), api (FastAPI), or full (both)"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query (non-interactive mode)"
    )
    
    parser.add_argument(
        "--exec-mode",
        choices=["vanilla_rag", "evirag"],
        default="evirag",
        help="Execution mode"
    )
    
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild corpus index"
    )

    parser.add_argument(
        "--speed",
        choices=["fast", "balanced", "deep"],
        default="balanced",
        help="Latency/quality tradeoff"
    )
    
    args = parser.parse_args()
    
    if args.mode == "full":
        run_full()
    elif args.mode == "ui":
        run_streamlit()
    elif args.mode == "api":
        run_api()
    elif args.query:
        # Single query mode
        config = EVIRAGConfig(
            mode=args.exec_mode,
            rebuild_index=args.rebuild,
            depth_vs_speed=args.speed,
        )
        system = EVIRAGSystem(config)
        system.initialize_corpus()
        
        result = system.query(args.query, config)
        print_results(result)
    else:
        # Interactive CLI
        cli_demo()


if __name__ == "__main__":
    main()
