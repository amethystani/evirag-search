#!/usr/bin/env python3
"""
EVIRAG Quick Start
Automated setup and quick testing
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║  ███████╗██╗   ██╗██╗██████╗  █████╗  ██████╗                          ║
║  ██╔════╝██║   ██║██║██╔══██╗██╔══██╗██╔════╝                          ║
║  █████╗  ██║   ██║██║██████╔╝███████║██║  ███╗                         ║
║  ██╔══╝  ╚██╗ ██╔╝██║██╔══██╗██╔══██║██║   ██║                         ║
║  ███████╗ ╚████╔╝ ██║██║  ██║██║  ██║╚██████╔╝                         ║
║  ╚══════╝  ╚═══╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝                          ║
║                                                                          ║
║     Evidence-Centric, Disagreement-Aware RAG for Scientific Lit         ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

def check_ollama():
    """Check if Ollama is running"""
    print("\n[1/5] Checking Ollama...")
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ Ollama is running")
            return True
        else:
            print("✗ Ollama is not running")
            print("  Start with: ollama serve")
            return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False

def check_models():
    """Check if required models are available"""
    print("\n[2/5] Checking models...")
    
    required_models = [
        "phi3:mini",
        "llama3:latest",
        "qwen2.5:4b",
        "deepseek-r1:7b"
    ]
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        
        available = result.stdout
        missing = []
        
        for model in required_models:
            if model in available:
                print(f"  ✓ {model}")
            else:
                print(f"  ✗ {model} (missing)")
                missing.append(model)
        
        if missing:
            print(f"\n  Pull missing models with:")
            for model in missing:
                print(f"    ollama pull {model}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking models: {e}")
        return False

def check_corpus():
    """Check if corpus directory has PDFs"""
    print("\n[3/5] Checking corpus...")
    
    corpus_path = Path(__file__).parent / "corpus"
    
    if not corpus_path.exists():
        print("✗ Corpus directory doesn't exist")
        print(f"  Create with: mkdir -p {corpus_path}")
        return False
    
    pdf_files = list(corpus_path.glob("*.pdf"))
    
    if not pdf_files:
        print("✗ No PDF files in corpus/")
        print("  Add PDFs with: cp /path/to/papers/*.pdf corpus/")
        return False
    
    print(f"✓ Found {len(pdf_files)} PDF files")
    return True

def install_dependencies():
    """Check/install Python dependencies"""
    print("\n[4/5] Checking Python dependencies...")
    
    try:
        import sentence_transformers
        import streamlit
        import networkx
        import faiss
        print("✓ All dependencies installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("\n  Install with:")
        print("    pip install -r requirements.txt --break-system-packages")
        return False

def run_quick_test():
    """Run a quick test query"""
    print("\n[5/5] Running quick test...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
        from evirag_system import EVIRAGSystem, EVIRAGConfig
        
        print("  Initializing system...")
        config = EVIRAGConfig(mode="vanilla_rag")  # Use fastest mode for test
        system = EVIRAGSystem(config)
        
        print("  Processing corpus...")
        system.initialize_corpus()
        
        print("  Running test query...")
        result = system.query("What is machine learning?")
        
        print(f"\n✓ Test completed!")
        print(f"  Mode: {result['mode']}")
        print(f"  Sources: {len(result.get('sources', []))}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print_banner()
    
    print("\nEVIRAG Quick Start - Checking system readiness...\n")
    print("="*76)
    
    checks = [
        ("Ollama Service", check_ollama),
        ("Ollama Models", check_models),
        ("Corpus Data", check_corpus),
        ("Python Deps", install_dependencies),
    ]
    
    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False
    
    print("\n" + "="*76)
    
    if not all_passed:
        print("\n⚠  Some checks failed. Please fix the issues above.")
        print("\nFor detailed setup instructions, see SETUP.md")
        sys.exit(1)
    
    print("\n✓ All checks passed!")
    
    # Ask if user wants to run test
    print("\n" + "="*76)
    response = input("\nRun quick test query? (y/n): ").strip().lower()
    
    if response == 'y':
        if run_quick_test():
            print("\n" + "="*76)
            print("\n✓ EVIRAG is ready!")
            print("\nNext steps:")
            print("  • Streamlit UI:  streamlit run ui/streamlit_app.py")
            print("  • CLI mode:      python3 run.py --mode cli")
            print("  • API service:   python3 orchestration/fastapi_service.py")
            print("\nSee SETUP.md for detailed usage instructions")
        else:
            print("\n⚠  Test failed. Check error messages above.")
    else:
        print("\n✓ System ready!")
        print("\nRun: python3 run.py --mode ui")

if __name__ == "__main__":
    main()
