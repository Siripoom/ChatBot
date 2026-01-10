#!/usr/bin/env python3
"""
Quick test script to verify evaluation setup before running full evaluation
"""

import os
from dotenv import load_dotenv

def test_setup():
    """Test if all required components are available"""

    print("=" * 60)
    print("🔍 Testing RAG Evaluation Setup")
    print("=" * 60)

    # Test 1: Check environment variables
    print("\n1️⃣  Checking environment variables...")
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')

    if api_key:
        print(f"   ✅ GEMINI_API_KEY found (length: {len(api_key)} chars)")
    else:
        print("   ❌ GEMINI_API_KEY not found")
        print("   💡 Create a .env file with: GEMINI_API_KEY=your_key")
        return False

    # Test 2: Check imports
    print("\n2️⃣  Checking required imports...")
    try:
        from datasets import Dataset
        print("   ✅ datasets package imported")
    except ImportError as e:
        print(f"   ❌ datasets package failed: {e}")
        return False

    try:
        from ragas import evaluate
        from ragas.metrics.collections import (
            AnswerRelevancy,
            Faithfulness,
            ContextRecall,
            ContextPrecision,
        )
        print("   ✅ ragas package imported")
    except ImportError as e:
        print(f"   ❌ ragas package failed: {e}")
        return False

    try:
        from chatbot_v04_keywords import HybridKnowledgeBase, TyphoonChatbot
        print("   ✅ chatbot modules imported")
    except ImportError as e:
        print(f"   ❌ chatbot modules failed: {e}")
        return False

    # Test 3: Check if knowledge base files exist
    print("\n3️⃣  Checking knowledge base files...")
    kb_files = [
        "chroma_db",
        "knowledge_base.json",
    ]

    found_kb = False
    for kb_file in kb_files:
        if os.path.exists(kb_file):
            print(f"   ✅ Found: {kb_file}")
            found_kb = True

    if not found_kb:
        print("   ⚠️  No knowledge base files found")
        print("   💡 You may need to initialize the knowledge base first")

    # Test 4: Try to initialize components
    print("\n4️⃣  Testing component initialization...")
    try:
        kb = HybridKnowledgeBase(use_reranker=False)
        print("   ✅ HybridKnowledgeBase initialized")
    except Exception as e:
        print(f"   ❌ HybridKnowledgeBase failed: {e}")
        return False

    try:
        chatbot = TyphoonChatbot(api_key, kb, use_compression=False)
        print("   ✅ TyphoonChatbot initialized")
    except Exception as e:
        print(f"   ❌ TyphoonChatbot failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ All checks passed! Ready to run evaluation.")
    print("=" * 60)
    print("\n💡 Run: python3 evaluate_rag.py")

    return True

if __name__ == "__main__":
    success = test_setup()
    exit(0 if success else 1)
