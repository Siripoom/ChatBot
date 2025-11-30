#!/usr/bin/env python3
"""
Test script for the vector database chatbot
"""

import os
from dotenv import load_dotenv
from chatbot import VectorKnowledgeBase, GeminiChatbot

def test_chatbot():
    """Test the chatbot with sample queries"""
    print("=" * 80)
    print("🧪 Testing Vector Database Chatbot")
    print("=" * 80)

    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print("❌ ไม่พบ GEMINI_API_KEY")
        return

    # Initialize knowledge base and chatbot
    print("\n📦 Initializing components...")
    kb = VectorKnowledgeBase(persist_directory="./chroma_db", collection_name="chatbot_knowledge")
    chatbot = GeminiChatbot(api_key, kb)

    # Test queries
    test_queries = [
        "หลักสูตรวิศวกรรมโยธาและการศึกษามีกี่ปี",
        "วิชาที่ต้องเรียนในหลักสูตรมีอะไรบ้าง",
        "การสำรวจมีเนื้อหาอย่างไร"
    ]

    print("\n" + "=" * 80)
    print("🔍 Running test queries")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print("=" * 80)

        # Test vector search
        print("\n📊 Vector search results:")
        results = kb.search_knowledge(query, n_results=3)
        for result in results:
            print(f"\n  Rank {result['rank']} (Distance: {result['distance']:.4f})")
            print(f"  Text preview: {result['text'][:200]}...")

        # Test chatbot response
        print(f"\n💬 Chatbot response:")
        print("-" * 80)
        response = chatbot.chat(query)
        print(response)
        print("-" * 80)

    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_chatbot()
