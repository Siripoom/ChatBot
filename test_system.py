#!/usr/bin/env python3
"""
ทดสอบระบบ Chatbot
"""

import os
from dotenv import load_dotenv
from chatbot import HybridKnowledgeBase, GeminiChatbot

def test_search():
    """ทดสอบการค้นหา"""
    print("="*80)
    print("🔍 ทดสอบการค้นหาใน Vector Database")
    print("="*80)

    # Load knowledge base
    kb = HybridKnowledgeBase(
        persist_directory="./chroma_db",
        collection_name="chatbot_knowledge",
        use_reranker=False  # ปิด reranker เพื่อทดสอบเบื้องต้น
    )

    # Test queries
    test_queries = [
        "หลักสูตรนี้เรียนกี่ปี",
        "จบแล้วได้อะไร",
        "มีวิชาอะไรบ้าง",
        "คุณสมบัติผู้สมัคร"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"❓ คำถาม: {query}")
        print("="*80)

        results = kb.search_knowledge(query, n_results=3)

        if not results:
            print("❌ ไม่พบผลลัพธ์!")
            continue

        for result in results:
            print(f"\nอันดับ {result['rank']} (Score: {result['score']:.4f})")
            print(f"  {result['text'][:200]}...")

    print("\n" + "="*80)
    print("✅ การทดสอบการค้นหาเสร็จสิ้น")
    print("="*80)

def test_chatbot():
    """ทดสอบ chatbot"""
    print("\n" + "="*80)
    print("🤖 ทดสอบ Chatbot")
    print("="*80)

    # Load environment
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print("❌ ไม่พบ GEMINI_API_KEY ใน .env")
        print("📝 กรุณาสร้างไฟล์ .env และเพิ่ม GEMINI_API_KEY=your_api_key")
        return

    # Initialize
    kb = HybridKnowledgeBase(
        persist_directory="./chroma_db",
        collection_name="chatbot_knowledge",
        use_reranker=False
    )
    chatbot = GeminiChatbot(api_key, kb, use_compression=False)

    # Test questions
    test_questions = [
        "หลักสูตรนี้เรียนกี่ปีครับ",
        "จบแล้วได้อะไรบ้าง"
    ]

    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"🙋 คำถาม: {question}")
        print("="*80)
        print("🤔 กำลังคิด...")

        try:
            response = chatbot.chat(question)
            print(f"\n🤖 คำตอบ:\n{response}")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")

    print("\n" + "="*80)
    print("✅ การทดสอบ Chatbot เสร็จสิ้น")
    print("="*80)

if __name__ == "__main__":
    # Test search first
    test_search()

    # Then test chatbot
    test_chatbot()
