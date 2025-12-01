#!/usr/bin/env python3
"""
Test script for Hybrid Retrieval (BM25 + Vector Search)
"""

import os
import sys
from dotenv import load_dotenv
from chatbot import HybridKnowledgeBase, GeminiChatbot

def test_search_comparison():
    """Compare BM25, Vector, and Hybrid search results"""
    
    print("=" * 80)
    print("🧪 ทดสอบระบบ Hybrid Retrieval")
    print("=" * 80)
    
    # Initialize knowledge base
    kb = HybridKnowledgeBase(persist_directory="./chroma_db", collection_name="chatbot_knowledge")
    
    # Test queries
    test_queries = [
        "หลักสูตรนี้เรียนกี่ปี",
        "ต้องสอบวิชาอะไรบ้าง",
        "ค่าเทอมเท่าไหร่",
        "มีทุนการศึกษาไหม",
        "คุณสมบัติผู้สมัคร"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"🔍 คำถาม: {query}")
        print(f"{'='*80}")
        
        # Test BM25 only
        print("\n📊 BM25 Search (Keyword-based):")
        print("-" * 80)
        bm25_results = kb._bm25_search(query, top_k=3)
        for i, (idx, score) in enumerate(bm25_results, 1):
            print(f"{i}. [Score: {score:.4f}] {kb.documents[idx][:100]}...")
        
        # Test Vector only
        print("\n🧠 Vector Search (Semantic-based):")
        print("-" * 80)
        vector_results = kb._vector_search(query, top_k=3)
        for i, (idx, score) in enumerate(vector_results, 1):
            print(f"{i}. [Score: {score:.4f}] {kb.documents[idx][:100]}...")
        
        # Test Hybrid
        print("\n⚡ Hybrid Search (BM25 40% + Vector 60%):")
        print("-" * 80)
        hybrid_results = kb.search_knowledge(query, n_results=3, bm25_weight=0.4, vector_weight=0.6)
        for result in hybrid_results:
            print(f"{result['rank']}. [Combined: {result['score']:.4f} | BM25: {result['bm25_score']:.4f} | Vector: {result['vector_score']:.4f}]")
            print(f"   {result['text'][:100]}...")
        
        print("\n" + "="*80)
        input("กด Enter เพื่อดูคำถามถัดไป...")


def test_weight_adjustment():
    """Test different weight combinations"""
    
    print("\n" + "=" * 80)
    print("⚖️  ทดสอบการปรับน้ำหนัก BM25 vs Vector")
    print("=" * 80)
    
    kb = HybridKnowledgeBase(persist_directory="./chroma_db", collection_name="chatbot_knowledge")
    
    query = "หลักสูตรนี้เรียนกี่ปี"
    print(f"\n🔍 คำถามทดสอบ: {query}\n")
    
    weight_configs = [
        (0.2, 0.8, "เน้น Semantic"),
        (0.4, 0.6, "Balanced (แนะนำ)"),
        (0.6, 0.4, "เน้น Keyword"),
        (0.8, 0.2, "เน้น Keyword มาก"),
    ]
    
    for bm25_w, vector_w, label in weight_configs:
        print(f"\n{'='*80}")
        print(f"⚙️  {label} - BM25: {bm25_w*100:.0f}% | Vector: {vector_w*100:.0f}%")
        print(f"{'='*80}")
        
        results = kb.search_knowledge(query, n_results=3, bm25_weight=bm25_w, vector_weight=vector_w)
        
        for result in results:
            print(f"\n{result['rank']}. Combined Score: {result['score']:.4f}")
            print(f"   BM25: {result['bm25_score']:.4f} | Vector: {result['vector_score']:.4f}")
            print(f"   {result['text'][:120]}...")


def test_with_chatbot():
    """Test full chatbot with hybrid retrieval"""
    
    print("\n" + "=" * 80)
    print("🤖 ทดสอบ Chatbot พร้อม Hybrid Retrieval")
    print("=" * 80)
    
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ ไม่พบ API Key")
        return
    
    kb = HybridKnowledgeBase(persist_directory="./chroma_db", collection_name="chatbot_knowledge")
    chatbot = GeminiChatbot(api_key, kb)
    
    test_questions = [
        "หลักสูตรนี้เรียนกี่ปีครับ",
        "จบแล้วได้อะไรบ้าง",
        "ค่าใช้จ่ายประมาณเท่าไหร่"
    ]
    
    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"🙋 คำถาม: {question}")
        print(f"{'='*80}")
        print("\n🤔 กำลังคิด...")
        
        response = chatbot.chat(question)
        print(f"\n🤖 คำตอบ:\n{response}")
        print("\n" + "="*80)
        input("กด Enter เพื่อดูคำถามถัดไป...")


def main():
    """Main function"""
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    🧪 Hybrid Retrieval Test Suite                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

เลือกการทดสอบ:
1. เปรียบเทียบ BM25 vs Vector vs Hybrid
2. ทดสอบการปรับน้ำหนัก (Weight Adjustment)
3. ทดสอบ Chatbot เต็มรูปแบบ
4. รันทุกการทดสอบ
0. ออก
""")
    
    choice = input("เลือก (0-4): ").strip()
    
    try:
        if choice == "1":
            test_search_comparison()
        elif choice == "2":
            test_weight_adjustment()
        elif choice == "3":
            test_with_chatbot()
        elif choice == "4":
            test_search_comparison()
            test_weight_adjustment()
            test_with_chatbot()
        elif choice == "0":
            print("\n👋 ขอบคุณที่ใช้บริการ!")
        else:
            print("\n❌ ตัวเลือกไม่ถูกต้อง")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
