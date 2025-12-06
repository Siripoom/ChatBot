#!/usr/bin/env python3
"""ทดสอบการค้นหาในแต่ละหลักสูตร"""

from chatbot import HybridKnowledgeBase

kb = HybridKnowledgeBase(
    persist_directory="./chroma_db",
    collection_name="chatbot_knowledge",
    use_reranker=False
)

# Test search for each curriculum
test_queries = [
    ("วิศวกรรมโยธาและการศึกษา ระยะเวลาเรียน ปี", "TM"),
    ("ครุศาสตร์อุตสาหกรรมบัณฑิต เทคโนโลยีคอมพิวเตอร์ ระยะเวลา", "CED"),
    ("เทียบโอน ปวส ระยะเวลา 4 ปี", "TCT")
]

print("="*80)
print("🔍 ทดสอบการค้นหาในแต่ละหลักสูตร")
print("="*80)

for query, curriculum in test_queries:
    print(f"\n🔎 Query: '{query}' (หลักสูตร {curriculum})")
    print("-"*80)

    results = kb.search_knowledge(query, n_results=3)

    if results:
        for i, result in enumerate(results[:3], 1):
            source = result.get('metadata', {}).get('source', 'Unknown')
            score = result.get('score', 0)
            text_preview = result['text'][:100].replace('\n', ' ') + "..."
            print(f"\n{i}. Score: {score:.4f}")
            print(f"   Source: {source}")
            print(f"   Text: {text_preview}")
    else:
        print("❌ ไม่พบผลลัพธ์!")

    print("-"*80)
