#!/usr/bin/env python3
"""Debug: ดูว่า chatbot ค้นหาและขยายคำถามอย่างไร"""

import os
from dotenv import load_dotenv
from chatbot import HybridKnowledgeBase, GeminiChatbot

load_dotenv()

kb = HybridKnowledgeBase(
    persist_directory="./chroma_db",
    collection_name="chatbot_knowledge",
    use_reranker=False
)

chatbot = GeminiChatbot(
    api_key=os.getenv('GEMINI_API_KEY'),
    knowledge_base=kb,
    use_compression=False
)

# Test the CED question
question = "หลักสูตรครุศาสตร์อุตสาหกรรมบัณฑิต สาขาเทคโนโลยีคอมพิวเตอร์เรียนกี่ปี"

print("="*80)
print(f"คำถาม: {question}")
print("="*80)

# Check expanded query
expanded = chatbot.expand_query(question)
print(f"\n🔍 Expanded query:\n{expanded}")

# Check what search returns
print(f"\n📚 ผลการค้นหา (10 อันดับแรก):")
print("-"*80)
results = kb.search_knowledge(expanded, n_results=10)

for i, result in enumerate(results, 1):
    source = result.get('metadata', {}).get('source', 'Unknown')
    score = result.get('score', 0)
    text = result['text'].replace('\n', ' ')[:150] + "..."
    print(f"\n{i}. Score: {score:.4f} | Source: {source}")
    print(f"   {text}")
