#!/usr/bin/env python3
"""ทดสอบด้วยคำถามที่เรียบง่าย ไม่ซับซ้อน"""

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

# คำถามเรียบง่าย ตรงไปตรงมา
simple_questions = [
    "หลักสูตรวิศวกรรมโยธาและการศึกษาเรียนกี่ปี",
    "ครุศาสตร์อุตสาหกรรมบัณฑิต สาขาเทคโนโลยีคอมพิวเตอร์ กี่ปี",
    "หลักสูตรเทียบโอน ปวส กี่ปี"
]

print("="*80)
print("🧪 ทดสอบด้วยคำถามเรียบง่าย")
print("="*80)

for i, question in enumerate(simple_questions, 1):
    print(f"\n[{i}] 🙋 {question}")
    print("-"*80)

    try:
        response = chatbot.chat(question)
        # Show first 3 lines
        lines = response.split('\n')
        preview = '\n'.join(lines[:3])
        print(f"🤖 {preview}")
        if len(lines) > 3:
            print("    ...")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("-"*80)

print("\n✅ ทดสอบเสร็จสิ้น!")
