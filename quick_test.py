#!/usr/bin/env python3
"""ทดสอบ chatbot แบบรวดเร็ว"""

import os
from dotenv import load_dotenv
from chatbot import HybridKnowledgeBase, GeminiChatbot

load_dotenv()

# Initialize
print("🚀 เริ่มต้นระบบ...")
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

# Test questions
questions = [
    "หลักสูตรนี้เรียนกี่ปี",
    "จบแล้วได้วุฒิอะไร",
    "มีวิชาอะไรบ้าง",
    "ต้องเรียนวิชาคอมพิวเตอร์ไหม"
]

print("\n" + "="*80)
print("📝 ทดสอบคำถาม")
print("="*80)

for i, question in enumerate(questions, 1):
    print(f"\n[{i}] 🙋 {question}")
    print("-"*80)

    try:
        response = chatbot.chat(question)
        print(f"🤖 {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("-"*80)

print("\n✅ ทดสอบเสร็จสิ้น!")
