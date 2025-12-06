#!/usr/bin/env python3
"""ทดสอบ chatbot กับทั้ง 3 หลักสูตร"""

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

# Test questions covering all 3 curricula
test_questions = [
    "หลักสูตรวิศวกรรมโยธาและการศึกษาเรียนกี่ปี",  # TM
    "หลักสูตรครุศาสตร์อุตสาหกรรมบัณฑิต สาขาเทคโนโลยีคอมพิวเตอร์เรียนกี่ปี",  # CED
    "หลักสูตรเทียบโอนสำหรับผู้จบ ปวส. เรียนกี่ปี"  # TCT
]

print("\n" + "="*80)
print("🧪 ทดสอบ Chatbot กับทั้ง 3 หลักสูตร")
print("="*80)

for i, question in enumerate(test_questions, 1):
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
