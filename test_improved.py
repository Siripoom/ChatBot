#!/usr/bin/env python3
"""ทดสอบ chatbot หลังปรับปรุง query expansion"""

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

# Test questions - ครอบคลุมหลายประเภท
questions = [
    # ปีการศึกษา/ระยะเวลา
    "เรียนกี่ปีครับ",

    # อาชีพ/วุฒิ
    "จบแล้วทำงานอะไรได้บ้าง",

    # หลักสูตร/โครงสร้าง
    "โครงสร้างหลักสูตรเป็นยังไง",

    # ฝึกงาน
    "ต้องฝึกงานกี่ชั่วโมง",

    # เฉพาะสาขา
    "หลักสูตรคอมพิวเตอร์ศึกษามีอะไรบ้าง",

    # คุณสมบัติ/การรับสมัคร
    "ต้องมีคุณสมบัติอะไรบ้างถึงจะสมัครได้"
]

print("\n" + "="*80)
print("📝 ทดสอบคำถามหลากหลายประเภท")
print("="*80)

for i, question in enumerate(questions, 1):
    print(f"\n[{i}] 🙋 {question}")

    # แสดง expanded query
    expanded = chatbot.expand_query(question)
    if expanded != question:
        print(f"🔍 Expanded: {expanded[:100]}...")

    print("-"*80)

    try:
        response = chatbot.chat(question)
        # แสดงแค่ส่วนแรกของคำตอบ
        lines = response.split('\n')
        preview = '\n'.join(lines[:5])
        print(f"🤖 {preview}")
        if len(lines) > 5:
            print("    ...")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("-"*80)

print("\n✅ ทดสอบเสร็จสิ้น!")
