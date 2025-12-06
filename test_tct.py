#!/usr/bin/env python3
"""ทดสอบคำถาม TCT"""

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

question = "หลักสูตรเทียบโอนสำหรับผู้สำเร็จการศึกษา ปวส เรียนกี่ปี"

print("="*80)
print(f"🙋 {question}")
print("="*80)

response = chatbot.chat(question)
print(f"\n🤖 {response}")
