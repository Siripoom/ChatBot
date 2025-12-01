#!/usr/bin/env python3
"""
Script to evaluate and tune chatbot performance with natural language queries
"""

import os
from dotenv import load_dotenv
from chatbot import VectorKnowledgeBase, GeminiChatbot
import time

def test_natural_queries():
    """Test the chatbot with natural, conversational queries"""
    print("=" * 80)
    print("🧪 ทดสอบระบบแชทบอทด้วยคำถามที่เป็นธรรมชาติ")
    print("=" * 80)

    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print("❌ ไม่พบ GEMINI_API_KEY")
        return

    # Initialize chatbot
    print("\n📦 กำลังโหลดระบบ...")
    kb = VectorKnowledgeBase(persist_directory="./chroma_db", collection_name="chatbot_knowledge")
    chatbot = GeminiChatbot(api_key, kb)

    # Natural language test queries (คำถามแบบธรรมชาติที่คนจริงจะถาม)
    natural_queries = [
        # คำถามแบบสบายๆ ไม่เป็นทางการ
        "เรียนกี่ปีจบครับ",
        "อยากรู้ว่าเรียนอะไรบ้าง",
        "ถ้าอยากสมัครเข้ามาเรียน ต้องทำยังไงบ้าง",
        "มีวิชาการสำรวจมั้ย เรียนเรื่องอะไร",
        "จบไปแล้วได้อะไร มีใบประกอบวิชาชีพมั้ย",
        "ต้องเรียนวิชาคอมด้วยหรือเปล่า",
        "เรียนเกี่ยวกับนวัตกรรมมั้ย",
        "ค่าเทอมเท่าไหร่ครับ",
        "สมัครเรียนเมื่อไหร่",
        "อยากทราบข้อมูลเพิ่มเติมติดต่อที่ไหน"
    ]

    print(f"\n🔍 จำนวนคำถามทดสอบ: {len(natural_queries)}")
    print("=" * 80)

    results = []

    for i, query in enumerate(natural_queries, 1):
        print(f"\n{'='*80}")
        print(f"คำถามที่ {i}: {query}")
        print("=" * 80)

        # Time the response
        start_time = time.time()

        # Get expanded query
        expanded = chatbot.expand_query(query)
        if expanded != query:
            print(f"🔍 Query Expansion: {query} → {expanded}")

        # Search results
        print(f"\n📊 ผลการค้นหาจาก Vector Database:")
        search_results = kb.search_knowledge(expanded, n_results=8)

        for j, result in enumerate(search_results[:3], 1):  # Show top 3
            print(f"\n  อันดับ {j} (Distance: {result['distance']:.4f})")
            preview = result['text'][:150].replace('\n', ' ')
            print(f"  {preview}...")

        # Get chatbot response
        print(f"\n💬 คำตอบจากแชทบอท:")
        print("-" * 80)
        response = chatbot.chat(query)
        print(response)
        print("-" * 80)

        end_time = time.time()
        response_time = end_time - start_time

        results.append({
            'query': query,
            'expanded': expanded,
            'num_results': len(search_results),
            'top_distance': search_results[0]['distance'] if search_results else None,
            'response_time': response_time,
            'response_length': len(response)
        })

        print(f"⏱️  เวลาตอบ: {response_time:.2f} วินาที")

        # Small delay to avoid rate limiting
        if i < len(natural_queries):
            time.sleep(1)

    # Summary statistics
    print("\n" + "=" * 80)
    print("📈 สรุปผลการทดสอบ")
    print("=" * 80)

    avg_time = sum(r['response_time'] for r in results) / len(results)
    avg_distance = sum(r['top_distance'] for r in results if r['top_distance']) / len([r for r in results if r['top_distance']])
    avg_length = sum(r['response_length'] for r in results) / len(results)

    print(f"\n📊 สถิติรวม:")
    print(f"  • จำนวนคำถามทั้งหมด: {len(results)}")
    print(f"  • เวลาตอบเฉลี่ย: {avg_time:.2f} วินาที")
    print(f"  • ระยะห่าง (distance) เฉลี่ยของผลลัพธ์อันดับ 1: {avg_distance:.4f}")
    print(f"  • ความยาวคำตอบเฉลี่ย: {avg_length:.0f} ตัวอักษร")

    # Best and worst matches
    best_match = min(results, key=lambda x: x['top_distance'] if x['top_distance'] else float('inf'))
    worst_match = max(results, key=lambda x: x['top_distance'] if x['top_distance'] else 0)

    print(f"\n🎯 คำถามที่ตรงที่สุด (distance ต่ำสุด):")
    print(f"  '{best_match['query']}' - Distance: {best_match['top_distance']:.4f}")

    print(f"\n⚠️  คำถามที่ตรงน้อยที่สุด (distance สูงสุด):")
    print(f"  '{worst_match['query']}' - Distance: {worst_match['top_distance']:.4f}")

    print("\n" + "=" * 80)
    print("✅ การทดสอบเสร็จสมบูรณ์!")
    print("=" * 80)

    return results


def compare_chunk_sizes():
    """Compare performance with different chunk sizes"""
    print("\n" + "=" * 80)
    print("🔬 เปรียบเทียบผลลัพธ์จาก Chunk Size ต่างๆ")
    print("=" * 80)
    print("\n💡 คำแนะนำ: การปรับ chunk size ต้องสร้าง vector database ใหม่")
    print("   แก้ไขค่าใน init_vector_db.py แล้วรันใหม่:")
    print("   - chunk_size=300 → เหมาะกับคำตอบสั้น ตรงประเด็น")
    print("   - chunk_size=500 → เหมาะกับคำอธิบายทั่วไป (ปัจจุบัน)")
    print("   - chunk_size=800 → เหมาะกับเนื้อหายาวที่ต้องบริบทเยอะ")
    print("\n   Overlap: 50-100 → ป้องกันการตัดข้อมูลสำคัญ")


def recommend_tuning():
    """Provide recommendations for tuning"""
    print("\n" + "=" * 80)
    print("💡 คำแนะนำในการปรับจูนระบบ")
    print("=" * 80)

    recommendations = """
📌 การปรับปรุงที่ทำไปแล้ว:
   ✅ ปรับ prompt ให้ตอบเป็นธรรมชาติมากขึ้น
   ✅ เพิ่ม query expansion สำหรับคำถามแบบสบายๆ
   ✅ เพิ่มจำนวน results จาก 5 เป็น 8 → ได้ context มากขึ้น

🔧 พารามิเตอร์ที่ปรับได้:

1. **จำนวน Results (n_results)**
   - ปัจจุบัน: 8 results
   - แนะนำ: 5-10 (มากเกินไปทำให้เสียเวลาและ context ยาว)
   - แก้ที่: chatbot.py line 134

2. **Chunk Size & Overlap**
   - ปัจจุบัน: chunk_size=500, overlap=50
   - แนะนำทดสอบ:
     * คำตอบสั้น: 300-400, overlap=50
     * คำตอบยาว: 600-800, overlap=100
   - แก้ที่: init_vector_db.py line 132 → รัน init ใหม่

3. **Query Expansion**
   - เพิ่มคำที่เกี่ยวข้องในคำถามเพื่อค้นหาได้ดีขึ้น
   - แก้ที่: chatbot.py method expand_query() line 96-126

4. **Temperature ของ Gemini** (ยังไม่ได้ปรับ)
   - ปัจจุบัน: default (~0.9)
   - แนะนำ: 0.3-0.5 (ตอบชัดเจนกว่า)
   - วิธีปรับ: เพิ่ม generation_config ใน setup_gemini()

📊 วิธีวัดผล:
   • Distance ต่ำ (< 8.0) = ผลลัพธ์ตรงมาก
   • Distance สูง (> 10.0) = อาจไม่ตรง ต้องปรับ chunk หรือ query
   • เวลาตอบ < 3 วินาที = ดี
   • ความยาวคำตอบ 150-400 ตัวอักษร = เหมาะสม

🎯 เป้าหมาย:
   • ตอบได้ตรงประเด็น (Direct answer)
   • ใช้ภาษาเป็นธรรมชาติ (Natural language)
   • รวดเร็ว (< 3 วินาที)
   • ครบถ้วน แต่กระชับ
"""

    print(recommendations)


if __name__ == "__main__":
    # Run natural language tests
    results = test_natural_queries()

    # Show recommendations
    compare_chunk_sizes()
    recommend_tuning()

    print("\n💾 บันทึก: ผลการทดสอบอยู่ใน memory เท่านั้น")
    print("   ถ้าต้องการเก็บ ให้เพิ่มโค้ด save เป็น JSON/CSV")
