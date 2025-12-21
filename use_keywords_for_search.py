#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
สคริปต์ตัวอย่างการนำ keywords ไปใช้กับระบบค้นหาและ Vector Database
"""

import json
from typing import List, Dict

def load_keywords(json_file: str) -> Dict:
    """โหลดข้อมูล keywords จากไฟล์ JSON"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_top_keywords_by_file(data: Dict, top_n: int = 20) -> Dict[str, List[str]]:
    """
    ดึง top N keywords จากแต่ละไฟล์

    Returns:
        Dictionary ที่มี filename เป็น key และ list ของ keywords เป็น value
    """
    result = {}
    for filename, file_data in data['files'].items():
        keywords = [kw['word'] for kw in file_data['keywords'][:top_n]]
        result[filename] = keywords
    return result

def create_metadata_for_vector_db(data: Dict, top_n: int = 30) -> List[Dict]:
    """
    สร้าง metadata สำหรับ Vector Database

    แต่ละเอกสารจะมี metadata ที่ประกอบด้วย:
    - filename: ชื่อไฟล์
    - keywords: keywords ทั้งหมด
    - top_keywords: top N keywords ที่มีความถี่สูงสุด
    - text_length: ความยาวของเอกสาร
    - keyword_categories: หมวดหมู่ที่คาดเดาจาก keywords
    """
    metadata_list = []

    for filename, file_data in data['files'].items():
        # ดึง keywords
        all_keywords = [kw['word'] for kw in file_data['keywords']]
        top_keywords = all_keywords[:top_n]

        # คาดเดาหมวดหมู่จาก keywords
        categories = categorize_by_keywords(top_keywords)

        metadata = {
            'filename': filename,
            'text_length': file_data['text_length'],
            'total_keywords': file_data['total_keywords'],
            'keywords': all_keywords,
            'top_keywords': top_keywords,
            'categories': categories,
            # สร้าง string สำหรับ full-text search
            'keywords_text': ' '.join(top_keywords)
        }

        metadata_list.append(metadata)

    return metadata_list

def categorize_by_keywords(keywords: List[str]) -> List[str]:
    """
    คาดเดาหมวดหมู่จาก keywords
    """
    categories = []

    # วิศวกรรมโยธา
    civil_keywords = {'วิศวกรรมโยธา', 'โยธา', 'โครงสร้าง', 'ออกแบบ', 'งาน'}
    if any(kw in civil_keywords for kw in keywords):
        categories.append('วิศวกรรมโยธา')

    # คอมพิวเตอร์/เทคโนโลยี
    cs_keywords = {'คอมพิวเตอร์', 'เทคโนโลยี', 'ระบบ', 'ข้อมูล', 'software', 'development'}
    if any(kw in cs_keywords for kw in keywords):
        categories.append('คอมพิวเตอร์/เทคโนโลยี')

    # วิชาศึกษาทั่วไป
    general_keywords = {'วิชาศึกษาทั่วไป', 'ทักษะ', 'ภาษาอังกฤษ', 'สังคม', 'ธุรกิจ'}
    if any(kw in general_keywords for kw in keywords):
        categories.append('วิชาศึกษาทั่วไป')

    # คู่มือนักศึกษา
    student_keywords = {'นักศึกษา', 'ลงทะเบียน', 'สอบ', 'ข้อบังคับ', 'ภาคการศึกษา'}
    if any(kw in student_keywords for kw in keywords):
        categories.append('คู่มือนักศึกษา')

    # หลักสูตร
    curriculum_keywords = {'หลักสูตร', 'รายวิชา', 'หน่วยกิต', 'วิชาบังคับ', 'คำอธิบาย'}
    if any(kw in curriculum_keywords for kw in keywords):
        categories.append('หลักสูตร')

    return categories if categories else ['ทั่วไป']

def search_by_keyword(data: Dict, search_term: str, top_n: int = 3) -> List[Dict]:
    """
    ค้นหาเอกสารที่มี keyword ที่ตรงกับ search_term

    Returns:
        List ของเอกสารที่เกี่ยวข้อง พร้อมคะแนน
    """
    results = []

    for filename, file_data in data['files'].items():
        # นับจำนวนครั้งที่ keyword ปรากฏ
        score = 0
        matched_keywords = []

        for kw_data in file_data['keywords']:
            keyword = kw_data['word']
            frequency = kw_data['frequency']

            # ค้นหาแบบ partial match
            if search_term.lower() in keyword.lower() or keyword.lower() in search_term.lower():
                score += frequency
                matched_keywords.append({
                    'keyword': keyword,
                    'frequency': frequency
                })

        if score > 0:
            results.append({
                'filename': filename,
                'score': score,
                'matched_keywords': matched_keywords
            })

    # เรียงตามคะแนน
    results.sort(key=lambda x: x['score'], reverse=True)

    return results[:top_n]

def get_related_keywords(data: Dict, keyword: str, top_n: int = 10) -> List[str]:
    """
    หา keywords ที่เกี่ยวข้องกับ keyword ที่ระบุ
    โดยดูจากไฟล์ที่มี keyword นั้นปรากฏ
    """
    related = set()

    # หาไฟล์ที่มี keyword
    for filename, file_data in data['files'].items():
        has_keyword = False

        for kw_data in file_data['keywords']:
            if keyword.lower() in kw_data['word'].lower():
                has_keyword = True
                break

        # ถ้าไฟล์มี keyword ดึง keywords อื่นๆ จากไฟล์นั้น
        if has_keyword:
            for kw_data in file_data['keywords'][:top_n]:
                if keyword.lower() not in kw_data['word'].lower():
                    related.add(kw_data['word'])

    return list(related)[:top_n]

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # โหลดข้อมูล
    keywords_file = "/home/siripoom/chatbot/data/keywords_analysis.json"
    data = load_keywords(keywords_file)

    print("="*80)
    print("ตัวอย่างการใช้งาน Keywords สำหรับระบบค้นหา")
    print("="*80)

    # 1. สร้าง metadata สำหรับ Vector Database
    print("\n1️⃣  สร้าง Metadata สำหรับ Vector Database")
    print("-"*80)
    metadata_list = create_metadata_for_vector_db(data, top_n=20)

    for i, meta in enumerate(metadata_list, 1):
        print(f"\n📄 เอกสารที่ {i}: {meta['filename']}")
        print(f"   หมวดหมู่: {', '.join(meta['categories'])}")
        print(f"   Top Keywords: {', '.join(meta['top_keywords'][:5])}...")

    # 2. ค้นหาด้วย keyword
    print("\n\n2️⃣  ตัวอย่างการค้นหาด้วย Keyword")
    print("-"*80)

    search_terms = ["คอมพิวเตอร์", "วิศวกรรม", "นักศึกษา"]

    for term in search_terms:
        print(f"\n🔍 ค้นหา: '{term}'")
        results = search_by_keyword(data, term, top_n=3)

        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['filename']}")
            print(f"      คะแนน: {result['score']}")
            print(f"      Keywords ที่ตรง: {', '.join([kw['keyword'] for kw in result['matched_keywords'][:3]])}")

    # 3. หา related keywords
    print("\n\n3️⃣  หา Keywords ที่เกี่ยวข้อง")
    print("-"*80)

    target_keywords = ["คอมพิวเตอร์", "วิศวกรรม"]

    for keyword in target_keywords:
        related = get_related_keywords(data, keyword, top_n=8)
        print(f"\n📌 Keywords ที่เกี่ยวข้องกับ '{keyword}':")
        print(f"   {', '.join(related)}")

    # 4. Export metadata เป็น JSON
    print("\n\n4️⃣  Export Metadata")
    print("-"*80)

    output_file = "/home/siripoom/chatbot/data/vector_db_metadata.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    print(f"✅ บันทึก metadata สำหรับ Vector DB ที่: {output_file}")

    print("\n" + "="*80)
    print("✨ เสร็จสิ้นตัวอย่างการใช้งาน!")
    print("="*80)
