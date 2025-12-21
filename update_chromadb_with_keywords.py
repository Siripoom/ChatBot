#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
สคริปต์สำหรับอัปเดต ChromaDB metadata ให้มี keywords จากการตัดคำ
"""

import json
import chromadb
from pathlib import Path
from typing import Dict, List
from pythainlp import word_tokenize

def load_keywords_data(keywords_file: str) -> Dict:
    """โหลดข้อมูล keywords จาก JSON"""
    with open(keywords_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_matching_file(doc_metadata: Dict, keywords_data: Dict) -> str:
    """
    หาไฟล์ที่ตรงกับ metadata ของ document

    Args:
        doc_metadata: metadata ของ document จาก ChromaDB
        keywords_data: ข้อมูล keywords จาก JSON

    Returns:
        ชื่อไฟล์ที่ตรงกัน หรือ None
    """
    # ลองหาจาก metadata
    if 'source' in doc_metadata or 'filename' in doc_metadata:
        source = doc_metadata.get('source') or doc_metadata.get('filename')

        # ตรวจสอบว่าตรงกับไฟล์ไหนใน keywords_data
        for filename in keywords_data['files'].keys():
            # ถ้าชื่อไฟล์มีส่วนตรงกัน
            if Path(filename).stem in source or source in filename:
                return filename

    return None

def extract_keywords_from_text(text: str, top_n: int = 20) -> List[str]:
    """
    สกัด keywords จากข้อความโดยใช้การตัดคำ

    Args:
        text: ข้อความที่จะสกัด keywords
        top_n: จำนวน keywords ที่ต้องการ

    Returns:
        List ของ keywords
    """
    from collections import Counter
    from pythainlp.corpus import thai_stopwords

    # ตัดคำ
    words = word_tokenize(text, engine='newmm')

    # กรอง stopwords และคำสั้น
    stopwords = thai_stopwords()
    filtered = [w for w in words if len(w) >= 2 and w not in stopwords]

    # นับความถี่และหา top keywords
    counter = Counter(filtered)
    return [word for word, _ in counter.most_common(top_n)]

def update_chromadb_metadata(
    persist_directory: str = "./chroma_db",
    collection_name: str = "chatbot_knowledge",
    keywords_file: str = "./data/keywords_analysis.json"
):
    """
    อัปเดต metadata ใน ChromaDB ให้มี keywords

    Args:
        persist_directory: ตำแหน่ง ChromaDB
        collection_name: ชื่อ collection
        keywords_file: ไฟล์ keywords JSON
    """
    print("=" * 80)
    print("🔄 อัปเดต ChromaDB Metadata ด้วย Keywords")
    print("=" * 80)

    # โหลดข้อมูล keywords
    print(f"\n📖 โหลดข้อมูล keywords จาก: {keywords_file}")
    keywords_data = load_keywords_data(keywords_file)
    print(f"✅ โหลดข้อมูล {len(keywords_data['files'])} ไฟล์")

    # เชื่อมต่อ ChromaDB
    print(f"\n💾 เชื่อมต่อ ChromaDB: {persist_directory}")
    client = chromadb.PersistentClient(path=persist_directory)

    try:
        collection = client.get_collection(name=collection_name)
        print(f"✅ เชื่อมต่อ collection: {collection_name}")
        print(f"📚 มีเอกสารทั้งหมด: {collection.count()} documents")
    except Exception as e:
        print(f"❌ ไม่พบ collection: {e}")
        return

    # ดึงข้อมูลทั้งหมดจาก collection
    print("\n🔍 ดึงข้อมูลทั้งหมดจาก ChromaDB...")
    all_docs = collection.get(include=['documents', 'metadatas', 'embeddings'])

    doc_ids = all_docs['ids']
    documents = all_docs['documents']
    metadatas = all_docs['metadatas']
    embeddings = all_docs['embeddings']

    print(f"✅ ดึงข้อมูลสำเร็จ: {len(documents)} documents")

    # อัปเดต metadata สำหรับแต่ละ document
    print("\n📝 อัปเดต metadata...")
    updated_count = 0
    fallback_count = 0

    for i, (doc_id, doc_text, metadata) in enumerate(zip(doc_ids, documents, metadatas), 1):
        # หาไฟล์ที่ตรงกัน
        matched_file = find_matching_file(metadata, keywords_data)

        if matched_file:
            # ใช้ keywords จากไฟล์ที่ตรง
            file_data = keywords_data['files'][matched_file]
            top_keywords = [kw['word'] for kw in file_data['keywords'][:30]]

            # เพิ่ม metadata
            metadata['keywords'] = top_keywords
            metadata['keywords_source'] = matched_file
            metadata['keywords_method'] = 'file_match'

            print(f"  [{i}/{len(documents)}] ✅ {doc_id[:50]}... → {matched_file}")
            updated_count += 1
        else:
            # ถ้าหาไม่เจอ ให้สกัด keywords จากข้อความโดยตรง
            doc_keywords = extract_keywords_from_text(doc_text, top_n=20)

            metadata['keywords'] = doc_keywords
            metadata['keywords_source'] = 'text_extraction'
            metadata['keywords_method'] = 'fallback'

            print(f"  [{i}/{len(documents)}] ⚠️  {doc_id[:50]}... → สกัดจากข้อความ")
            fallback_count += 1

        # สร้าง keywords_text สำหรับ BM25
        metadata['keywords_text'] = ' '.join(metadata['keywords'][:15])

    # อัปเดตกลับเข้า ChromaDB
    print(f"\n💾 อัปเดตข้อมูลกลับเข้า ChromaDB...")

    try:
        # ต้องใช้ upsert เพราะ ChromaDB ไม่มี update โดยตรง
        collection.upsert(
            ids=doc_ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f"✅ อัปเดตสำเร็จ!")
    except Exception as e:
        print(f"❌ อัปเดตล้มเหลว: {e}")
        return

    # สรุปผล
    print("\n" + "=" * 80)
    print("📊 สรุปผลการอัปเดต")
    print("=" * 80)
    print(f"📝 เอกสารทั้งหมด: {len(documents)} documents")
    print(f"✅ ใช้ keywords จากไฟล์: {updated_count} documents")
    print(f"⚠️  สกัด keywords จากข้อความ: {fallback_count} documents")
    print(f"🎯 อัปเดตสำเร็จ: {updated_count + fallback_count} documents")

    # แสดงตัวอย่าง metadata
    print("\n📋 ตัวอย่าง metadata ที่อัปเดตแล้ว:")
    print("-" * 80)

    sample_idx = 0
    print(f"Document ID: {doc_ids[sample_idx]}")
    print(f"Metadata: {json.dumps(metadatas[sample_idx], ensure_ascii=False, indent=2)}")
    print(f"Top Keywords: {', '.join(metadatas[sample_idx]['keywords'][:10])}")

    print("\n" + "=" * 80)
    print("✨ อัปเดตเสร็จสมบูรณ์!")
    print("=" * 80)

if __name__ == "__main__":
    update_chromadb_metadata(
        persist_directory="./chroma_db",
        collection_name="chatbot_knowledge",
        keywords_file="./data/keywords_analysis.json"
    )
