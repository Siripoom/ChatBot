#!/usr/bin/env python3
"""
Script to initialize vector database from PDF files
Extracts text from PDF and creates embeddings using sentence-transformers
"""

import os
import re
from typing import List, Dict
import PyPDF2
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from semantic_chunker import SemanticChunker


class PDFProcessor:
    """Process PDF files and extract text content"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_text(self) -> str:
        """Extract all text from PDF"""
        text = ""
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"📄 Processing PDF with {len(pdf_reader.pages)} pages...")

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    text += page_text + "\n"
                    if page_num % 10 == 0:
                        print(f"   Processed {page_num}/{len(pdf_reader.pages)} pages")

                print(f"✅ Extracted {len(text)} characters from PDF")
                return text
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {e}")
            raise

    def split_into_chunks_semantic(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, str]]:
        """
        Split text using SEMANTIC CHUNKING - แยกตามความหมายและโครงสร้าง
        ไม่ใช่แค่ตัดตามจำนวนตัวอักษร

        Args:
            text: Full text to split
            chunk_size: Target size of each chunk in characters
            overlap: Number of overlapping characters between chunks

        Returns:
            List of dictionaries with chunk text and metadata (รวม heading, type, etc.)
        """
        print(f"🎯 ใช้ Semantic Chunking แทนการตัดแบบเดิม")
        print(f"   • แยกตามหัวข้อและโครงสร้างเอกสาร")
        print(f"   • ใช้ Thai NLP (pythainlp) สำหรับ sentence tokenization")
        print(f"   • รักษา context และความหมายของแต่ละส่วน")

        # ใช้ SemanticChunker
        chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = chunker.chunk_text(text, source=os.path.basename(self.pdf_path))

        # แสดงการวิเคราะห์
        chunker.print_chunk_analysis(chunks)

        return chunks


class VectorDBBuilder:
    """Build and manage ChromaDB vector database"""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "chatbot_knowledge"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        print(f"🤖 Loaded embedding model: paraphrase-multilingual-MiniLM-L12-v2")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        print(f"💾 ChromaDB initialized at: {persist_directory}")

    def create_collection(self, chunks: List[Dict[str, str]]):
        """Create or recreate collection and add documents"""
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"🗑️  Deleted existing collection: {self.collection_name}")
        except:
            pass

        # Create new collection
        collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Knowledge base for chatbot from PDF documents"}
        )
        print(f"✨ Created new collection: {self.collection_name}")

        # Prepare data for ChromaDB
        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [{"source": chunk["source"]} for chunk in chunks]

        # Generate embeddings
        print(f"🔄 Generating embeddings for {len(documents)} chunks...")
        embeddings = self.model.encode(documents, show_progress_bar=True)

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end].tolist(),
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            print(f"   Added batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")

        print(f"✅ Successfully added {len(ids)} documents to vector database")

        # Verify count immediately
        actual_count = collection.count()
        print(f"🔍 Verification: Collection actually has {actual_count} documents")
        if actual_count != len(ids):
            print(f"⚠️  WARNING: Expected {len(ids)} but got {actual_count}!")

        # Print sample to verify
        print(f"\n📊 Sample document from collection:")
        print(f"   ID: {ids[0]}")
        print(f"   Text preview: {documents[0][:200]}...")

    def test_search(self, query: str = "สมัครเรียน", n_results: int = 3):
        """Test vector search with a sample query"""
        collection = self.client.get_collection(name=self.collection_name)

        print(f"\n🔍 Testing search with query: '{query}'")
        query_embedding = self.model.encode([query])

        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )

        print(f"\n📋 Top {n_results} results:")
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0]), 1):
            print(f"\n{i}. (Distance: {distance:.4f})")
            print(f"   {doc[:300]}...")


def main():
    """Main function to build vector database"""
    print("=" * 80)
    print("🚀 Vector Database Initialization")
    print("=" * 80)

    # PDF file paths - รองรับหลายไฟล์
    pdf_paths = [
        "/home/siripoom/chatbot/data/ชุดข้อมูลที่ 1 ภาคโยธาTM.pdf",
        "/home/siripoom/chatbot/data/ชุดข้อมูลที่ 2 ภาคคอม CED.pdf",
        "/home/siripoom/chatbot/data/ชุดข้อมูลที่ 3 ภาคคอมTCT.pdf"
    ]

    # Check if PDFs exist
    existing_pdfs = []
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            existing_pdfs.append(pdf_path)
            print(f"✅ Found: {os.path.basename(pdf_path)}")
        else:
            print(f"⚠️  Not found: {os.path.basename(pdf_path)}")

    if not existing_pdfs:
        print(f"❌ No PDF files found!")
        return

    print(f"\n📂 Processing {len(existing_pdfs)} PDF file(s)")

    # ประมวลผลแต่ละไฟล์และรวม chunks
    all_chunks = []

    for idx, pdf_path in enumerate(existing_pdfs, 1):
        print(f"\n{'='*80}")
        print(f"Processing PDF {idx}/{len(existing_pdfs)}: {os.path.basename(pdf_path)}")
        print("=" * 80)

        # Step 1: Extract text from PDF
        print(f"\nStep 1.{idx}: Extracting text from PDF")
        pdf_processor = PDFProcessor(pdf_path)
        text = pdf_processor.extract_text()

        # Step 2: Split into chunks using SEMANTIC CHUNKING
        print(f"\nStep 2.{idx}: Splitting text into chunks (Semantic Chunking)")
        chunks = pdf_processor.split_into_chunks_semantic(text, chunk_size=500, overlap=100)

        print(f"✅ Got {len(chunks)} chunks from {os.path.basename(pdf_path)}")
        all_chunks.extend(chunks)

    print(f"\n{'='*80}")
    print(f"📊 Total chunks from all PDFs: {len(all_chunks)}")
    print("=" * 80)

    # Step 3: Build vector database
    print(f"\n{'='*80}")
    print("Step 3: Building vector database")
    print("=" * 80)
    db_builder = VectorDBBuilder(
        persist_directory="./chroma_db",
        collection_name="chatbot_knowledge"
    )
    db_builder.create_collection(all_chunks)

    # Step 4: Test the database
    print(f"\n{'='*80}")
    print("Step 4: Testing vector search")
    print("=" * 80)
    db_builder.test_search(query="สมัครเรียน", n_results=3)

    print(f"\n{'='*80}")
    print("✅ Vector database initialization complete!")
    print("=" * 80)
    print(f"\n💡 Database location: ./chroma_db")
    print(f"💡 Collection name: chatbot_knowledge")
    print(f"💡 Total chunks: {len(all_chunks)}")
    print(f"💡 Source PDFs: {len(existing_pdfs)} files")


if __name__ == "__main__":
    main()
