#!/usr/bin/env python3
"""
Semantic Chunker - แยก text ตาม context และความหมาย ไม่ใช่แค่จำนวนตัวอักษร
"""

import re
from typing import List, Dict
from pythainlp import sent_tokenize
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SemanticChunker:
    """
    Advanced text chunker that splits based on semantic meaning and document structure
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def detect_headings(self, text: str) -> List[Dict[str, any]]:
        """
        ตรวจจับหัวข้อและโครงสร้างเอกสาร
        """
        headings = []

        # Pattern สำหรับหัวข้อ (เช่น "1.1", "1.1.1", "ก)", "ข)", etc.)
        patterns = [
            # หมายเลขหัวข้อ เช่น 1.1, 1.1.1, 1.1.1.1
            r'^(\d+\.(?:\d+\.)*\d*)\s+(.+?)$',
            # หัวข้อแบบอักษร เช่น ก), ข), ค)
            r'^([ก-ฮ]\))\s+(.+?)$',
            # หัวข้อแบบวงเล็บ เช่น (1), (2)
            r'^\((\d+)\)\s+(.+?)$',
            # หัวข้อที่ขึ้นต้นด้วยบท เช่น "บทที่ 1"
            r'^(บทที่\s*\d+)\s*(.*)$',
            # หัวข้อที่เป็นตัวหนา (อาจมี marker)
            r'^(.+?)(?=\n{2,}|\Z)',
        ]

        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            for pattern in patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    headings.append({
                        'line_num': i,
                        'level': self._get_heading_level(match.group(1) if match.groups() else line),
                        'number': match.group(1) if match.groups() else '',
                        'title': match.group(2) if len(match.groups()) > 1 else line,
                        'full_text': line
                    })
                    break

        return headings

    def _get_heading_level(self, number: str) -> int:
        """กำหนดระดับของหัวข้อ"""
        if 'บทที่' in number:
            return 1
        if re.match(r'^\d+$', number):  # เลขเดี่ยว
            return 2
        if re.match(r'^\d+\.\d+$', number):  # 1.1
            return 3
        if re.match(r'^\d+\.\d+\.\d+$', number):  # 1.1.1
            return 4
        if re.match(r'^[ก-ฮ]\)$', number):  # ก)
            return 5
        return 6

    def split_by_structure(self, text: str) -> List[Dict[str, str]]:
        """
        แยก text ตามโครงสร้างเอกสาร (หัวข้อ, หัวข้อย่อย)
        """
        headings = self.detect_headings(text)
        chunks = []

        if not headings:
            # ถ้าไม่มีหัวข้อ ใช้วิธี sentence-based chunking
            return self.split_by_sentences(text)

        lines = text.split('\n')

        for i, heading in enumerate(headings):
            start_line = heading['line_num']
            end_line = headings[i + 1]['line_num'] if i + 1 < len(headings) else len(lines)

            # Extract content under this heading
            section_lines = lines[start_line:end_line]
            section_text = '\n'.join(section_lines).strip()

            if section_text:
                # ถ้าส่วนนี้ยาวเกินไป แยกย่อยต่อ
                if len(section_text) > self.chunk_size * 1.5:
                    sub_chunks = self.split_large_section(section_text, heading)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append({
                        'text': section_text,
                        'metadata': {
                            'heading': heading['full_text'],
                            'level': heading['level'],
                            'type': 'section'
                        }
                    })

        return chunks

    def split_large_section(self, text: str, heading: Dict) -> List[Dict[str, str]]:
        """
        แยกส่วนที่ยาวเกินไปออกเป็นชิ้นเล็ก โดยรักษา context
        """
        # ใช้ sentence tokenization สำหรับภาษาไทย
        sentences = sent_tokenize(text, engine='crfcut')

        chunks = []
        current_chunk = ""
        current_sentences = []

        for sentence in sentences:
            # ถ้าเพิ่มประโยคนี้แล้วยาวเกินไป
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        'heading': heading['full_text'],
                        'level': heading['level'],
                        'type': 'subsection',
                        'sentences': len(current_sentences)
                    }
                })

                # เริ่ม chunk ใหม่ พร้อม overlap
                if self.chunk_overlap > 0 and current_sentences:
                    # เอาประโยคสุดท้ายมาทับซ้อน
                    overlap_text = ' '.join(current_sentences[-2:]) if len(current_sentences) > 1 else current_sentences[-1]
                    current_chunk = overlap_text + ' ' + sentence
                    current_sentences = current_sentences[-2:] + [sentence]
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
                current_sentences.append(sentence)

        # เพิ่ม chunk สุดท้าย
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'heading': heading['full_text'],
                    'level': heading['level'],
                    'type': 'subsection',
                    'sentences': len(current_sentences)
                }
            })

        return chunks

    def split_by_sentences(self, text: str) -> List[Dict[str, str]]:
        """
        แยกตามประโยค (สำหรับกรณีที่ไม่มีหัวข้อชัดเจน)
        """
        sentences = sent_tokenize(text, engine='crfcut')

        chunks = []
        current_chunk = ""
        sentence_count = 0

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        'type': 'sentences',
                        'sentence_count': sentence_count
                    }
                })

                # Overlap
                words = current_chunk.split()
                if len(words) > 20:
                    overlap = ' '.join(words[-20:])
                    current_chunk = overlap + ' ' + sentence
                else:
                    current_chunk = sentence

                sentence_count = 1
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
                sentence_count += 1

        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    'type': 'sentences',
                    'sentence_count': sentence_count
                }
            })

        return chunks

    def chunk_text(self, text: str, source: str = "") -> List[Dict[str, str]]:
        """
        Main method: แยก text ด้วย semantic chunking
        """
        print(f"🔍 กำลังวิเคราะห์โครงสร้างเอกสาร...")

        # ลองแยกตามโครงสร้างก่อน
        chunks = self.split_by_structure(text)

        print(f"✅ แยกได้ {len(chunks)} chunks ตามโครงสร้างเอกสาร")

        # เพิ่ม metadata
        for i, chunk in enumerate(chunks):
            # สร้าง unique ID โดยใช้ source filename + index
            source_prefix = source.replace('.pdf', '').replace(' ', '_').replace('ชุดข้อมูลที่', 'doc')
            chunk['id'] = f"{source_prefix}_chunk_{i}"
            chunk['source'] = source
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata']['chunk_index'] = i
            chunk['metadata']['total_chunks'] = len(chunks)

        return chunks

    def print_chunk_analysis(self, chunks: List[Dict[str, str]]):
        """แสดงการวิเคราะห์ chunks"""
        print("\n📊 การวิเคราะห์ Chunks:")
        print("=" * 80)

        # สถิติ
        total_chunks = len(chunks)
        avg_length = sum(len(c['text']) for c in chunks) / total_chunks if total_chunks > 0 else 0
        min_length = min(len(c['text']) for c in chunks) if chunks else 0
        max_length = max(len(c['text']) for c in chunks) if chunks else 0

        print(f"จำนวน chunks: {total_chunks}")
        print(f"ขนาดเฉลี่ย: {avg_length:.0f} ตัวอักษร")
        print(f"ขนาดเล็กสุด: {min_length} ตัวอักษร")
        print(f"ขนาดใหญ่สุด: {max_length} ตัวอักษร")

        # นับประเภท
        types = {}
        for chunk in chunks:
            chunk_type = chunk.get('metadata', {}).get('type', 'unknown')
            types[chunk_type] = types.get(chunk_type, 0) + 1

        print(f"\nประเภท chunks:")
        for chunk_type, count in types.items():
            print(f"  - {chunk_type}: {count}")

        # แสดงตัวอย่าง
        print(f"\n📋 ตัวอย่าง chunks (3 อันแรก):")
        print("=" * 80)
        for i, chunk in enumerate(chunks[:3], 1):
            heading = chunk.get('metadata', {}).get('heading', 'ไม่มีหัวข้อ')
            text_preview = chunk['text'][:150].replace('\n', ' ')
            print(f"\n{i}. หัวข้อ: {heading}")
            print(f"   ความยาว: {len(chunk['text'])} ตัวอักษร")
            print(f"   เนื้อหา: {text_preview}...")


# ฟังก์ชันทดสอบ
def test_semantic_chunker():
    """ทดสอบ semantic chunker"""
    test_text = """
    บทที่ 1 ข้อมูลทั่วไป

    1.1 ชื่อหลักสูตร
    หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธาและการศึกษา

    1.2 ชื่อปริญญา
    ก) ภาษาไทย: วิศวกรรมศาสตรบัณฑิต (วิศวกรรมโยธาและการศึกษา)
    ข) ภาษาอังกฤษ: Bachelor of Engineering (Civil Engineering and Education)

    1.3 หน่วยงานที่รับผิดชอบ
    คณะครุศาสตร์อุตสาหกรรม มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ
    """

    chunker = SemanticChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_text(test_text, source="test.pdf")
    chunker.print_chunk_analysis(chunks)


if __name__ == "__main__":
    test_semantic_chunker()
