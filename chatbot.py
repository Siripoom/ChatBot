import json
import os
from typing import List, Dict, Optional, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize

class HybridKnowledgeBase:
    """Class to handle hybrid retrieval (BM25 + Vector Search)"""

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "chatbot_knowledge"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize the embedding model
        print("🤖 Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        # Initialize ChromaDB client
        print(f"💾 Connecting to ChromaDB at: {persist_directory}")
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get the collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✅ Connected to collection: {collection_name}")
            print(f"📚 Collection contains {self.collection.count()} documents")
        except Exception as e:
            print(f"❌ Error loading collection: {e}")
            raise

        # Load all documents for BM25
        print("📝 Loading documents for BM25 indexing...")
        self._load_documents_for_bm25()

    def _load_documents_for_bm25(self):
        """Load all documents and create BM25 index"""
        # Get all documents from ChromaDB
        all_docs = self.collection.get()
        self.documents = all_docs['documents']
        self.doc_ids = all_docs['ids']
        
        # Tokenize documents for BM25 using Thai word tokenizer
        print("🔤 Tokenizing documents for BM25...")
        self.tokenized_docs = [word_tokenize(doc, engine='newmm') for doc in self.documents]
        
        # Create BM25 index
        print("🔍 Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"✅ BM25 index ready with {len(self.documents)} documents")

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform BM25 keyword search
        
        Args:
            query: User's question
            top_k: Number of top results to return
            
        Returns:
            List of (doc_index, score) tuples
        """
        # Tokenize query using Thai tokenizer
        tokenized_query = word_tokenize(query, engine='newmm')
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results with indices
        top_indices = bm25_scores.argsort()[-top_k:][::-1]
        results = [(idx, bm25_scores[idx]) for idx in top_indices if bm25_scores[idx] > 0]
        
        return results
    
    def _vector_search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform vector similarity search
        
        Args:
            query: User's question
            top_k: Number of top results to return
            
        Returns:
            List of (doc_index, similarity_score) tuples
        """
        # Generate embedding for the query
        query_embedding = self.model.encode([query])

        # Search in vector database
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )

        # Convert to (index, score) format
        # ChromaDB returns distance (lower is better), convert to similarity (higher is better)
        vector_results = []
        if results['documents'][0]:
            for doc_text, distance in zip(results['documents'][0], results['distances'][0]):
                # Find document index
                try:
                    doc_idx = self.documents.index(doc_text)
                    # Convert distance to similarity score (1 - normalized_distance)
                    similarity = 1 / (1 + distance)  # Higher score = more similar
                    vector_results.append((doc_idx, similarity))
                except ValueError:
                    continue
        
        return vector_results
    
    def search_knowledge(self, query: str, n_results: int = 5, 
                        bm25_weight: float = 0.4, vector_weight: float = 0.6) -> List[Dict[str, str]]:
        """
        Hybrid search combining BM25 and vector similarity
        
        Args:
            query: User's question
            n_results: Number of top results to return
            bm25_weight: Weight for BM25 scores (default 0.4)
            vector_weight: Weight for vector scores (default 0.6)
            
        Returns:
            List of relevant knowledge items with text and relevance score
        """
        # Perform both searches
        bm25_results = self._bm25_search(query, top_k=15)
        vector_results = self._vector_search(query, top_k=15)
        
        # Normalize scores for both methods
        def normalize_scores(results: List[Tuple[int, float]]) -> Dict[int, float]:
            if not results:
                return {}
            scores = [score for _, score in results]
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                return {idx: 1.0 for idx, _ in results}
            
            return {
                idx: (score - min_score) / (max_score - min_score)
                for idx, score in results
            }
        
        bm25_normalized = normalize_scores(bm25_results)
        vector_normalized = normalize_scores(vector_results)
        
        # Combine scores using weighted sum
        combined_scores = {}
        all_indices = set(bm25_normalized.keys()) | set(vector_normalized.keys())
        
        for idx in all_indices:
            bm25_score = bm25_normalized.get(idx, 0.0)
            vector_score = vector_normalized.get(idx, 0.0)
            
            # Weighted combination
            combined_scores[idx] = (bm25_weight * bm25_score) + (vector_weight * vector_score)
        
        # Sort by combined score and get top results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n_results]
        
        # Format results
        relevant_knowledge = []
        for rank, (idx, score) in enumerate(sorted_results, 1):
            relevant_knowledge.append({
                'text': self.documents[idx],
                'score': score,
                'rank': rank,
                'bm25_score': bm25_normalized.get(idx, 0.0),
                'vector_score': vector_normalized.get(idx, 0.0)
            })
        
        return relevant_knowledge

    def get_context_string(self, relevant_items: List[Dict[str, str]]) -> str:
        """Convert relevant knowledge items to context string"""
        if not relevant_items:
            return ""

        context = "ข้อมูลที่เกี่ยวข้องจากฐานความรู้:\n\n"
        for item in relevant_items:
            context += f"{item['rank']}. {item['text']}\n\n"

        return context


class GeminiChatbot:
    """Main chatbot class using Gemini API"""

    def __init__(self, api_key: str, knowledge_base: HybridKnowledgeBase):
        self.api_key = api_key
        self.knowledge_base = knowledge_base
        self.setup_gemini()
        self.conversation_history = []
    
    def setup_gemini(self):
        """Initialize Gemini API"""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("✅ เชื่อมต่อ Gemini API สำเร็จ!")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ Gemini API: {e}")
            raise
    
    def expand_query(self, query: str) -> str:
        """
        Expand user query to improve search results for natural language questions
        """
        # Common conversational patterns in Thai
        query_lower = query.lower()

        # Normalize casual questions
        if any(word in query_lower for word in ['อยาก', 'ต้องการ', 'สนใจ']):
            # User expressing interest
            if 'รู้' in query_lower or 'ทราบ' in query_lower:
                query += " ข้อมูล รายละเอียด"

        # Normalize question words
        if 'ทำไง' in query_lower or 'ยังไง' in query_lower:
            query += " วิธีการ ขั้นตอน"

        if 'มี' in query_lower and 'ไหม' in query_lower:
            query += " มีหรือไม่"

        # Common topics expansion
        if 'สมัคร' in query_lower:
            query += " การรับสมัคร คุณสมบัติ เอกสาร"

        if 'เรียน' in query_lower:
            query += " หลักสูตร วิชา การศึกษา"

        if 'ค่า' in query_lower and ('ใช้จ่าย' in query_lower or 'เทอม' in query_lower or 'เล่าเรียน' in query_lower):
            query += " ค่าธรรมเนียม ค่าลงทะเบียน ค่าใช้จ่าย"

        return query

    def create_prompt(self, user_question: str) -> str:
        """Create a comprehensive prompt with context"""
        # Expand query for better search
        expanded_query = self.expand_query(user_question)

        # Search for relevant knowledge with more results
        relevant_knowledge = self.knowledge_base.search_knowledge(expanded_query, n_results=8)
        context = self.knowledge_base.get_context_string(relevant_knowledge)

        prompt = f"""คุณคือ "น้องบอท" ผู้ช่วยให้คำปรึกษาเกี่ยวกับหลักสูตรและการรับสมัครนักศึกษา
ของคณะครุศาสตร์อุตสาหกรรม มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (มจพ.)

**บุคลิกภาพและสไตล์การตอบ:**
- ตอบด้วยภาษาที่เป็นกันเอง เหมือนพี่ให้คำแนะนำน้อง ไม่เป็นทางการจนเกินไป
- ใช้คำว่า "ครับ/ค่ะ" ตามความเหมาะสม
- อธิบายให้เข้าใจง่าย ไม่ใช้ศัพท์เทคนิคที่ซับซ้อนเกินไป
- ตอบสั้นกระชับ ตรงประเด็น ไม่ยืดยาวเกินไป
- ถ้าคำถามไม่ชัดเจน ถามกลับอย่างสุภาพ

**วิธีการตอบ:**
1. ตอบโจทย์หลักของคำถามก่อนเสมอ (ตรงๆ ไม่อ้อมค้อม)
2. ให้รายละเอียดเพิ่มเติมที่จำเป็น (ถ้ามี)
3. ถ้าไม่แน่ใจหรือไม่มีข้อมูล บอกตรงๆ และแนะนำช่องทางติดต่อ

**ข้อมูลอ้างอิง:**
{context if context else "(ไม่พบข้อมูลที่เกี่ยวข้องในฐานความรู้)"}

**คำถามจากผู้ใช้:** {user_question}

**หมายเหตุสำคัญ:**
- ใช้เฉพาะข้อมูลจากฐานความรู้ที่ให้มา
- ถ้าไม่มีข้อมูล ให้บอกว่า "ขออภัยนะคะ ข้อมูลนี้ไม่มีในระบบ" และแนะนำให้ติดต่อ:
  * เว็บไซต์: admission.kmutnb.ac.th
  * โทร: 02-555-2000 (สำนักงานคณะ)
  * Facebook: คณะครุศาสตร์อุตสาหกรรม มจพ.

**ตัวอย่างการตอบที่ดี:**
❌ "ตามข้อมูลที่ระบุไว้ในหลักสูตร หลักสูตรวิศวกรรมโยธาและการศึกษามีระยะเวลาศึกษาทั้งหมด 5 ปีครับ"
✅ "หลักสูตรวิศวกรรมโยธาและการศึกษาเรียน 5 ปีนะคะ จบแล้วได้วุฒิครูและวิศวกรทั้งสองอาชีพเลย"

**คำตอบของคุณ:**"""

        return prompt
    
    def chat(self, user_input: str) -> str:
        """Process user input and return chatbot response"""
        try:
            # Create prompt with knowledge base context
            prompt = self.create_prompt(user_input)
            
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            if response.text:
                # Store conversation history
                self.conversation_history.append({
                    "user": user_input,
                    "bot": response.text
                })
                return response.text
            else:
                return "ขออภัย ไม่สามารถสร้างคำตอบได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
                
        except Exception as e:
            print(f"เกิดข้อผิดพลาด: {e}")
            return "ขออภัย เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้ง"
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


def main():
    """Main function to run the chatbot"""

    # Load environment variables
    load_dotenv()

    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ ไม่พบ API Key ใน environment variables")
        return

    try:
        # Initialize vector knowledge base and chatbot
        print("=" * 80)
        print("🔧 Initializing chatbot with hybrid retrieval system...")
        print("=" * 80)
        kb = HybridKnowledgeBase(persist_directory="./chroma_db", collection_name="chatbot_knowledge")
        chatbot = GeminiChatbot(api_key, kb)
        
        print("=" * 80)
        print("🤖 ยินดีต้อนรับสู่ระบบแชทบอทคณะครุศาสตร์อุตสาหกรรม")
        print("    มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ")
        print("=" * 80)
        print("📚 ระบบ: Hybrid Retrieval (BM25 + Vector Search)")
        print("🔍 เทคโนโลยี: BM25 (keyword) + ChromaDB (semantic) + PyThaiNLP")
        print("-" * 80)
        print("📝 คุณสามารถถามคำถามเกี่ยวกับหลักสูตรวิศวกรรมโยธาและการศึกษาได้")
        print("💡 พิมพ์ 'quit', 'exit', หรือ 'ออก' เพื่อปิดโปรแกรม")
        print("🔄 พิมพ์ 'clear' เพื่อล้างประวัติการสนทนา")
        print("📜 พิมพ์ 'history' เพื่อดูประวัติการสนทนา")
        print("-" * 80)
        
        while True:
            # Get user input
            user_input = input("\n🙋 คุณ: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'ออก', '']:
                print("\n👋 ขอบคุณที่ใช้บริการ! สวัสดีครับ/ค่ะ")
                break
            
            # Check for clear command
            if user_input.lower() in ['clear', 'ล้าง']:
                chatbot.clear_history()
                print("\n🧹 ล้างประวัติการสนทนาเรียบร้อยแล้ว")
                continue
            
            # Check for history command
            if user_input.lower() in ['history', 'ประวัติ']:
                history = chatbot.get_conversation_history()
                if history:
                    print("\n📜 ประวัติการสนทนา:")
                    print("-" * 40)
                    for i, conv in enumerate(history, 1):
                        print(f"[{i}] คุณ: {conv['user']}")
                        print(f"[{i}] บอท: {conv['bot'][:100]}...")
                        print("-" * 40)
                else:
                    print("\n📭 ยังไม่มีประวัติการสนทนา")
                continue
            
            # Process the question
            print("\n🤔 กำลังคิด...")
            response = chatbot.chat(user_input)
            print(f"\n🤖 บอท: {response}")
            
    except KeyboardInterrupt:
        print("\n\n👋 ขอบคุณที่ใช้บริการ! สวัสดีครับ/ค่ะ")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")


if __name__ == "__main__":
    main()
