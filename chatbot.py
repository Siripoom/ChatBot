import json
import os
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

class VectorKnowledgeBase:
    """Class to handle vector database-based knowledge retrieval"""

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

    def search_knowledge(self, query: str, n_results: int = 5) -> List[Dict[str, str]]:
        """
        Search for relevant knowledge using vector similarity

        Args:
            query: User's question
            n_results: Number of top results to return

        Returns:
            List of relevant knowledge items with text and relevance score
        """
        # Generate embedding for the query
        query_embedding = self.model.encode([query])

        # Search in vector database
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )

        # Format results
        relevant_knowledge = []
        if results['documents'][0]:
            for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                relevant_knowledge.append({
                    'text': doc,
                    'distance': distance,
                    'rank': i + 1
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

    def __init__(self, api_key: str, knowledge_base: VectorKnowledgeBase):
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
    
    def create_prompt(self, user_question: str) -> str:
        """Create a comprehensive prompt with context"""
        # Search for relevant knowledge
        relevant_knowledge = self.knowledge_base.search_knowledge(user_question)
        context = self.knowledge_base.get_context_string(relevant_knowledge)

        prompt = f"""คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับหลักสูตรของคณะครุศาสตร์อุตสาหกรรม มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ
ได้แก่ หลักสูตรคอมพิวเตอร์ศึกษา และหลักสูตรวิศวกรรมโยธาและการศึกษา

{context}

คำถามจากผู้ใช้: {user_question}

กรุณาตอบคำถามโดย:
1. ใช้ข้อมูลจากฐานความรู้ที่ให้มาเป็นหลัก
2. ตอบเป็นภาษาไทยที่เข้าใจง่าย
3. หากไม่มีข้อมูลในฐานความรู้ ให้บอกว่าไม่มีข้อมูลและแนะนำให้สอบถามจากทางมหาวิทยาลัยโดยตรง
4. ให้คำตอบที่ครบถ้วนและเป็นประโยชน์
5. ระบุชื่อหลักสูตรที่เกี่ยวข้องให้ชัดเจน (คอมพิวเตอร์ศึกษา หรือ วิศวกรรมโยธาและการศึกษา)

คำตอบ:"""

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
        print("🔧 Initializing chatbot with vector database...")
        print("=" * 80)
        kb = VectorKnowledgeBase(persist_directory="./chroma_db", collection_name="chatbot_knowledge")
        chatbot = GeminiChatbot(api_key, kb)
        
        print("=" * 80)
        print("🤖 ยินดีต้อนรับสู่ระบบแชทบอทคณะครุศาสตร์อุตสาหกรรม")
        print("    มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ")
        print("=" * 80)
        print("📚 ระบบ: Vector Database (Semantic Search)")
        print("🔍 เทคโนโลยี: ChromaDB + Sentence Transformers")
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
