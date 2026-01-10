# RAGAS Evaluation Without OpenAI

เอกสารนี้อธิบายวิธีการใช้ RAGAS สำหรับประเมิน RAG Chatbot โดย**ไม่ต้องใช้ OpenAI API**

## 🎯 ทางเลือกที่มี

### 1. ใช้ Google Gemini ⭐ (แนะนำ)
- **ไฟล์**: `ragas_with_gemini.py`
- **ข้อดี**: เร็ว, ราคาถูก, รองรับภาษาไทยดี
- **ข้อเสีย**: ต้องมี API key (ฟรี quota มี)
- **Setup**: ดูด้านล่าง

### 2. ใช้ Ollama (Local Models) 💰 (ฟรี 100%)
- **ไฟล์**: `ragas_with_ollama.py`
- **ข้อดี**: ฟรีทั้งหมด, ไม่ต้อง API key, ข้อมูลไม่ออกจากเครื่อง
- **ข้อเสีย**: ช้ากว่า, ต้องติดตั้ง Ollama
- **Setup**: ดูด้านล่าง

---

## 🚀 Setup Google Gemini (แนะนำ)

### ขั้นตอนที่ 1: ติดตั้ง Dependencies

```bash
pip install langchain-google-genai
```

### ขั้นตอนที่ 2: รับ API Key

1. ไปที่ [https://ai.google.dev/](https://ai.google.dev/)
2. คลิก "Get API key in Google AI Studio"
3. สร้าง API key ใหม่
4. Copy API key

### ขั้นตอนที่ 3: เพิ่ม API Key ใน .env

แก้ไขไฟล์ `.env` และเพิ่มบรรทัดนี้:

```bash
GEMINI_API_KEY=your_api_key_here
```

### ขั้นตอนที่ 4: รันการประเมิน

```bash
python ragas_with_gemini.py
```

### ตัวอย่างการใช้งานใน Code

```python
from ragas_with_gemini import RAGASEvaluatorGemini
from chatbot_v04_keywords import HybridKnowledgeBase, TyphoonChatbot

# Initialize chatbot
kb = HybridKnowledgeBase(persist_directory="./chroma_db")
chatbot = TyphoonChatbot(typhoon_api_key, kb)

# Initialize evaluator
evaluator = RAGASEvaluatorGemini(chatbot, gemini_api_key)

# Run evaluation
results = evaluator.evaluate(use_all_metrics=True)

# Print results
evaluator.print_results(results)
evaluator.save_results(results)
```

---

## 💻 Setup Ollama (ฟรี 100%)

### ขั้นตอนที่ 1: ติดตั้ง Ollama

**Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
ดาวน์โหลดจาก [https://ollama.ai/](https://ollama.ai/)

### ขั้นตอนที่ 2: ติดตั้ง Dependencies

```bash
pip install langchain-ollama
```

### ขั้นตอนที่ 3: Download Models

```bash
# LLM สำหรับประเมิน (เลือก 1 อัน)
ollama pull llama3.2          # แนะนำ - เร็ว, ไม่กิน RAM มาก
ollama pull llama3.1:8b       # ดีกว่า llama3.2 แต่ช้ากว่า
ollama pull qwen2.5:7b        # ดีสำหรับภาษาไทย

# Embedding model
ollama pull nomic-embed-text  # จำเป็น
```

### ขั้นตอนที่ 4: เริ่ม Ollama Server (ถ้ายังไม่ได้รัน)

```bash
ollama serve
```

### ขั้นตอนที่ 5: รันการประเมิน

```bash
python ragas_with_ollama.py
```

### ตัวอย่างการใช้งานใน Code

```python
from ragas_with_ollama import RAGASEvaluatorOllama
from chatbot_v04_keywords import HybridKnowledgeBase, TyphoonChatbot

# Initialize chatbot
kb = HybridKnowledgeBase(persist_directory="./chroma_db")
chatbot = TyphoonChatbot(typhoon_api_key, kb)

# Initialize evaluator
evaluator = RAGASEvaluatorOllama(
    chatbot,
    ollama_model="llama3.2",           # หรือ qwen2.5:7b
    ollama_embedding="nomic-embed-text"
)

# Run evaluation
results = evaluator.evaluate(use_all_metrics=True)

# Print results
evaluator.print_results(results)
evaluator.save_results(results)
```

---

## 📊 เปรียบเทียบทางเลือก

| ฟีเจอร์ | OpenAI (เดิม) | Google Gemini | Ollama |
|--------|--------------|---------------|---------|
| **ราคา** | แพง (~$0.15/1K tokens) | ถูกกว่า (~$0.075/1K tokens) | ฟรี 100% |
| **ความเร็ว** | เร็วมาก | เร็วมาก | ช้า (ขึ้นอยู่กับ GPU) |
| **ภาษาไทย** | ดี | ดีมาก | ปานกลาง-ดี |
| **API Key** | ต้องมี | ต้องมี (ฟรี quota) | ไม่ต้อง |
| **Privacy** | ข้อมูลส่งออก | ข้อมูลส่งออก | ข้อมูลอยู่ในเครื่อง |
| **Setup** | ง่าย | ง่าย | ซับซ้อนกว่า |

---

## 📝 Metrics ที่ประเมิน

### Core Metrics (ใช้ API น้อย)
- **Faithfulness**: คำตอบตรงกับ context ที่ให้หรือไม่
- **Answer Relevancy**: คำตอบตรงประเด็นคำถามหรือไม่

### Full Metrics
- **Context Precision**: Context ที่ retrieve มาแม่นยำแค่ไหน
- **Context Recall**: Context ครอบคลุม ground truth หรือไม่
- **Faithfulness**: เหมือนข้างบน
- **Answer Relevancy**: เหมือนข้างบน

---

## 🔧 Troubleshooting

### Google Gemini

**ปัญหา: "Invalid API Key"**
```bash
# ตรวจสอบใน .env
GEMINI_API_KEY=AIza...  # ต้องขึ้นต้นด้วย AIza
```

**ปัญหา: "Quota exceeded"**
- รอ 1 นาที แล้วลองใหม่
- หรือใช้ Ollama แทน (ฟรี)

### Ollama

**ปัญหา: "Connection refused"**
```bash
# เริ่ม Ollama server
ollama serve
```

**ปัญหา: "Model not found"**
```bash
# Download model
ollama pull llama3.2
ollama pull nomic-embed-text
```

**ปัญหา: ช้ามาก**
- ใช้ model เล็กกว่า (llama3.2 แทน llama3.1:70b)
- ลด test dataset (ใช้คำถามน้อยลง)
- หรือใช้ Gemini แทน

---

## 💡 Tips

1. **เริ่มต้น**: ใช้ Google Gemini (ง่าย, เร็ว, quota ฟรีมี)
2. **ประหยัด**: ใช้ Core metrics แทน Full metrics
3. **Privacy**: ใช้ Ollama ถ้าห่วงเรื่องข้อมูล
4. **ทดสอบ**: ลองทั้ง 2 แบบแล้วเปรียบเทียบผล

---

## 📚 Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [Google AI Studio](https://ai.google.dev/)
- [Ollama Models](https://ollama.ai/library)
- [Langchain Integrations](https://python.langchain.com/docs/integrations/)

---

## ❓ FAQ

**Q: ใช้ Gemini แล้วยังต้อง OpenAI ไหม?**
A: ไม่ต้อง! Gemini ทำงานแทน OpenAI ได้เลย

**Q: Ollama ต้องใช้ GPU ไหม?**
A: ไม่จำเป็น แต่มี GPU จะเร็วกว่ามาก

**Q: ภาษาไทยใช้ model ไหนดี?**
A: Gemini ดีที่สุด, Ollama ใช้ qwen2.5:7b สำหรับภาษาไทย

**Q: ประเมินครั้งหนึ่งใช้เวลานานแค่ไหน?**
A:
- Gemini: 2-5 นาที (10 คำถาม)
- Ollama: 10-30 นาที (ขึ้นอยู่กับ spec เครื่อง)

**Q: ค่าใช้จ่าย?**
A:
- Gemini: ~$0.01-0.05 ต่อการประเมิน (มี free quota)
- Ollama: ฟรี 100%

---

## 📞 Support

หากมีปัญหาหรือคำถาม:
1. ตรวจสอบ error message ใน terminal
2. ดู Troubleshooting section ด้านบน
3. ตรวจสอบ API keys และ model installation
