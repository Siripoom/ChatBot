"""
RAGAS Evaluation with Google Gemini (No OpenAI Required)
=========================================================
This script evaluates the RAG chatbot using RAGAS metrics with Google Gemini
instead of OpenAI for evaluation.

RAGAS Metrics:
- Faithfulness: How factually accurate is the answer based on the context?
- Answer Relevancy: How relevant is the answer to the question?
- Context Precision: How precise are the retrieved contexts?
- Context Recall: How well do contexts cover the ground truth?
"""

import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd

# Import the chatbot
from chatbot_v04_keywords import HybridKnowledgeBase, TyphoonChatbot

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    # Use Langchain Google Generative AI instead of OpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except ImportError as e:
    print("❌ กรุณาติดตั้ง required libraries:")
    print("   pip install ragas langchain-google-genai")
    print(f"   Error: {e}")
    exit(1)


class RAGASEvaluatorGemini:
    """Class for evaluating RAG chatbot using RAGAS metrics with Google Gemini"""

    def __init__(self, chatbot: TyphoonChatbot, gemini_api_key: str):
        """
        Initialize RAGAS evaluator with Google Gemini

        Args:
            chatbot: TyphoonChatbot instance to evaluate
            gemini_api_key: Google Gemini API key for RAGAS evaluation
        """
        self.chatbot = chatbot
        self.gemini_api_key = gemini_api_key

        # Initialize Gemini LLM and Embeddings for RAGAS
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=gemini_api_key,
            temperature=0,
            convert_system_message_to_human=True  # Important for compatibility
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )

        print("✅ Initialized RAGAS with Google Gemini (No OpenAI required)")

    def create_test_dataset(self) -> List[Dict]:
        """
        Create test dataset with questions and ground truth answers

        Returns:
            List of test cases with questions and ground truth
        """
        # Test dataset - customize based on your knowledge base
        test_data = [
            {
                "question": "หลักสูตรวิศวกรรมโยธาและการศึกษาเรียนกี่ปี",
                "ground_truth": "หลักสูตรวิศวกรรมโยธาและการศึกษาใช้เวลาเรียน 5 ปี ตามระบบทวิภาค"
            },
            {
                "question": "จบหลักสูตรวิศวกรรมโยธาและการศึกษาแล้วได้ปริญญาอะไร",
                "ground_truth": "ได้รับปริญญาวิศวกรรมศาสตรบัณฑิต (วศ.บ.) สาขาวิชาวิศวกรรมโยธา และใบประกอบวิชาชีพครู"
            },
            {
                "question": "หลักสูตรนี้มีการฝึกประสบการณ์วิชาชีพอย่างไร",
                "ground_truth": "มีการฝึกปฏิบัติการสอนในสถานศึกษา และการฝึกงานในสถานประกอบการหรือโครงการวิศวกรรมโยธา"
            },
            {
                "question": "คุณสมบัติของผู้สมัครเข้าเรียนหลักสูตรนี้",
                "ground_truth": "ต้องสำเร็จการศึกษาระดับมัธยมศึกษาตอนปลายหรือเทียบเท่า และผ่านการคัดเลือกตามระบบ TCAS หรือระบบรับตรงของมหาวิทยาลัย"
            },
            {
                "question": "หลักสูตรนี้มีการเรียนการสอนเป็นภาษาอังกฤษหรือไม่",
                "ground_truth": "มีการจัดการเรียนการสอนทั้งภาษาไทยและภาษาอังกฤษ โดยมีเอกสารและตำราทั้งสองภาษา"
            },
            {
                "question": "จบแล้วสามารถประกอบอาชีพอะไรได้บ้าง",
                "ground_truth": "สามารถเป็นวิศวกรโยธา ครูสอนวิชาชีพด้านวิศวกรรม นักวิชาการ หรือนักวิจัยด้านวิศวกรรมโยธาและการศึกษา"
            },
            {
                "question": "โครงสร้างหลักสูตรมีหน่วยกิตทั้งหมดกี่หน่วยกิต",
                "ground_truth": "โครงสร้างหลักสูตรมีจำนวนหน่วยกิตรวมไม่น้อยกว่า 200 หน่วยกิต แบ่งเป็นหมวดวิชาศึกษาทั่วไป วิชาเฉพาะ และวิชาเลือกเสรี"
            },
            {
                "question": "มีการเทียบโอนหน่วยกิตได้หรือไม่",
                "ground_truth": "สามารถเทียบโอนหน่วยกิตได้ตามระเบียบของมหาวิทยาลัย สำหรับผู้ที่มีคุณสมบัติและผ่านเกณฑ์การพิจารณา"
            },
            {
                "question": "หลักสูตรนี้มีความร่วมมือกับหน่วยงานภายนอกหรือไม่",
                "ground_truth": "มีความร่วมมือกับสถาบันการศึกษา บริษัท และหน่วยงานภาครัฐและเอกชนในการจัดการเรียนการสอนและฝึกประสบการณ์"
            },
            {
                "question": "ค่าใช้จ่ายในการเรียนต่อภาคเรียนประมาณเท่าไร",
                "ground_truth": "ค่าใช้จ่ายขึ้นอยู่กับจำนวนหน่วยกิตที่ลงทะเบียน โดยสามารถสอบถามรายละเอียดเพิ่มเติมได้ที่สำนักงานคณะหรือเว็บไซต์ของมหาวิทยาลัย"
            }
        ]

        return test_data

    def generate_answers_and_contexts(self, test_data: List[Dict]) -> Dict[str, List]:
        """
        Generate answers and retrieve contexts for each question

        Args:
            test_data: List of test cases

        Returns:
            Dictionary with questions, answers, contexts, and ground truths
        """
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        print("\n" + "="*80)
        print("🔍 Generating answers and retrieving contexts...")
        print("="*80)

        for i, test_case in enumerate(test_data, 1):
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]

            print(f"\n[{i}/{len(test_data)}] Processing: {question}")

            try:
                # Expand query for better retrieval
                expanded_query = self.chatbot.expand_query(question)

                # Get contexts from knowledge base
                relevant_knowledge = self.chatbot.knowledge_base.search_knowledge(
                    expanded_query,
                    n_results=5
                )

                # Extract context texts
                context_list = [item['text'] for item in relevant_knowledge]

                # Generate answer
                answer = self.chatbot.chat(question)

                # Store results
                questions.append(question)
                answers.append(answer)
                contexts.append(context_list)
                ground_truths.append(ground_truth)

                print(f"   ✅ Answer: {answer[:80]}...")
                print(f"   📚 Retrieved {len(context_list)} contexts")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                # Add placeholder to maintain consistency
                questions.append(question)
                answers.append("Error generating answer")
                contexts.append(["Error retrieving context"])
                ground_truths.append(ground_truth)

        return {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

    def evaluate(self, test_data: List[Dict] = None, use_all_metrics: bool = False) -> Dict:
        """
        Evaluate the chatbot using RAGAS metrics with Google Gemini

        Args:
            test_data: List of test cases (if None, use default dataset)
            use_all_metrics: Whether to use all metrics (requires more API calls)

        Returns:
            Evaluation results
        """
        # Use default dataset if none provided
        if test_data is None:
            test_data = self.create_test_dataset()

        # Generate answers and contexts
        data = self.generate_answers_and_contexts(test_data)

        # Create dataset for RAGAS
        dataset = Dataset.from_dict(data)

        # Select metrics to evaluate
        if use_all_metrics:
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]
            print("\n📊 Evaluating with all available RAGAS metrics...")
        else:
            # Use only the most important metrics to save API costs
            metrics = [
                faithfulness,
                answer_relevancy,
            ]
            print("\n📊 Evaluating with core RAGAS metrics (faithfulness, relevancy)...")

        print("⏳ This may take a few minutes...")
        print("🤖 Using Google Gemini for evaluation (No OpenAI required)")

        # Run evaluation with Gemini
        try:
            results = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )

            return results

        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            print("\n💡 Troubleshooting tips:")
            print("   1. Check your GEMINI_API_KEY is valid")
            print("   2. Ensure you have internet connection")
            print("   3. Check Gemini API quota/limits")
            raise

    def save_results(self, results, output_file: str = "ragas_results_gemini.json"):
        """
        Save evaluation results to a JSON file

        Args:
            results: RAGAS evaluation results
            output_file: Output file path
        """
        # Convert results to dictionary
        results_dict = {
            "evaluator": "Google Gemini",
            "metrics": {k: float(v) for k, v in results.items()},
            "timestamp": pd.Timestamp.now().isoformat()
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        print(f"\n💾 Results saved to {output_file}")

    def print_results(self, results):
        """
        Print evaluation results in a readable format

        Args:
            results: RAGAS evaluation results
        """
        print("\n" + "="*80)
        print("📊 RAGAS Evaluation Results (Using Google Gemini)")
        print("="*80)

        for metric, score in results.items():
            # Format metric name
            metric_name = metric.replace('_', ' ').title()

            # Interpret score
            if score >= 0.8:
                emoji = "🟢"
                rating = "Excellent"
            elif score >= 0.6:
                emoji = "🟡"
                rating = "Good"
            elif score >= 0.4:
                emoji = "🟠"
                rating = "Fair"
            else:
                emoji = "🔴"
                rating = "Needs Improvement"

            print(f"{emoji} {metric_name:25s}: {score:.4f} ({rating})")

        print("="*80)

        # Provide interpretation
        print("\n📖 Metric Explanations:")
        print("-" * 80)
        print("• Faithfulness: How factually accurate is the answer based on context? (1.0 = perfect)")
        print("• Answer Relevancy: How relevant is the answer to the question? (1.0 = perfect)")
        print("• Context Precision: How precise are the top-ranked contexts? (1.0 = perfect)")
        print("• Context Recall: How well do contexts cover ground truth? (1.0 = perfect)")
        print("-" * 80)
        print("\n🤖 Evaluation performed by: Google Gemini (No OpenAI required)")


def main():
    """Main function to run RAGAS evaluation with Gemini"""

    # Load environment variables
    load_dotenv()

    # Get API keys
    typhoon_api_key = os.getenv('TYPHOON_API_KEY')
    gemini_api_key = os.getenv('GEMINI_API_KEY')

    if not typhoon_api_key:
        print("❌ ไม่พบ TYPHOON_API_KEY ใน environment variables")
        print("💡 กรุณาเพิ่ม TYPHOON_API_KEY ในไฟล์ .env")
        return

    if not gemini_api_key:
        print("❌ ไม่พบ GEMINI_API_KEY ใน environment variables")
        print("💡 กรุณาเพิ่ม GEMINI_API_KEY ในไฟล์ .env")
        print("   สามารถสมัครได้ที่: https://ai.google.dev/")
        return

    try:
        # Initialize chatbot
        print("="*80)
        print("🔧 Initializing chatbot...")
        print("="*80)

        kb = HybridKnowledgeBase(
            persist_directory="./chroma_db",
            collection_name="chatbot_knowledge",
            use_keyword_boost=True
        )
        chatbot = TyphoonChatbot(typhoon_api_key, kb)

        # Initialize evaluator with Gemini
        print("\n🤖 Initializing RAGAS Evaluator with Google Gemini...")
        evaluator = RAGASEvaluatorGemini(chatbot, gemini_api_key)

        # Ask user about evaluation scope
        print("\n" + "="*80)
        print("⚙️  Evaluation Options")
        print("="*80)
        print("1. Core evaluation (faithfulness, relevancy) - Fast and economical")
        print("2. Full evaluation (all metrics) - More comprehensive")
        print("-"*80)

        choice = input("Choose evaluation mode (1 or 2) [default: 1]: ").strip()
        use_all_metrics = (choice == "2")

        # Run evaluation
        results = evaluator.evaluate(use_all_metrics=use_all_metrics)

        # Print results
        evaluator.print_results(results)

        # Save results
        evaluator.save_results(results)

        print("\n✅ Evaluation completed successfully!")
        print("🎉 No OpenAI API required - powered by Google Gemini!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
