"""
RAGAS Evaluation with Ollama (100% Free, No API Keys Required)
================================================================
This script evaluates the RAG chatbot using RAGAS metrics with Ollama
local models. No OpenAI, no API costs!

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Pull models:
   - ollama pull llama3.2
   - ollama pull nomic-embed-text

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
    # Use Langchain Ollama for local models
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except ImportError as e:
    print("❌ กรุณาติดตั้ง required libraries:")
    print("   pip install ragas langchain-ollama")
    print(f"   Error: {e}")
    exit(1)


class RAGASEvaluatorOllama:
    """Class for evaluating RAG chatbot using RAGAS metrics with Ollama (100% Free)"""

    def __init__(
        self,
        chatbot: TyphoonChatbot,
        ollama_model: str = "llama3.2",
        ollama_embedding: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize RAGAS evaluator with Ollama (local models)

        Args:
            chatbot: TyphoonChatbot instance to evaluate
            ollama_model: Ollama model name for evaluation (default: llama3.2)
            ollama_embedding: Ollama embedding model (default: nomic-embed-text)
            ollama_base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.chatbot = chatbot
        self.ollama_model = ollama_model
        self.ollama_embedding = ollama_embedding

        print(f"🤖 Initializing RAGAS with Ollama...")
        print(f"   LLM Model: {ollama_model}")
        print(f"   Embedding Model: {ollama_embedding}")
        print(f"   Base URL: {ollama_base_url}")

        try:
            # Initialize Ollama LLM for RAGAS
            self.llm = ChatOllama(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=0,
            )

            # Initialize Ollama Embeddings for RAGAS
            self.embeddings = OllamaEmbeddings(
                model=ollama_embedding,
                base_url=ollama_base_url,
            )

            print("✅ Ollama initialized successfully (100% Free, No API required!)")

        except Exception as e:
            print(f"❌ Error initializing Ollama: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Make sure Ollama is installed: https://ollama.ai/")
            print("   2. Start Ollama server: ollama serve")
            print(f"   3. Pull required models:")
            print(f"      ollama pull {ollama_model}")
            print(f"      ollama pull {ollama_embedding}")
            raise

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
        Evaluate the chatbot using RAGAS metrics with Ollama

        Args:
            test_data: List of test cases (if None, use default dataset)
            use_all_metrics: Whether to use all metrics

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
            # Use only the most important metrics
            metrics = [
                faithfulness,
                answer_relevancy,
            ]
            print("\n📊 Evaluating with core RAGAS metrics (faithfulness, relevancy)...")

        print("⏳ This may take a while (local models are slower but FREE)...")
        print(f"🤖 Using Ollama ({self.ollama_model}) for evaluation")

        # Run evaluation with Ollama
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
            print("   1. Make sure Ollama is running: ollama serve")
            print(f"   2. Check if model is available: ollama list")
            print(f"   3. Pull model if needed: ollama pull {self.ollama_model}")
            raise

    def save_results(self, results, output_file: str = "ragas_results_ollama.json"):
        """
        Save evaluation results to a JSON file

        Args:
            results: RAGAS evaluation results
            output_file: Output file path
        """
        # Convert results to dictionary
        results_dict = {
            "evaluator": f"Ollama ({self.ollama_model})",
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
        print(f"📊 RAGAS Evaluation Results (Using Ollama {self.ollama_model})")
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
        print(f"\n🤖 Evaluation performed by: Ollama ({self.ollama_model})")
        print("💰 Cost: FREE! (No API charges)")


def check_ollama_installed():
    """Check if Ollama is installed and running"""
    import subprocess

    try:
        # Check if ollama command exists
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print("✅ Ollama is installed and running")
            print("\n📋 Available models:")
            print(result.stdout)
            return True
        else:
            print("⚠️  Ollama is installed but may not be running")
            return False

    except FileNotFoundError:
        print("❌ Ollama is not installed")
        print("\n💡 Install Ollama:")
        print("   Visit: https://ollama.ai/")
        print("   Or run: curl -fsSL https://ollama.ai/install.sh | sh")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  Ollama is not responding")
        print("💡 Start Ollama: ollama serve")
        return False
    except Exception as e:
        print(f"⚠️  Error checking Ollama: {e}")
        return False


def main():
    """Main function to run RAGAS evaluation with Ollama"""

    print("="*80)
    print("🚀 RAGAS Evaluation with Ollama (100% Free)")
    print("="*80)

    # Check if Ollama is installed
    if not check_ollama_installed():
        print("\n❌ Please install and start Ollama first")
        return

    # Load environment variables
    load_dotenv()

    # Get Typhoon API key (for the chatbot being evaluated)
    typhoon_api_key = os.getenv('TYPHOON_API_KEY')

    if not typhoon_api_key:
        print("\n❌ ไม่พบ TYPHOON_API_KEY ใน environment variables")
        print("💡 กรุณาเพิ่ม TYPHOON_API_KEY ในไฟล์ .env")
        return

    try:
        # Initialize chatbot
        print("\n" + "="*80)
        print("🔧 Initializing chatbot...")
        print("="*80)

        kb = HybridKnowledgeBase(
            persist_directory="./chroma_db",
            collection_name="chatbot_knowledge",
            use_keyword_boost=True
        )
        chatbot = TyphoonChatbot(typhoon_api_key, kb)

        # Initialize evaluator with Ollama
        print("\n🤖 Initializing RAGAS Evaluator with Ollama...")
        evaluator = RAGASEvaluatorOllama(
            chatbot,
            ollama_model="llama3.2",  # Change to your preferred model
            ollama_embedding="nomic-embed-text"
        )

        # Ask user about evaluation scope
        print("\n" + "="*80)
        print("⚙️  Evaluation Options")
        print("="*80)
        print("1. Core evaluation (faithfulness, relevancy) - Faster")
        print("2. Full evaluation (all metrics) - More comprehensive but slower")
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
        print("🎉 100% Free evaluation with Ollama - No API costs!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
