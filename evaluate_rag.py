"""
RAG Evaluation Script using Ragas

This script evaluates the RAG chatbot using various metrics:
- Context Precision: How relevant are the retrieved contexts?
- Context Recall: Does the context contain the answer?
- Faithfulness: Is the answer grounded in the context?
- Answer Relevance: Is the answer relevant to the question?
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    AnswerRelevancy,
    Faithfulness,
    ContextRecall,
    ContextPrecision,
)
from chatbot_v04_keywords import HybridKnowledgeBase, TyphoonChatbot


def create_test_dataset() -> List[Dict]:
    """
    Create a test dataset with question-answer pairs

    Returns:
        List of test cases with questions and ground truth answers
    """
    test_cases = [
        {
            "question": "หลักสูตรวิศวกรรมโยธาและการศึกษาเรียนกี่ปี",
            "ground_truth": "หลักสูตรวิศวกรรมโยธาและการศึกษาใช้เวลาศึกษา 5 ปี"
        },
        {
            "question": "จบหลักสูตรวิศวกรรมโยธาและการศึกษาแล้วได้อะไรบ้าง",
            "ground_truth": "จบแล้วได้ทั้งวุฒิครูและวิศวกร สามารถประกอบอาชีพได้ทั้งสองด้าน"
        },
        {
            "question": "คณะครุศาสตร์อุตสาหกรรมอยู่มหาวิทยาลัยไหน",
            "ground_truth": "คณะครุศาสตร์อุตสาหกรรมอยู่ที่มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (มจพ.)"
        },
        {
            "question": "การรับสมัครนักศึกษาใหม่เปิดรับสมัครเมื่อไหร่",
            "ground_truth": "ข้อมูลการรับสมัครนักศึกษาใหม่จะประกาศในช่วงเดือนพฤศจิกายน-ธันวาคม"
        },
        {
            "question": "ค่าเทอมหลักสูตรวิศวกรรมโยธาและการศึกษาเท่าไหร่",
            "ground_truth": "ค่าเทอมอยู่ที่ประมาณ 17,000-20,000 บาทต่อภาคการศึกษา"
        }
    ]

    return test_cases


def save_results(results: Dict, config_name: str, output_dir: str = "evaluation_results"):
    """
    Save evaluation results to a JSON file

    Args:
        results: Evaluation results dictionary
        config_name: Name of the configuration being evaluated
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{config_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Convert results to serializable format
    results_dict = {
        "config": config_name,
        "timestamp": timestamp,
        "metrics": {
            "context_precision": float(results.get('context_precision', 0)),
            "context_recall": float(results.get('context_recall', 0)),
            "faithfulness": float(results.get('faithfulness', 0)),
            "answer_relevancy": float(results.get('answer_relevancy', 0)),
        }
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {filepath}")
    return filepath


def evaluate_rag_system(
    knowledge_base: HybridKnowledgeBase,
    chatbot: TyphoonChatbot,
    test_cases: List[Dict],
    use_reranker: bool = True,
    use_compression: bool = True
) -> Dict:
    """
    Evaluate RAG system using Ragas metrics

    Args:
        knowledge_base: The hybrid knowledge base
        chatbot: The chatbot instance
        test_cases: List of test questions and ground truth answers
        use_reranker: Whether to use re-ranker
        use_compression: Whether to use context compression

    Returns:
        Evaluation results
    """
    print("=" * 80)
    print(f"🔍 Evaluating RAG System")
    print(f"   Re-ranker: {'✅ Enabled' if use_reranker else '❌ Disabled'}")
    print(f"   Context Compression: {'✅ Enabled' if use_compression else '❌ Disabled'}")
    print("=" * 80)

    # Prepare data for evaluation
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]

        print(f"\n[{i}/{len(test_cases)}] Processing: {question}")

        try:
            # Get relevant contexts
            relevant_knowledge = knowledge_base.search_knowledge(question, n_results=5)
            context_list = [item['text'] for item in relevant_knowledge]

            # Apply compression if enabled
            if use_compression and context_list:
                try:
                    relevant_knowledge = chatbot.compress_context(question, relevant_knowledge)
                    context_list = [item['text'] for item in relevant_knowledge]
                except Exception as e:
                    print(f"   ⚠️  Warning: Context compression failed: {e}")
                    # Continue with uncompressed context

            # Get chatbot answer
            answer = chatbot.chat(question)

            print(f"   ✅ Answer: {answer[:100]}...")
            print(f"   📄 Contexts: {len(context_list)} items")

            questions.append(question)
            answers.append(answer)
            contexts.append(context_list)
            ground_truths.append(ground_truth)

        except Exception as e:
            print(f"   ❌ Error processing question: {e}")
            # Add placeholder data to maintain dataset consistency
            questions.append(question)
            answers.append("Error: Could not generate answer")
            contexts.append(["Error: Could not retrieve context"])
            ground_truths.append(ground_truth)

    # Create dataset for Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(data)

    # Evaluate using Ragas metrics
    print("\n" + "=" * 80)
    print("📊 Running Ragas Evaluation...")
    print("=" * 80)

    try:
        result = evaluate(
            dataset,
            metrics=[
                ContextPrecision(),
                ContextRecall(),
                Faithfulness(),
                AnswerRelevancy(),
            ],
        )
        return result
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("Please check your API keys and network connection.")
        raise


def compare_configurations():
    """
    Compare different RAG configurations
    """
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        return

    # Create test dataset
    test_cases = create_test_dataset()
    results_all = []

    try:
        # Configuration 1: No Re-ranker, No Compression
        print("\n" + "=" * 80)
        print("📋 Configuration 1: Baseline (No Re-ranker, No Compression)")
        print("=" * 80)
        kb1 = HybridKnowledgeBase(use_reranker=False)
        bot1 = TyphoonChatbot(api_key, kb1, use_compression=False)
        result1 = evaluate_rag_system(kb1, bot1, test_cases, use_reranker=False, use_compression=False)
        save_results(result1, "baseline")
        results_all.append(("Baseline", result1))
    except Exception as e:
        print(f"❌ Configuration 1 failed: {e}")
        result1 = None

    try:
        # Configuration 2: With Re-ranker, No Compression
        print("\n" + "=" * 80)
        print("📋 Configuration 2: With Re-ranker Only")
        print("=" * 80)
        kb2 = HybridKnowledgeBase(use_reranker=True)
        bot2 = TyphoonChatbot(api_key, kb2, use_compression=False)
        result2 = evaluate_rag_system(kb2, bot2, test_cases, use_reranker=True, use_compression=False)
        save_results(result2, "reranker_only")
        results_all.append(("Re-ranker Only", result2))
    except Exception as e:
        print(f"❌ Configuration 2 failed: {e}")
        result2 = None

    try:
        # Configuration 3: With Re-ranker and Compression
        print("\n" + "=" * 80)
        print("📋 Configuration 3: Full Stack (Re-ranker + Compression)")
        print("=" * 80)
        kb3 = HybridKnowledgeBase(use_reranker=True)
        bot3 = TyphoonChatbot(api_key, kb3, use_compression=True)
        result3 = evaluate_rag_system(kb3, bot3, test_cases, use_reranker=True, use_compression=True)
        save_results(result3, "full_stack")
        results_all.append(("Full Stack", result3))
    except Exception as e:
        print(f"❌ Configuration 3 failed: {e}")
        result3 = None

    if not result1:
        print("\n⚠️  Cannot compare: Baseline configuration failed")
        return

    # Print comparison
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS COMPARISON")
    print("=" * 80)

    print("\n1️⃣  Baseline (No Re-ranker, No Compression):")
    if result1:
        print(f"   Context Precision: {result1['context_precision']:.4f}")
        print(f"   Context Recall: {result1['context_recall']:.4f}")
        print(f"   Faithfulness: {result1['faithfulness']:.4f}")
        print(f"   Answer Relevancy: {result1['answer_relevancy']:.4f}")

    if result2:
        print("\n2️⃣  With Re-ranker Only:")
        print(f"   Context Precision: {result2['context_precision']:.4f} ({'+' if result2['context_precision'] > result1['context_precision'] else ''}{(result2['context_precision'] - result1['context_precision']):.4f})")
        print(f"   Context Recall: {result2['context_recall']:.4f} ({'+' if result2['context_recall'] > result1['context_recall'] else ''}{(result2['context_recall'] - result1['context_recall']):.4f})")
        print(f"   Faithfulness: {result2['faithfulness']:.4f} ({'+' if result2['faithfulness'] > result1['faithfulness'] else ''}{(result2['faithfulness'] - result1['faithfulness']):.4f})")
        print(f"   Answer Relevancy: {result2['answer_relevancy']:.4f} ({'+' if result2['answer_relevancy'] > result1['answer_relevancy'] else ''}{(result2['answer_relevancy'] - result1['answer_relevancy']):.4f})")

    if result3:
        print("\n3️⃣  Full Stack (Re-ranker + Compression):")
        print(f"   Context Precision: {result3['context_precision']:.4f} ({'+' if result3['context_precision'] > result1['context_precision'] else ''}{(result3['context_precision'] - result1['context_precision']):.4f})")
        print(f"   Context Recall: {result3['context_recall']:.4f} ({'+' if result3['context_recall'] > result1['context_recall'] else ''}{(result3['context_recall'] - result1['context_recall']):.4f})")
        print(f"   Faithfulness: {result3['faithfulness']:.4f} ({'+' if result3['faithfulness'] > result1['faithfulness'] else ''}{(result3['faithfulness'] - result1['faithfulness']):.4f})")
        print(f"   Answer Relevancy: {result3['answer_relevancy']:.4f} ({'+' if result3['answer_relevancy'] > result1['answer_relevancy'] else ''}{(result3['answer_relevancy'] - result1['answer_relevancy']):.4f})")

    print("\n" + "=" * 80)

    return results_all


def main():
    """Main function with improved user interaction"""
    import sys

    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("💡 Please create a .env file with: GEMINI_API_KEY=your_api_key")
        return

    print("=" * 80)
    print("🚀 RAG System Evaluation with Ragas")
    print("=" * 80)
    print("\nChoose evaluation mode:")
    print("1. Quick Evaluation (Full Stack Configuration)")
    print("2. Compare All Configurations (Baseline, Re-ranker, Full Stack)")
    print("3. Custom Configuration")
    print("4. Exit")

    try:
        choice = input("\nEnter your choice (1-4): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n\n👋 Exiting...")
        return

    test_cases = create_test_dataset()

    try:
        if choice == "1":
            # Quick evaluation with full stack
            print("\n📋 Running Quick Evaluation (Full Stack)...")
            kb = HybridKnowledgeBase(use_reranker=True)
            chatbot = TyphoonChatbot(api_key, kb, use_compression=True)
            result = evaluate_rag_system(kb, chatbot, test_cases, use_reranker=True, use_compression=True)

            print("\n" + "=" * 80)
            print("📊 EVALUATION RESULTS")
            print("=" * 80)
            print(f"Context Precision: {result['context_precision']:.4f}")
            print(f"Context Recall: {result['context_recall']:.4f}")
            print(f"Faithfulness: {result['faithfulness']:.4f}")
            print(f"Answer Relevancy: {result['answer_relevancy']:.4f}")
            print("=" * 80)

            save_results(result, "quick_eval_full_stack")

        elif choice == "2":
            # Compare all configurations
            print("\n📋 Running Configuration Comparison...")
            print("⚠️  This will take longer as it runs 3 different configurations")
            compare_configurations()

        elif choice == "3":
            # Custom configuration
            print("\n📋 Custom Configuration")
            try:
                use_reranker = input("Use re-ranker? (y/n): ").strip().lower() == 'y'
                use_compression = input("Use context compression? (y/n): ").strip().lower() == 'y'
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Cancelled...")
                return

            print(f"\n🔧 Configuration: Re-ranker={use_reranker}, Compression={use_compression}")
            kb = HybridKnowledgeBase(use_reranker=use_reranker)
            chatbot = TyphoonChatbot(api_key, kb, use_compression=use_compression)
            result = evaluate_rag_system(kb, chatbot, test_cases, use_reranker=use_reranker, use_compression=use_compression)

            print("\n" + "=" * 80)
            print("📊 EVALUATION RESULTS")
            print("=" * 80)
            print(f"Context Precision: {result['context_precision']:.4f}")
            print(f"Context Recall: {result['context_recall']:.4f}")
            print(f"Faithfulness: {result['faithfulness']:.4f}")
            print(f"Answer Relevancy: {result['answer_relevancy']:.4f}")
            print("=" * 80)

            config_name = f"custom_reranker{use_reranker}_compression{use_compression}"
            save_results(result, config_name)

        elif choice == "4":
            print("\n👋 Exiting...")
            return

        else:
            print("\n❌ Invalid choice. Please run again and select 1-4.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during evaluation: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()
