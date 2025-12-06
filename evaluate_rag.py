"""
RAG Evaluation Script using Ragas

This script evaluates the RAG chatbot using various metrics:
- Context Precision: How relevant are the retrieved contexts?
- Context Recall: Does the context contain the answer?
- Faithfulness: Is the answer grounded in the context?
- Answer Relevance: Is the answer relevant to the question?
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from chatbot import HybridKnowledgeBase, GeminiChatbot


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


def evaluate_rag_system(
    knowledge_base: HybridKnowledgeBase,
    chatbot: GeminiChatbot,
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

        # Get relevant contexts
        relevant_knowledge = knowledge_base.search_knowledge(question, n_results=5)
        context_list = [item['text'] for item in relevant_knowledge]

        # Apply compression if enabled
        if use_compression:
            relevant_knowledge = chatbot.compress_context(question, relevant_knowledge)
            context_list = [item['text'] for item in relevant_knowledge]

        # Get chatbot answer
        answer = chatbot.chat(question)

        print(f"   Answer: {answer[:100]}...")
        print(f"   Contexts: {len(context_list)} items")

        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)
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

    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    return result


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

    # Configuration 1: No Re-ranker, No Compression
    print("\n" + "=" * 80)
    print("📋 Configuration 1: Baseline (No Re-ranker, No Compression)")
    print("=" * 80)
    kb1 = HybridKnowledgeBase(use_reranker=False)
    bot1 = GeminiChatbot(api_key, kb1, use_compression=False)
    result1 = evaluate_rag_system(kb1, bot1, test_cases, use_reranker=False, use_compression=False)

    # Configuration 2: With Re-ranker, No Compression
    print("\n" + "=" * 80)
    print("📋 Configuration 2: With Re-ranker Only")
    print("=" * 80)
    kb2 = HybridKnowledgeBase(use_reranker=True)
    bot2 = GeminiChatbot(api_key, kb2, use_compression=False)
    result2 = evaluate_rag_system(kb2, bot2, test_cases, use_reranker=True, use_compression=False)

    # Configuration 3: With Re-ranker and Compression
    print("\n" + "=" * 80)
    print("📋 Configuration 3: Full Stack (Re-ranker + Compression)")
    print("=" * 80)
    kb3 = HybridKnowledgeBase(use_reranker=True)
    bot3 = GeminiChatbot(api_key, kb3, use_compression=True)
    result3 = evaluate_rag_system(kb3, bot3, test_cases, use_reranker=True, use_compression=True)

    # Print comparison
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS COMPARISON")
    print("=" * 80)

    print("\n1️⃣  Baseline (No Re-ranker, No Compression):")
    print(f"   Context Precision: {result1['context_precision']:.4f}")
    print(f"   Context Recall: {result1['context_recall']:.4f}")
    print(f"   Faithfulness: {result1['faithfulness']:.4f}")
    print(f"   Answer Relevancy: {result1['answer_relevancy']:.4f}")

    print("\n2️⃣  With Re-ranker Only:")
    print(f"   Context Precision: {result2['context_precision']:.4f} ({'+' if result2['context_precision'] > result1['context_precision'] else ''}{(result2['context_precision'] - result1['context_precision']):.4f})")
    print(f"   Context Recall: {result2['context_recall']:.4f} ({'+' if result2['context_recall'] > result1['context_recall'] else ''}{(result2['context_recall'] - result1['context_recall']):.4f})")
    print(f"   Faithfulness: {result2['faithfulness']:.4f} ({'+' if result2['faithfulness'] > result1['faithfulness'] else ''}{(result2['faithfulness'] - result1['faithfulness']):.4f})")
    print(f"   Answer Relevancy: {result2['answer_relevancy']:.4f} ({'+' if result2['answer_relevancy'] > result1['answer_relevancy'] else ''}{(result2['answer_relevancy'] - result1['answer_relevancy']):.4f})")

    print("\n3️⃣  Full Stack (Re-ranker + Compression):")
    print(f"   Context Precision: {result3['context_precision']:.4f} ({'+' if result3['context_precision'] > result1['context_precision'] else ''}{(result3['context_precision'] - result1['context_precision']):.4f})")
    print(f"   Context Recall: {result3['context_recall']:.4f} ({'+' if result3['context_recall'] > result1['context_recall'] else ''}{(result3['context_recall'] - result1['context_recall']):.4f})")
    print(f"   Faithfulness: {result3['faithfulness']:.4f} ({'+' if result3['faithfulness'] > result1['faithfulness'] else ''}{(result3['faithfulness'] - result1['faithfulness']):.4f})")
    print(f"   Answer Relevancy: {result3['answer_relevancy']:.4f} ({'+' if result3['answer_relevancy'] > result1['answer_relevancy'] else ''}{(result3['answer_relevancy'] - result1['answer_relevancy']):.4f})")

    print("\n" + "=" * 80)


def main():
    """Main function"""
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        return

    print("=" * 80)
    print("🚀 RAG System Evaluation with Ragas")
    print("=" * 80)

    # Option 1: Quick evaluation with current configuration
    print("\n📋 Running quick evaluation...")
    kb = HybridKnowledgeBase(use_reranker=True)
    chatbot = GeminiChatbot(api_key, kb, use_compression=True)
    test_cases = create_test_dataset()
    result = evaluate_rag_system(kb, chatbot, test_cases)

    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    print(f"Context Precision: {result['context_precision']:.4f}")
    print(f"Context Recall: {result['context_recall']:.4f}")
    print(f"Faithfulness: {result['faithfulness']:.4f}")
    print(f"Answer Relevancy: {result['answer_relevancy']:.4f}")
    print("=" * 80)

    # Option 2: Compare different configurations (uncomment to run)
    # print("\n🔄 Would you like to compare different configurations? (This will take longer)")
    # response = input("Enter 'yes' to compare: ").strip().lower()
    # if response == 'yes':
    #     compare_configurations()


if __name__ == "__main__":
    main()
