"""
Quick RAGAS Test - Minimal Example
===================================
This is a simplified version for quick testing with just 3 questions.
Use this to verify everything works before running the full evaluation.
"""

import os
from dotenv import load_dotenv
from test_ragas import RAGASEvaluator
from chatbot_v04_keywords import HybridKnowledgeBase, TyphoonChatbot


def quick_test():
    """Run a quick RAGAS test with 3 questions"""

    # Load environment variables
    load_dotenv()

    typhoon_api_key = os.getenv('TYPHOON_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if not typhoon_api_key or not openai_api_key:
        print("❌ Please set TYPHOON_API_KEY and OPENAI_API_KEY in .env file")
        return

    print("="*80)
    print("🚀 Quick RAGAS Test - 3 Questions")
    print("="*80)

    # Initialize chatbot
    print("\n🔧 Loading chatbot...")
    kb = HybridKnowledgeBase(
        persist_directory="./chroma_db",
        collection_name="chatbot_knowledge",
        use_keyword_boost=True
    )
    chatbot = TyphoonChatbot(typhoon_api_key, kb)

    # Create evaluator
    evaluator = RAGASEvaluator(chatbot, openai_api_key)

    # Quick test dataset with only 3 questions
    test_data = [
        {
            "question": "หลักสูตรวิศวกรรมโยธาและการศึกษาเรียนกี่ปี",
            "ground_truth": "หลักสูตรวิศวกรรมโยธาและการศึกษาใช้เวลาเรียน 5 ปี"
        },
        {
            "question": "จบแล้วได้ปริญญาอะไร",
            "ground_truth": "ได้รับปริญญาวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมโยธา"
        },
        {
            "question": "มีการฝึกประสบการณ์วิชาชีพหรือไม่",
            "ground_truth": "มีการฝึกปฏิบัติการสอนและการฝึกงานในสถานประกอบการ"
        }
    ]

    print(f"📝 Testing with {len(test_data)} questions\n")

    # Run evaluation (core metrics only for speed)
    results = evaluator.evaluate(test_data, use_all_metrics=False)

    # Print results
    evaluator.print_results(results)

    # Calculate average score
    avg_score = sum(results.values()) / len(results)
    print(f"\n📈 Average Score: {avg_score:.4f}")

    if avg_score >= 0.7:
        print("✅ Great! Your chatbot is performing well!")
    elif avg_score >= 0.5:
        print("⚠️  Good start, but there's room for improvement")
    else:
        print("❌ Needs significant improvement - check your retrieval and prompts")

    print("\n💡 Tip: Run 'python test_ragas.py' for full evaluation with 10 questions")


if __name__ == "__main__":
    quick_test()
