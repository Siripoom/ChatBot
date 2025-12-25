# RAGAS Evaluation for Chatbot v04

This document explains how to evaluate your RAG chatbot using RAGAS (Retrieval Augmented Generation Assessment).

## What is RAGAS?

RAGAS is a framework for evaluating Retrieval Augmented Generation (RAG) systems. It provides several metrics to assess the quality of your chatbot's retrieval and generation capabilities.

## Metrics Explained

### 1. **Faithfulness** (โมเต็มความถูกต้องตามบริบท)
- Measures how factually accurate the answer is based on the retrieved context
- Score: 0.0 to 1.0 (higher is better)
- **1.0 = Perfect**: All claims in the answer are supported by the context
- **< 0.5**: Answer contains hallucinations or unsupported claims

### 2. **Answer Relevancy** (ความเกี่ยวข้องของคำตอบ)
- Measures how relevant the answer is to the question
- Score: 0.0 to 1.0 (higher is better)
- **1.0 = Perfect**: Answer directly addresses the question
- **< 0.5**: Answer is off-topic or doesn't address the question

### 3. **Context Precision** (ความแม่นยำของบริบท)
- Measures how precise the top-ranked retrieved contexts are
- Score: 0.0 to 1.0 (higher is better)
- **1.0 = Perfect**: All top contexts are highly relevant
- **< 0.5**: Many irrelevant contexts in top positions

### 4. **Context Recall** (ความครอบคลุมของบริบท)
- Measures how well the retrieved contexts cover the ground truth answer
- Score: 0.0 to 1.0 (higher is better)
- **1.0 = Perfect**: Retrieved contexts contain all information needed
- **< 0.5**: Important information is missing from contexts

### 5. **Answer Similarity** (ความคล้ายคลึงของคำตอบ)
- Measures semantic similarity between generated and ground truth answers
- Score: 0.0 to 1.0 (higher is better)
- Uses embeddings to compare answers

### 6. **Answer Correctness** (ความถูกต้องรวม)
- Combines factual correctness and semantic similarity
- Score: 0.0 to 1.0 (higher is better)
- Most comprehensive metric for answer quality

## Installation

### Step 1: Install dependencies

```bash
pip install -r requirements_ragas.txt
```

### Step 2: Set up OpenAI API Key

RAGAS requires OpenAI API for evaluation. Add your OpenAI API key to `.env`:

```bash
# .env file
TYPHOON_API_KEY=your_typhoon_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Required for RAGAS
```

**Note**: You can get an OpenAI API key from https://platform.openai.com/api-keys

## Usage

### Basic Evaluation

Run the evaluation with default settings (core metrics only):

```bash
python test_ragas.py
```

This will:
1. Load your chatbot
2. Run 10 test questions
3. Retrieve contexts for each question
4. Generate answers
5. Evaluate using RAGAS metrics
6. Display and save results

### Full Evaluation

For comprehensive evaluation with all metrics:

```bash
python test_ragas.py
# Then choose option "1" when prompted
```

**Warning**: Full evaluation uses more OpenAI API calls and will cost more.

## Customizing Test Cases

Edit the `create_test_dataset()` method in [test_ragas.py](test_ragas.py) to add your own test cases:

```python
test_data = [
    {
        "question": "Your question here",
        "ground_truth": "Expected answer here"
    },
    # Add more test cases...
]
```

### Tips for Good Test Cases:

1. **Cover diverse topics** from your knowledge base
2. **Include different question types**: factual, comparison, how-to, etc.
3. **Write realistic ground truths**: What a good answer should include
4. **Start with 10-20 questions**, then expand based on results

## Understanding Results

### Score Interpretation

| Score Range | Rating | Action |
|------------|--------|---------|
| 0.8 - 1.0 | 🟢 Excellent | Keep doing what you're doing! |
| 0.6 - 0.8 | 🟡 Good | Minor improvements possible |
| 0.4 - 0.6 | 🟠 Fair | Needs attention and tuning |
| < 0.4 | 🔴 Poor | Significant improvements needed |

### Common Issues and Solutions

#### Low Faithfulness (<0.6)
**Problem**: Chatbot is hallucinating or making unsupported claims

**Solutions**:
- Strengthen RAG prompts to stick to context
- Add "cite sources" instruction
- Reduce temperature in generation
- Improve context quality

#### Low Answer Relevancy (<0.6)
**Problem**: Answers don't address the question

**Solutions**:
- Improve query expansion in `expand_query()`
- Tune retrieval weights (BM25, vector, keyword)
- Check if contexts are too generic

#### Low Context Recall (<0.6)
**Problem**: Retrieved contexts missing important information

**Solutions**:
- Increase `n_results` in retrieval
- Improve BM25/vector weights
- Add more documents to knowledge base
- Check keyword matching effectiveness

#### Low Context Precision (<0.6)
**Problem**: Retrieved contexts include irrelevant information

**Solutions**:
- Tune re-ranking with cross-encoder
- Adjust retrieval weights
- Improve query expansion
- Filter low-score contexts

## Output Files

After evaluation, you'll get:

### `ragas_results.json`
Complete evaluation results in JSON format:
```json
{
  "metrics": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78,
    ...
  },
  "timestamp": "2025-12-25T10:30:00"
}
```

## Cost Estimation

RAGAS uses OpenAI API for evaluation:

- **Core metrics** (3 metrics): ~$0.01-0.02 per question
- **Full metrics** (6 metrics): ~$0.03-0.05 per question

For 10 questions:
- Core: ~$0.10-0.20
- Full: ~$0.30-0.50

## Advanced Usage

### Batch Evaluation

To evaluate multiple configurations:

```python
# Test different retrieval weights
configs = [
    {"bm25": 0.3, "vector": 0.5, "keyword": 0.2},
    {"bm25": 0.2, "vector": 0.6, "keyword": 0.2},
    {"bm25": 0.4, "vector": 0.4, "keyword": 0.2},
]

for config in configs:
    # Modify search_knowledge parameters
    # Run evaluation
    # Compare results
```

### Per-Question Analysis

To see detailed results for each question:

```python
# Modify the evaluate() method to return per-question scores
# This helps identify which questions perform poorly
```

## Troubleshooting

### Error: "OPENAI_API_KEY not found"
**Solution**: Add your OpenAI API key to `.env` file

### Error: "Module 'ragas' not found"
**Solution**: Run `pip install -r requirements_ragas.txt`

### Error: "ChromaDB collection not found"
**Solution**: Make sure you've initialized your vector database first

### API Rate Limits
**Solution**: Add delays between evaluations or use a paid OpenAI account

## Best Practices

1. **Start Small**: Test with 5-10 questions first
2. **Iterate**: Run evaluation → analyze → improve → repeat
3. **Track Progress**: Save results over time to see improvements
4. **Focus on Weak Areas**: Prioritize fixing lowest-scoring metrics
5. **Use Real Questions**: Include actual user questions in test set

## References

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS GitHub](https://github.com/explodinggradients/ragas)
- [RAG Evaluation Best Practices](https://docs.ragas.io/en/latest/concepts/metrics/)

## Support

For issues or questions:
- Check RAGAS documentation
- Review your chatbot logs
- Verify API keys are correctly set
- Check test case format matches expectations
