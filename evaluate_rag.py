"""
RAG Evaluation Framework
========================
Evaluates RAG pipeline performance on 5 benchmark domain questions using
retrieval against Qdrant Cloud vector database and Groq LLM (openai/gpt-oss-120b).

Metrics Evaluated:
1. Faithfulness (Groundedness / Hallucination-free score)
2. Answer Relevancy (Question-Answer direct relevance)
3. Context Recall (Ground truth information captured in context)
4. Context Precision (Signal-to-noise ratio of retrieved chunks)
5. Overall RAG Composite Score
"""

import os
import sys
import json
import time
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Ensure safe UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------------
# 1. EVALUATION DATASET (5 High-Fidelity Domain Benchmark Questions)
# -------------------------------------------------------------
BENCHMARK_TESTSET = [
    {
        "id": "Q1",
        "question": "What prompt instructions and background style are specified for the hair oil brand image post generation in Topic 19?",
        "ground_truth": (
            "In Topic 19, the prompt instructs to create a premium Instagram post design for a hair oil brand. "
            "Use the provided product image exactly as it is without altering the product. Change the background "
            "to keep the design clean and luxurious, and add the text: 'Introducing Hair Oil'."
        ),
    },
    {
        "id": "Q2",
        "question": "What are the ad copy requirements and call-to-action phrases for the men's perfume 'Junoon' Google Search Ad Campaign in Topic 93?",
        "ground_truth": (
            "In Topic 93, the prompt asks to act as an expert Google Ads copywriter writing high-converting search ad copy "
            "for a men's perfume called 'Junoon' inspired by aromastudio. It must include strong call-to-action phrases "
            "such as 'Shop Now', 'Order Today', and 'Try Junoon', targeting men looking for premium perfumes."
        ),
    },
    {
        "id": "Q3",
        "question": "What role and years of experience are specified for the Facebook Marketing Strategy prompt in Topic 56?",
        "ground_truth": (
            "In Topic 56, the prompt specifies acting as a senior Meta (Facebook) Ads strategist with 8+ years of experience "
            "managing high-performing e-commerce and lead generation campaigns to create a complete step-by-step Facebook paid ads strategy."
        ),
    },
    {
        "id": "Q4",
        "question": "What animation effects and background aesthetic are requested for the perfume promotional video in Topic 21?",
        "ground_truth": (
            "In Topic 21, the prompt instructs to create a cinematic product video turning a static perfume image into smooth animations "
            "with slow zoom, subtle rotation, light reflections, and floating particles or mist effect, using a luxury-inspired background "
            "with dark elegant tones like black, navy, or deep brown with soft lighting."
        ),
    },
    {
        "id": "Q5",
        "question": "What landing page destination URL and previous campaign challenges are noted for the TikTok Ads Strategy in Topic 85?",
        "ground_truth": (
            "In Topic 85, the destination URL is www.digiskills.pk, and the business noted that they previously ran some TikTok campaigns "
            "but experienced poor ROAS. The prompt asks a senior performance marketing strategist with 8+ years experience to build a detailed strategy."
        ),
    },
]

# -------------------------------------------------------------
# 2. RAG GENERATION PROMPT (Matching Production NexusAI Pipeline)
# -------------------------------------------------------------
RAG_SYSTEM_PROMPT = ChatPromptTemplate.from_template(
    """You are NexusAI, an accurate and concise enterprise knowledge assistant.

Core Directives:
1. Direct & Concise: Answer ONLY what the user explicitly asks for.
2. Strict Factual Accuracy: Base your answer directly on the provided Context. Do not invent information.
3. If the context does not contain enough info, state clearly what is missing.

Context:
{context}

Question: {question}

Concise & Accurate Answer:"""
)

# -------------------------------------------------------------
# 3. LLM-AS-A-JUDGE METRIC EVALUATORS
# -------------------------------------------------------------
JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert impartial evaluator of Retrieval-Augmented Generation (RAG) systems.
Analyze the following Question, Retrieved Context, Ground Truth, and Generated Answer.

Question: {question}
Ground Truth: {ground_truth}
Retrieved Context:
{context}
Generated Answer: {answer}

Evaluate the response across these 4 standard RAG metrics on a scale from 0.0 to 1.0:

1. faithfulness (0.0 to 1.0): Are all statements and facts in the Generated Answer directly supported by the Retrieved Context? (1.0 = 100% grounded with zero hallucination).
2. answer_relevancy (0.0 to 1.0): Does the Generated Answer directly, concisely, and completely address the Question asked?
3. context_recall (0.0 to 1.0): Does the Retrieved Context contain the necessary factual points specified in the Ground Truth?
4. context_precision (0.0 to 1.0): How relevant and signal-rich is the Retrieved Context to answering the question (low noise)?

Provide your judgment in STRICT JSON format with no additional text or Markdown code fences:
{{
  "faithfulness": <float between 0.0 and 1.0>,
  "answer_relevancy": <float between 0.0 and 1.0>,
  "context_recall": <float between 0.0 and 1.0>,
  "context_precision": <float between 0.0 and 1.0>,
  "reasoning": "<concise 1-2 sentence evaluation summary>"
}}"""
)


def evaluate_sample_with_judge(judge_chain, payload: dict, max_retries: int = 6) -> dict:
    """
    Executes exact LLM-as-a-judge evaluation without any fallback.
    Retries gracefully with rate-limit backoff until exact scores and reasoning are obtained.
    """
    for attempt in range(max_retries):
        try:
            judge_raw = judge_chain.invoke(payload).strip()

            # Clean markdown code blocks if present
            cleaned = judge_raw
            if "```" in cleaned:
                m = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
                if m:
                    cleaned = m.group(1).strip()
                else:
                    lines = cleaned.split("\n")
                    cleaned = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

            # Extract outermost JSON block
            m_brace = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if m_brace:
                cleaned = m_brace.group(1).strip()

            scores = json.loads(cleaned)

            required_keys = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
            if all(k in scores for k in required_keys):
                # Ensure values are valid floats clamped in [0.0, 1.0]
                for k in required_keys:
                    scores[k] = max(0.0, min(1.0, float(scores[k])))
                if "reasoning" not in scores or not scores["reasoning"]:
                    scores["reasoning"] = "Evaluated directly by LLM judge."
                return scores
            else:
                print(f"  [!] Missing expected keys in judge output: {list(scores.keys())}. Retrying...")

        except Exception as e:
            err_msg = str(e)
            print(f"  [!] Judge evaluation attempt {attempt + 1}/{max_retries} encountered error: {err_msg[:140]}")

            # Calculate sleep duration (respect Groq rate limit reset if provided)
            wait_sec = 10 * (attempt + 1)
            time_match = re.search(r"try again in (\d+(?:\.\d+)?)s", err_msg)
            if time_match:
                wait_sec = max(wait_sec, float(time_match.group(1)) + 2.0)

            print(f"  -> Pausing {wait_sec:.1f}s before retrying to ensure exact results...")
            time.sleep(wait_sec)

    raise RuntimeError(
        "Evaluation failed to obtain exact LLM judge output after retries. Fallback is disabled."
    )


def run_rag_evaluation():
    print("=" * 70)
    print("[*] NEXUS-AI RAG PIPELINE BENCHMARK EVALUATION")
    print("=" * 70)

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        raise ValueError("GROQ_API_KEY not found in environment!")

    print(f"Connecting to Qdrant Cloud: {qdrant_url}")
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="enterprise_knowledge_base",
        embedding=embeddings,
    )

    # Generation LLM
    generator_llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.1,
    )
    rag_chain = RAG_SYSTEM_PROMPT | generator_llm | StrOutputParser()

    # Judge LLM with native JSON mode for exact metric evaluation
    judge_llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    judge_chain = JUDGE_PROMPT | judge_llm | StrOutputParser()

    results = []
    print("\nStarting evaluation of 5 benchmark questions...\n" + "-" * 70)

    for item in BENCHMARK_TESTSET:
        qid = item["id"]
        question = item["question"]
        gt = item["ground_truth"]

        print(f"\n[{qid}] Question: {question}")

        # Add pause between questions to respect Groq rate limits
        time.sleep(4)

        # Step A: Vector Retrieval (Top k=8)
        retrieved_docs = vectorstore.similarity_search(question, k=8)
        context_str = "\n\n---\n\n".join([d.page_content for d in retrieved_docs])
        print(f"  -> Retrieved Chunks: {len(retrieved_docs)} (Total Characters: {len(context_str)})")

        # Step B: Answer Generation
        gen_answer = rag_chain.invoke({"context": context_str, "question": question}).strip()
        print(f"  -> Generated Answer: {gen_answer[:120]}...")

        # Step C: Exact Metric Evaluation via LLM Judge (Zero Fallback)
        time.sleep(3)
        scores = evaluate_sample_with_judge(
            judge_chain,
            {
                "question": question,
                "ground_truth": gt,
                "context": context_str,
                "answer": gen_answer,
            },
        )

        f = float(scores["faithfulness"])
        ar = float(scores["answer_relevancy"])
        cr = float(scores["context_recall"])
        cp = float(scores["context_precision"])
        composite = round((f + ar + cr + cp) / 4.0, 3)

        print(f"  -> Exact Scores: Faithfulness: {f:.2f} | Relevancy: {ar:.2f} | Recall: {cr:.2f} | Precision: {cp:.2f} => Overall: {composite:.3f}")
        print(f"  -> Judge Reasoning: {scores.get('reasoning', 'N/A')}")

        results.append({
            "id": qid,
            "question": question,
            "ground_truth": gt,
            "answer": gen_answer,
            "faithfulness": f,
            "answer_relevancy": ar,
            "context_recall": cr,
            "context_precision": cp,
            "overall_score": composite,
            "reasoning": scores.get("reasoning", ""),
        })

    # Summary Aggregates
    avg_f = round(sum(r["faithfulness"] for r in results) / len(results), 3)
    avg_ar = round(sum(r["answer_relevancy"] for r in results) / len(results), 3)
    avg_cr = round(sum(r["context_recall"] for r in results) / len(results), 3)
    avg_cp = round(sum(r["context_precision"] for r in results) / len(results), 3)
    avg_overall = round((avg_f + avg_ar + avg_cr + avg_cp) / 4.0, 3)

    print("\n" + "=" * 70)
    print("[+] FINAL BENCHMARK EVALUATION SUMMARY SCORECARD")
    print("=" * 70)
    print(f"Total Questions Evaluated : {len(results)}")
    print(f"Average Faithfulness      : {avg_f * 100:.1f}% (Groundedness / Hallucination-free)")
    print(f"Average Answer Relevancy  : {avg_ar * 100:.1f}% (Direct question-to-answer alignment)")
    print(f"Average Context Recall    : {avg_cr * 100:.1f}% (Ground-truth facts retrieved)")
    print(f"Average Context Precision : {avg_cp * 100:.1f}% (Signal-to-noise retrieval ratio)")
    print("-" * 70)
    print(f"[RESULT] COMPOSITE RAG QUALITY SCORE: {avg_overall * 100:.1f}% / 100%")
    print("=" * 70)


    # Save to eval_results directory
    out_dir = Path("eval_results")
    out_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json_path = out_dir / f"eval_report_{timestamp}.json"
    report_md_path = out_dir / f"eval_report_{timestamp}.md"

    summary_data = {
        "timestamp": timestamp,
        "model": "openai/gpt-oss-120b",
        "embeddings": "all-MiniLM-L6-v2",
        "retriever": "Qdrant (k=8, Cosine)",
        "averages": {
            "faithfulness": avg_f,
            "answer_relevancy": avg_ar,
            "context_recall": avg_cr,
            "context_precision": avg_cp,
            "overall_score": avg_overall,
        },
        "questions": results,
    }

    with open(report_json_path, "w", encoding="utf-8") as fj:
        json.dump(summary_data, fj, indent=2, ensure_ascii=False)

    md_content = f"""# ⚡ NexusAI — RAG Benchmark Evaluation Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**LLM Evaluated:** `openai/gpt-oss-120b` (via Groq)  
**Embeddings:** `all-MiniLM-L6-v2` (Dense 384-dim)  
**Vector Store:** Qdrant Cloud (`enterprise_knowledge_base`)  

---

## 📊 Summary Metrics

| Metric | Score | Benchmark Target | Status |
| :--- | :--- | :--- | :--- |
| **Faithfulness** | **{avg_f * 100:.1f}%** | ≥ 80.0% | {'✅ Passed' if avg_f >= 0.8 else '⚠️ Needs Review'} |
| **Answer Relevancy** | **{avg_ar * 100:.1f}%** | ≥ 80.0% | {'✅ Passed' if avg_ar >= 0.8 else '⚠️ Needs Review'} |
| **Context Recall** | **{avg_cr * 100:.1f}%** | ≥ 75.0% | {'✅ Passed' if avg_cr >= 0.75 else '⚠️ Needs Review'} |
| **Context Precision** | **{avg_cp * 100:.1f}%** | ≥ 75.0% | {'✅ Passed' if avg_cp >= 0.75 else '⚠️ Needs Review'} |
| **Overall RAG Score** | **{avg_overall * 100:.1f}%** | ≥ 80.0% | {'🏆 Optimal' if avg_overall >= 0.8 else '⚠️ Sub-optimal'} |

---

## 📝 Detailed Question-by-Question Results

"""
    for r in results:
        md_content += f"""### [{r['id']}] {r['question']}
- **Ground Truth:** {r['ground_truth']}
- **Generated Answer:** {r['answer']}
- **Faithfulness:** `{r['faithfulness'] * 100:.1f}%`
- **Relevancy:** `{r['answer_relevancy'] * 100:.1f}%`
- **Recall:** `{r['context_recall'] * 100:.1f}%`
- **Precision:** `{r['context_precision'] * 100:.1f}%`
- **Overall:** `{r['overall_score'] * 100:.1f}%`
- **Judge Reasoning:** *{r['reasoning']}*

---
"""

    with open(report_md_path, "w", encoding="utf-8") as fm:
        fm.write(md_content)

    print(f"\nSaved JSON report to: {report_json_path}")
    print(f"Saved Markdown report to: {report_md_path}")
    return summary_data


if __name__ == "__main__":
    run_rag_evaluation()
