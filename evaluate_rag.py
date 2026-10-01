"""
RAG Evaluation Framework
========================
Evaluates RAG pipeline performance on 5 benchmark questions derived from the
WHO "Substances under Surveillance" report
(who_substances_surveillance.pdf) using retrieval against Qdrant Cloud vector
database and Groq LLM (openai/gpt-oss-120b).

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
import math
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
        "question": "What is the purpose of the WHO ECDD Surveillance List, and what happens once a substance is placed on it?",
        "ground_truth": (
            "A substance is placed on the WHO ECDD Surveillance List if the Committee considers the evidence of the impact "
            "of a new psychoactive substance (NPS) in causing substantial harm to health too scarce to recommend placement "
            "under international control. If the substance has therapeutic applications (is a psychotropic medicine), the "
            "Committee weighs the therapeutic benefits against the evidence of harm, considering the availability of "
            "alternative medicines. The ECDD Secretariat then actively monitors whether additional data on the harm of the "
            "substance becomes available to justify a subsequent critical review."
        ),
    },
    {
        "id": "Q2",
        "question": "What adverse effects and abuse-related harms are reported for gabapentin, and what is its surveillance status under the ECDD?",
        "ground_truth": (
            "Gabapentin is used therapeutically as an anticonvulsant or antiepileptic drug and may also be used in the "
            "treatment of nerve pain. It has been brought to WHO's attention that gabapentin may be being misused in some "
            "Member States. Adverse effects associated with gabapentin include hypoventilation, respiratory failure, "
            "myopathy, self-harm behaviour, suicidal behaviour, somnolence, dizziness and drowsiness. Gabapentin has been "
            "associated with several cases of abuse and drug-related harm (for example suicide). To date, gabapentin has "
            "not been pre- or critically reviewed by the ECDD. It was added to the surveillance list by the 2nd Working "
            "Group meeting (2017)."
        ),
    },
    {
        "id": "Q3",
        "question": "What are the reported effects of 4-Fluoromethcathinone (flephedrone; 4-FMC) and what did the 36th ECDD recommend for it?",
        "ground_truth": (
            "4-Fluoromethcathinone (flephedrone; 4-FMC) is being misused in a number of Member States, is clandestinely "
            "manufactured and has been identified in seized products. 4-FMC produces effects similar to psychomotor "
            "stimulants such as cocaine and methamphetamine, although it appears to be less potent than methamphetamine. "
            "It has been associated with a few fatal and non-fatal intoxications. Owing to the insufficiency of data "
            "regarding dependence, abuse and risks to public health, the 36th ECDD recommended that 4-FMC not be placed "
            "under international control at this time but be kept under surveillance."
        ),
    },
    {
        "id": "Q4",
        "question": "Why did the 41st ECDD decide not to schedule tramadol, and what did it recommend instead?",
        "ground_truth": (
            "The 41st ECDD was strongly of the view that the extent of tramadol abuse and the evidence of public health "
            "risks associated with tramadol warranted consideration of scheduling. However, it recommended that tramadol "
            "not be scheduled at this time in order to avoid an adverse impact on access to this medication, especially in "
            "countries where tramadol may be the only available opioid analgesic, or in crisis situations where there is "
            "little or no access at all to other opioids. The 41st ECDD also strongly urged WHO and its partners to address "
            "the grossly inadequate access to and availability of opioid pain medication in low-income countries, and "
            "recommended that the WHO Secretariat continue to keep tramadol under surveillance, collect information on the "
            "extent of problems associated with tramadol misuse and its medical use, and consider tramadol for review at a "
            "future meeting."
        ),
    },
    {
        "id": "Q5",
        "question": "What adverse effects are associated with AMT (Alpha-methyltryptamine) and what did the 36th ECDD recommend for it?",
        "ground_truth": (
            "AMT (Alpha-methyltryptamine) is a tryptamine derivative that shares several similarities with the Schedule I "
            "tryptamine hallucinogens. Adverse effects of AMT include mild increases in blood pressure or respiration "
            "rate, restlessness, tachycardia, severe nausea, severe vomiting, impaired coordination, and visual and "
            "auditory disturbances and distortions. AMT has been associated with fatal intoxications, although other "
            "drugs were present. The 36th ECDD (June 2014) recommended that, due to the insufficiency of evidence required "
            "to satisfy the criteria for international scheduling under the Conventions, AMT not be placed under "
            "international control but be kept under surveillance."
        ),
    },
]

# -------------------------------------------------------------
# 2. DENSE-ONLY BENCHMARK GENERATION PROMPT (independent of the app UI)
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
    Obtain validated LLM-judge estimates without substituting fabricated scores.
    Retry invalid outputs and transient provider errors with rate-limit backoff.
    """
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
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
            if isinstance(scores, dict) and all(k in scores for k in required_keys):
                for k in required_keys:
                    value = scores[k]
                    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"Invalid judge metric {k}: {value!r}")
                    scores[k] = float(value)
                if "reasoning" not in scores or not scores["reasoning"]:
                    scores["reasoning"] = "Evaluated directly by LLM judge."
                return scores
            else:
                raise ValueError("Judge output must be an object containing all four metrics")

        except Exception as e:
            err_msg = str(e)
            print(f"  [!] Judge evaluation attempt {attempt + 1}/{max_retries} encountered error: {err_msg[:140]}")

            # Calculate sleep duration (respect Groq rate limit reset if provided)
            wait_sec = 10 * (attempt + 1)
            time_match = re.search(r"try again in (\d+(?:\.\d+)?)s", err_msg)
            if time_match:
                wait_sec = max(wait_sec, float(time_match.group(1)) + 2.0)

            if attempt + 1 < max_retries:
                print(f"  -> Pausing {wait_sec:.1f}s before retrying...")
                time.sleep(wait_sec)

    raise RuntimeError(
        "Evaluation failed to obtain exact LLM judge output after retries. Fallback is disabled."
    )


def _run_rag_evaluation(client):
    print("=" * 70)
    print("[*] NEXUS-AI RAG PIPELINE BENCHMARK EVALUATION")
    print("=" * 70)

    qdrant_url = os.getenv("QDRANT_URL")
    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        raise ValueError("GROQ_API_KEY not found in environment!")
    if not qdrant_url:
        raise ValueError("QDRANT_URL is required for the cloud evaluation.")
    collection_name = os.getenv("QDRANT_COLLECTION", "who_hybrid")

    print(f"Connecting to Qdrant Cloud: {qdrant_url}")
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    # Generation LLM
    generator_llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.1,
        timeout=60,
        max_retries=2,
    )
    rag_chain = RAG_SYSTEM_PROMPT | generator_llm | StrOutputParser()

    # Judge LLM with native JSON mode for exact metric evaluation
    judge_llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.0,
        timeout=60,
        max_retries=2,
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

        print(f"  -> Judge Scores: Faithfulness: {f:.2f} | Relevancy: {ar:.2f} | Recall: {cr:.2f} | Precision: {cp:.2f} => Overall: {composite:.3f}")
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
    out_dir = Path(__file__).resolve().parent / "eval_results"
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
**Vector Store:** Qdrant Cloud (`{collection_name}`)

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


def run_rag_evaluation():
    url = os.getenv("QDRANT_URL")
    if not url or not os.getenv("GROQ_API_KEY"):
        raise ValueError("QDRANT_URL and GROQ_API_KEY are required for cloud evaluation.")
    client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=30)
    try:
        return _run_rag_evaluation(client)
    finally:
        client.close()


if __name__ == "__main__":
    run_rag_evaluation()
