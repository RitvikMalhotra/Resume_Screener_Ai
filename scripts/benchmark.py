"""Benchmark script: measures throughput on a synthetic 500-resume corpus."""
import sys
import os
import time
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.pipeline import Pipeline, ResumeInput
from utils.config import Config, RetrieverConfig, RerankerConfig, AppConfig

SKILLS = [
    "Python, PyTorch, LLMs, transformers, FAISS",
    "React, TypeScript, Node.js, GraphQL",
    "Kubernetes, Terraform, AWS, CI/CD",
    "SQL, Tableau, Power BI, Excel",
    "Java, Spring Boot, Microservices, Kafka",
    "Data Science, scikit-learn, XGBoost, Spark",
    "iOS, Swift, SwiftUI, Xcode",
    "C++, embedded systems, RTOS",
]

JD = (
    "We are looking for a Machine Learning Engineer with expertise in "
    "LLMs, PyTorch, transformers, and vector search. "
    "You will build, fine-tune, and deploy production ML models."
)


def build_corpus(n: int) -> list[ResumeInput]:
    resumes = []
    for i in range(n):
        skill_set = SKILLS[i % len(SKILLS)]
        resumes.append(
            ResumeInput(
                resume_id=f"r{i:05d}",
                content=(
                    f"Candidate {i}. {i % 15 + 1} years experience. "
                    f"Skills: {skill_set}. "
                    f"Worked at Company {i % 20}. "
                    f"Education: BS Computer Science, University {i % 10}."
                ),
            )
        )
    return resumes


def run_benchmark(n_resumes: int = 500, top_n: int = 10, runs: int = 3):
    cfg = Config(
        retriever=RetrieverConfig(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            batch_size=128,
            top_k=50,
        ),
        reranker=RerankerConfig(
            use_llm_reranker=False,
            cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_batch_size=32,
            top_n=top_n,
        ),
        app=AppConfig(),
    )

    pipeline = Pipeline(config=cfg)
    corpus   = build_corpus(n_resumes)

    timings = []
    for run in range(runs):
        t0 = time.perf_counter()
        response = pipeline.rank(JD, corpus, top_n=top_n)
        elapsed  = time.perf_counter() - t0
        timings.append(elapsed)
        print(
            f"  Run {run+1}: {elapsed:.2f}s | "
            f"preprocess={response.timing['preprocess_s']:.2f}s "
            f"index={response.timing['index_build_s']:.2f}s "
            f"retrieve={response.timing['retrieval_s']:.2f}s "
            f"rerank={response.timing['rerank_s']:.2f}s"
        )

    avg = statistics.mean(timings)
    print(f"\n  Average over {runs} runs: {avg:.2f}s")
    print(f"  Throughput: {n_resumes / avg:.0f} resumes/sec")
    print(f"  Budget compliance (120s): {'✓ PASS' if avg < 120 else '✗ FAIL'}")
    return avg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Resume screener benchmark")
    parser.add_argument("--n", type=int, default=500, help="Number of resumes")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Benchmark: {args.n} resumes, top-{args.top_n}, {args.runs} runs")
    print(f"{'='*60}")
    run_benchmark(args.n, args.top_n, args.runs)
