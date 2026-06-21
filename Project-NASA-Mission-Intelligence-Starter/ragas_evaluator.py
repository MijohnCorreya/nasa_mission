import os
import json
from typing import Dict, List, Any

from rag_client import (
    discover_chroma_backends,
    initialize_rag_system,
    retrieve_documents,
    format_context,
)

from llm_client import generate_response

try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass




def load_test_questions(file_path: str = "test_questions.json") -> List[Dict]:
    """Load evaluation dataset from JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    reference: str = ""
) -> Dict[str, Any]:
    """Evaluate response quality using RAGAS metrics."""
    try:
        from ragas import SingleTurnSample
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import ResponseRelevancy, Faithfulness
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as e:
        return {"error": f"RAGAS import failed: {str(e)}"}

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CHROMA_OPENAI_API_KEY")
    if not api_key:
        return {"error": "Missing OPENAI_API_KEY or CHROMA_OPENAI_API_KEY"}

    try:
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=api_key,
                base_url="https://openai.vocareum.com/v1",
            )
        )

        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key,
                base_url="https://openai.vocareum.com/v1",
            )
        )

        metrics = [
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ]

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference
        )

        scores = {}
        for metric in metrics:
            try:
                score = metric.single_turn_score(sample)
                scores[metric.name] = float(score) if score is not None else None
            except Exception as e:
                scores[metric.name] = f"error: {str(e)}"

        return scores

    except Exception as e:
        return {"error": f"RAGAS evaluation failed: {str(e)}"}



def compute_aggregate_averages(results: List[Dict]) -> Dict[str, float]:
    """Compute average numeric metric scores across all evaluated questions."""
    metric_buckets: Dict[str, List[float]] = {}

    for result in results:
        scores = result.get("scores", {})
        for metric_name, metric_value in scores.items():
            if isinstance(metric_value, (int, float)):
                metric_buckets.setdefault(metric_name, []).append(float(metric_value))

    averages = {}
    for metric_name, values in metric_buckets.items():
        if values:
            averages[metric_name] = sum(values) / len(values)

    return averages


def save_results_to_json(
    results: List[Dict],
    aggregate_averages: Dict[str, float],
    output_file: str = "evaluation_results.json"
) -> None:
    """Save detailed evaluation results and aggregate averages to a JSON file."""
    payload = {
        "aggregate_averages": aggregate_averages,
        "results": results
    }

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def run_batch_evaluation(save_json: bool = True):
    """Run end-to-end batch evaluation using retrieval + generation + RAGAS scoring."""
    dataset = load_test_questions()

    if not dataset:
        print("No test questions found in test_questions.json")
        return

    backends = discover_chroma_backends()
    if not backends:
        print("No Chroma backends found.")
        return

    first_backend = next(iter(backends.values()))
    collection_name = first_backend.get("collection_name", "")
    chroma_dir = first_backend.get("chroma_dir", "")

    if not collection_name or not chroma_dir:
        print("Could not initialize Chroma collection from discovered backends.")
        return

    collection = initialize_rag_system(chroma_dir, collection_name)

    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CHROMA_OPENAI_API_KEY")
    if not openai_key:
        print("Missing OPENAI_API_KEY or CHROMA_OPENAI_API_KEY")
        return

    all_results = []

    print(f"Loaded {len(dataset)} test questions")
    print(f"Using backend: {chroma_dir} / {collection_name}")

    for index, item in enumerate(dataset, start=1):
        question = item.get("question", "").strip()
        reference = item.get("reference", "")
        category = item.get("category", "unknown")

        if not question:
            print(f"\nSkipping item {index}: missing question")
            continue

        print(f"\n--- Evaluating Question {index}/{len(dataset)} ---")
        print(f"Category: {category}")
        print(f"Question: {question}")

        try:
            retrieval_results = retrieve_documents(collection, question, n_results=3)

            documents = retrieval_results.get("documents", [[]])[0] if retrieval_results else []
            metadatas = retrieval_results.get("metadatas", [[]])[0] if retrieval_results else []

            contexts = documents if documents else []
            formatted_context = format_context(documents, metadatas)

            answer = generate_response(
                openai_key=openai_key,
                user_message=question,
                context=formatted_context,
                conversation_history=[],
            )

            scores = evaluate_response_quality(
                question=question,
                answer=answer,
                contexts=contexts,
                reference=reference
            )

            result = {
                "question": question,
                "category": category,
                "reference": reference,
                "retrieved_contexts": contexts,
                "answer": answer,
                "scores": scores,
            }
            all_results.append(result)

            print("Answer:")
            print(answer)
            print("Scores:")
            print(scores)

        except Exception as e:
            error_result = {
                "question": question,
                "category": category,
                "reference": reference,
                "retrieved_contexts": [],
                "answer": "",
                "scores": {"error": str(e)},
            }
            all_results.append(error_result)

            print(f"Error evaluating question: {e}")

    aggregate_averages = compute_aggregate_averages(all_results)

    print("\n=== AGGREGATE AVERAGES ===")
    if aggregate_averages:
        for metric_name, avg_score in aggregate_averages.items():
            print(f"{metric_name}: {avg_score:.4f}")
    else:
        print("No numeric aggregate scores available.")

    if save_json:
        save_results_to_json(all_results, aggregate_averages)
        print("\nSaved evaluation results to evaluation_results.json")


if __name__ == "__main__":
    run_batch_evaluation(save_json=True)
