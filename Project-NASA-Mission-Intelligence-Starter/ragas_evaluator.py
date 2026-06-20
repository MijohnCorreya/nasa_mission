import os
from typing import Dict, List

from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from ragas.metrics import BleuScore, ResponseRelevancy, Faithfulness, RougeScore
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False




def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # Create evaluator LLM pointed at the Vocareum proxy
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("CHROMA_OPENAI_API_KEY"),
            base_url="https://openai.vocareum.com/v1",
        )
    )

    # Create evaluator embeddings pointed at the Vocareum proxy
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("CHROMA_OPENAI_API_KEY"),
            base_url="https://openai.vocareum.com/v1",
        )
    )

    # Define metrics — BleuScore and RougeScore are reference-free (no LLM needed)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        BleuScore(),
        RougeScore(),
    ]

    # Build a SingleTurnSample for RAGAS
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    # Score each metric individually so one failure doesn't block the others
    scores = {}
    for metric in metrics:
        try:
            scores[metric.name] = metric.single_turn_score(sample)
        except Exception as e:
            scores[metric.name] = f"error: {str(e)}"

    return scores