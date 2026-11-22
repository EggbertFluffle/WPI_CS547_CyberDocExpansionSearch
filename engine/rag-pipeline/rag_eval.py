"""
DeepEval scoring for the RAG pipeline (Answer Relevancy, Faithfulness, Contextual Relevancy).

The judge is the same HuggingFace LLM the RAG pipeline uses (rag_npy.MODEL_NAME).

Use from another file (e.g. the API):
    import rag_eval
    scores = rag_eval.evaluate_response(query, explanation, original_document_text)
    results = rag_eval.evaluate_results(query, results)   # results = rag_npy.search_exploits(...)

Use from the command line:
    python rag_eval.py "how can someone get a shell by overflowing the mail program"

Install:
    pip install deepeval
"""

import json
import os
import re
import sys

# The judge is a local 8B model and makes many sequential calls; DeepEval's default
# per-attempt timeout is too short. Must be set before deepeval is imported.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "1800")

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

import rag_npy

JUDGE_MAX_NEW_TOKENS = 8192  # judge answers are long JSON verdicts plus the model's <think> block


class HFJudge(DeepEvalBaseLLM):
    """DeepEval judge model backed by the same HuggingFace LLM as the RAG pipeline."""

    def __init__(self):
        self.chat_model = rag_npy.build_llm(JUDGE_MAX_NEW_TOKENS)

    def load_model(self):
        return self.chat_model

    def _complete(self, prompt):
        return rag_npy.strip_think(self.chat_model.invoke(prompt).content)

    def generate(self, prompt, schema=None):
        if schema is None:
            return self._complete(prompt)
        # DeepEval passes a pydantic schema when it needs structured output
        prompt += (
            "\n\nReturn ONLY a valid JSON object (no markdown, no prose) that is an INSTANCE of this "
            "JSON schema - fill in the actual values, do not repeat the schema itself "
            "(no 'properties', 'type' or 'title' keys):\n"
            + json.dumps(schema.model_json_schema())
        )
        raw = self._complete(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0) if match else raw)
        # the model sometimes mirrors the schema shape: {"properties": {...actual values...}}
        if isinstance(data, dict) and isinstance(data.get("properties"), dict) \
                and not set(data) & set(schema.model_fields):
            data = data["properties"]
        return schema.model_validate(data)

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return f"HuggingFace ({rag_npy.MODEL_NAME})"


_judge = None


def get_judge():
    """Load the judge model once and reuse it."""
    global _judge
    if _judge is None:
        _judge = HFJudge()
    return _judge


def evaluate_response(input_query, llm_output, retrieved_context):
    """Score one answer with DeepEval.

    input_query:       the user's query
    llm_output:        the LLM explanation being scored
    retrieved_context: the text the LLM was given (a string, or a list of strings)

    Returns {metric name: {"score", "reason", "passed"}}.
    """
    judge = get_judge()
    test_case = LLMTestCase(
        input=input_query,
        actual_output=llm_output,
        retrieval_context=retrieved_context if isinstance(retrieved_context, list) else [retrieved_context],
    )

    scores = {}
    for metric in (
        AnswerRelevancyMetric(model=judge, async_mode=False),
        FaithfulnessMetric(model=judge, async_mode=False),
        ContextualRelevancyMetric(model=judge, async_mode=False),
    ):
        # measure() on its own prints nothing (evaluate() prints a test-run summary)
        metric.measure(test_case, _show_indicator=False)
        scores[metric.__name__] = {
            "score": metric.score,
            "reason": metric.reason,
            "passed": metric.is_successful(),
        }
    return scores


def evaluate_results(query, results):
    """Add DeepEval scores to every result from rag_npy.search_exploits().
    Each result's explanation is scored against the original document the LLM was given."""
    return [
        {**r, "deepeval": evaluate_response(query, r["explanation"], r["content"])}
        for r in results
    ]


def main():
    query = " ".join(sys.argv[1:]) or (
        "How can an attacker execute commands remotely through a vulnerable web application?"
    )

    pipeline = rag_npy.load_pipeline()
    results = rag_npy.search_exploits(query, *pipeline)
    results = evaluate_results(query, results)

    print(f"\nQuery: {query}")
    for i, r in enumerate(results, start=1):
        print(f"\n{'=' * 60}")
        print(f"Result {i}")
        print(f"Exploit ID: {r['exploit_id']}")
        print(f"File: {r['path']}")
        print("\nAI-Generated Explanation:")
        print(r["explanation"])
        print("\nDeepEval:")
        for name, m in r["deepeval"].items():
            status = "pass" if m["passed"] else "fail"
            print(f"  {name}: {m['score']} ({status})")
            print(f"    {m['reason']}")


if __name__ == "__main__":
    main()