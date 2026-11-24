import json
from pathlib import Path

from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from judges.faithfulness_template import ERFaithfulnessTemplate

# Load the cases file and pick the case to judge. Change the id to try another case.
data = json.loads((Path(__file__).parent.parent / "data" / "cases.json").read_text(encoding="utf-8"))
case = next(c for c in data["cases"] if c["id"] == "dropped")

# The judge: local Qwen through Ollama.
judge = OllamaModel(
    model="qwen3.8:27b",
    base_url="http://localhost:11434",
    temperature=0,
    generation_kwargs={"num_ctx": 16384},
)

# Faithfulness: the raw note is the only source of truth (retrieval_context), and the case output is what gets checked.
test_case = LLMTestCase(
    input=data["raw_note"],
    actual_output=case["actual_output"],
    retrieval_context=[data["raw_note"]],
)

# Strict mode and penalize_ambiguous_claims make any unsupported claim fail the note. The judge is always passed explicitly.
metric = FaithfulnessMetric(model=judge, verbose_mode=True, strict_mode=True, penalize_ambiguous_claims=True, evaluation_template=ERFaithfulnessTemplate)

# Run the metric, then print the results.
metric.measure(test_case)
print("Score:", metric.score)
print("Passed:", metric.success)
print("Reason:", metric.reason)

# Print the claim and verdict counts first, so a mismatch in the pairing is easy to spot.
print("Claims:", len(metric.claims), "Verdicts:", len(metric.verdicts))
for claim, verdict in zip(metric.claims, metric.verdicts):
    print(f"{verdict.verdict}: {claim}")
