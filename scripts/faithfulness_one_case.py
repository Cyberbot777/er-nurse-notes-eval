# Standard library module for reading JSON files.
import json
# Standard library helper for building file paths.
from pathlib import Path

# DeepEval's wrapper for a local Ollama model.
from deepeval.models import OllamaModel
# DeepEval's container for one thing to evaluate.
from deepeval.test_case import LLMTestCase
# DeepEval's faithfulness metric.
from deepeval.metrics import FaithfulnessMetric
# Our faithfulness template with the ER charting convention added.
from judges.faithfulness_template import ERFaithfulnessTemplate

# Read data/cases.json, which sits one folder above this script, so it works from any directory.
data = json.loads((Path(__file__).parent.parent / "data" / "cases.json").read_text(encoding="utf-8"))

# Pick the one case whose id is "bad".
case = next(c for c in data["cases"] if c["id"] == "dropped")

# Create the judge with the model name, the Ollama address, temperature 0, and a 16384 token context window passed to Ollama.
judge = OllamaModel(
    model="qwen3.8:27b",
    base_url="http://localhost:11434",
    temperature=0,
    generation_kwargs={"num_ctx": 16384},
)

# Build the test case: the raw note is the input and the only source of truth, and the bad note is the output being checked.
test_case = LLMTestCase(
    input=data["raw_note"],
    actual_output=case["actual_output"],
    retrieval_context=[data["raw_note"]],
)

# Create the metric with our judge passed explicitly, verbose logs on, strict mode on, borderline claims counted against the score, and our ER template.
metric = FaithfulnessMetric(model=judge, verbose_mode=True, strict_mode=True, penalize_ambiguous_claims=True, evaluation_template=ERFaithfulnessTemplate)

# Run the metric. This calls the judge several times and stores the results on the metric object.
metric.measure(test_case)

# Print the score, a number from 0 to 1.
print("Score:", metric.score)

# Print whether the score met the pass threshold.
print("Passed:", metric.success)

# Print the judge's explanation for the score.
print("Reason:", metric.reason)

# Print how many claims and verdicts there are, so we can see if they line up one to one.
print("Claims:", len(metric.claims), "Verdicts:", len(metric.verdicts))

# Walk through each claim next to its verdict.
for claim, verdict in zip(metric.claims, metric.verdicts):
    # Print the verdict (yes, no or borderline) followed by the claim text.
    print(f"{verdict.verdict}: {claim}")
