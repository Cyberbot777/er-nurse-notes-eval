# Standard library module for reading JSON files.
import json
# Standard library helper for building file paths.
from pathlib import Path

# DeepEval's wrapper for a local Ollama model.
from deepeval.models import OllamaModel
# DeepEval's container for one thing to evaluate.
from deepeval.test_case import LLMTestCase
# Our function that builds the recall metric.
from judges.recall import make_recall_metric

# Read data/cases.json, which sits one folder above this script, so it works from any directory.
data = json.loads((Path(__file__).parent.parent / "data" / "cases.json").read_text(encoding="utf-8"))

# Pick the one case whose id is "dropped".
case = next(c for c in data["cases"] if c["id"] == "bad")

# Create the judge with the model name, the Ollama address, temperature 0, and a 16384 token context window passed to Ollama.
judge = OllamaModel(
    model="qwen3.8:27b",
    base_url="http://localhost:11434",
    temperature=0,
    generation_kwargs={"num_ctx": 16384},
)

# Build the test case: the raw note as input, the dropped note as the output being checked, and the golden note as the answer key.
test_case = LLMTestCase(
    input=data["raw_note"],
    actual_output=case["actual_output"],
    expected_output=data["golden_note"],
)

# Create the recall metric with our judge passed explicitly.
metric = make_recall_metric(judge)

# Run the metric. This calls the judge and stores the results on the metric object.
metric.measure(test_case)

# Print the score. In strict mode it is 1 or 0.
print("Score:", metric.score)

# Print whether the score met the pass threshold.
print("Passed:", metric.success)

# Print the judge's explanation for the score.
print("Reason:", metric.reason)
