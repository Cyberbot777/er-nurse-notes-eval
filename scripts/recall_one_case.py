import json
from pathlib import Path

from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from judges.recall import make_recall_metric

# Load the cases file and pick the case to judge. Change the id to try another case.
data = json.loads((Path(__file__).parent.parent / "data" / "cases.json").read_text(encoding="utf-8"))
case = next(c for c in data["cases"] if c["id"] == "bad")

# The judge: local Qwen through Ollama.
judge = OllamaModel(
    model="qwen3.8:27b",
    base_url="http://localhost:11434",
    temperature=0,
    generation_kwargs={"num_ctx": 16384},
)

# Recall: the golden note is the answer key (expected_output), and the case output is what gets checked.
test_case = LLMTestCase(
    input=data["raw_note"],
    actual_output=case["actual_output"],
    expected_output=data["golden_note"],
)

# The judge is always passed explicitly. verbose=True prints the full log.
metric = make_recall_metric(judge, verbose=True)

# Run the metric, then print the results. In strict mode the score is 1 or 0.
metric.measure(test_case)
print("Score:", metric.score)
print("Passed:", metric.success)
print("Reason:", metric.reason)
