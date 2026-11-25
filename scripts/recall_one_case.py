import json
from pathlib import Path

from deepeval.test_case import LLMTestCase
from judges.model import make_judge
from judges.recall import make_recall_metric

# Load the cases file and pick the case to judge. Change the id to try another case.
data = json.loads((Path(__file__).parent.parent / "data" / "cases.json").read_text(encoding="utf-8"))
case = next(c for c in data["cases"] if c["id"] == "dropped")

# The judge: local Qwen through Ollama, defined once in judges/model.py.
judge = make_judge()

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
