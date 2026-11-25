import argparse
import json
from datetime import datetime
from pathlib import Path

from deepeval.test_case import LLMTestCase

from agent.soapie_agent import write_soapie
from judges.faithfulness_template import make_faithfulness_metric
from judges.model import make_judge
from judges.recall import make_recall_metric

# The one optional argument. With --from-file the agent is skipped and the note is read from that file.
parser = argparse.ArgumentParser()
parser.add_argument("--from-file", metavar="PATH", help="Judge a saved SOAPIE note instead of calling the agent.")
args = parser.parse_args()

# Load the raw note and the golden note from the cases file.
root = Path(__file__).parent.parent
data = json.loads((root / "data" / "cases.json").read_text(encoding="utf-8"))
raw_note = data["raw_note"]
golden_note = data["golden_note"]

# Get the SOAPIE note: from the file if given, otherwise from the agent.
if args.from_file:
    note = Path(args.from_file).read_text(encoding="utf-8")
else:
    note = write_soapie(raw_note)

    # Save the agent's note with a timestamp in the name, creating outputs/ if needed.
    outputs = root / "outputs"
    outputs.mkdir(exist_ok=True)
    saved = outputs / f"agent_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    saved.write_text(note, encoding="utf-8")
    print("Saved:", saved)

# Show the note that is being judged.
print("\n=== SOAPIE note being judged ===")
print(note)

# The one judge model, shared by both metrics.
judge = make_judge()

# Faithfulness: the raw note is the only source of truth.
faithfulness = make_faithfulness_metric(judge, verbose=True)
faithfulness.measure(
    LLMTestCase(
        input=raw_note,
        actual_output=note,
        retrieval_context=[raw_note],
    )
)

# Recall: the golden note is the answer key.
recall = make_recall_metric(judge, verbose=True)
recall.measure(
    LLMTestCase(
        input=raw_note,
        actual_output=note,
        expected_output=golden_note,
    )
)

# Print the summary: score, pass or fail, and reason for each judge.
print("\n=== Summary ===")
for name, metric in (("Faithfulness", faithfulness), ("Recall", recall)):
    print(f"{name}: score {metric.score}, {'PASS' if metric.success else 'FAIL'}")
    print(f"  Reason: {metric.reason}")
