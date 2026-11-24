import json
from pathlib import Path

import pytest
from deepeval.test_case import LLMTestCase

from judges.model import make_judge
from judges.faithfulness_template import make_faithfulness_metric
from judges.recall import make_recall_metric

# The answer key.
DATA = json.loads((Path(__file__).parent.parent / "data" / "cases.json").read_text(encoding="utf-8"))

# One test per case and judge, 8 in all, with ids like "bad-faithfulness".
PARAMS = [
    pytest.param(case, judge_name, id=f"{case['id']}-{judge_name}")
    for case in DATA["cases"]
    for judge_name in ("faithfulness", "recall")
]


# One judge model, shared by every test in this file.
@pytest.fixture(scope="module")
def judge():
    return make_judge()


# Check that each judge's verdict matches the answer key.
@pytest.mark.parametrize("case, judge_name", PARAMS)
def test_judge_matches_answer_key(judge, case, judge_name):
    # Faithfulness checks against the raw note. Recall checks against the golden note.
    if judge_name == "faithfulness":
        metric = make_faithfulness_metric(judge)
        test_case = LLMTestCase(
            input=DATA["raw_note"],
            actual_output=case["actual_output"],
            retrieval_context=[DATA["raw_note"]],
        )
    else:
        metric = make_recall_metric(judge)
        test_case = LLMTestCase(
            input=DATA["raw_note"],
            actual_output=case["actual_output"],
            expected_output=DATA["golden_note"],
        )

    metric.measure(test_case)

    # The answer key says "pass" or "fail". metric.success should be True or False to match.
    expected_verdict = case["expected"][judge_name]
    expected_success = expected_verdict == "pass"

    # The failure message: case, judge, expected verdict, score and reason.
    message = (
        f"Case: {case['id']}\n"
        f"Judge: {judge_name}\n"
        f"Expected verdict: {expected_verdict}\n"
        f"Score: {metric.score}\n"
        f"Reason: {metric.reason}"
    )

    # Faithfulness failures also list every claim with its verdict and reason.
    if judge_name == "faithfulness":
        message += f"\nClaims: {len(metric.claims)}, Verdicts: {len(metric.verdicts)}"
        for claim, verdict in zip(metric.claims, metric.verdicts):
            message += f"\n  {verdict.verdict}: {claim} | reason: {verdict.reason}"

    assert metric.success == expected_success, message
