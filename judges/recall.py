# DeepEval's GEval metric, which scores an output by following steps we write ourselves.
from deepeval.metrics import GEval
# The enum that names test case fields, used to choose which fields the judge sees.
from deepeval.test_case import SingleTurnParams

# The six steps the judge follows. GEval sends them word for word as a numbered list.
RECALL_STEPS = [
    "Read the golden note (expected output) and list every clinical fact in it: each symptom, finding, vital sign, time, medication with dose and route, procedure, lab, history item, allergy status, and patient response.",
    "For each fact, check whether the agent's note (actual output) contains it. Different wording or abbreviations count as a match when the clinical meaning is the same, so \"high blood pressure\" matches \"hypertension\" and \"NKDA\" matches \"no known drug allergies.\"",
    "A fact counts as present if it appears anywhere in the agent's note, even in a different SOAPIE section.",
    "Times are facts. Apply the ER charting convention: a timestamp opens a block of care, and every event charted under it belongs to that time until the next timestamp.",
    "Ignore anything in the agent's note that is not in the golden note. Extra content is checked by a different judge.",
    "If even one clinical fact from the golden note is missing, the note fails. List every missing fact in your reason.",
]


# Build the recall metric around the judge model passed in, so the judge is always explicit.
def make_recall_metric(model):
    # Return a GEval metric set up for recall.
    return GEval(
        # The name shown in results.
        name="Recall",
        # Our steps, used as written. GEval skips generating its own steps when these are given.
        evaluation_steps=RECALL_STEPS,
        # The only two fields the judge sees: the agent's note and the golden note.
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        # The judge model, always passed in, never a default.
        model=model,
        # Strict mode: the judge answers 1 or 0, and only 1 passes.
        strict_mode=True,
        # Print the steps, score and reason while it runs.
        verbose_mode=True,
    )
