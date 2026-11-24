from deepeval.metrics import GEval
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


# Build the recall metric. The judge is always passed in, never a default.
def make_recall_metric(model, verbose=False):
    return GEval(
        name="Recall",
        # Our own steps, so GEval does not generate its own.
        evaluation_steps=RECALL_STEPS,
        # The judge sees only the agent's note and the golden note.
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=model,
        # Strict mode: the judge answers 1 or 0, and only 1 passes.
        strict_mode=True,
        verbose_mode=verbose,
    )
