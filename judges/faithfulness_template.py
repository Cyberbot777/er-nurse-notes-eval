# DeepEval's default faithfulness template class, built from the prompt files bundled with DeepEval.
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessTemplate

# The ER charting rule, written once so the truths and verdicts prompts always carry the same wording.
ER_CHARTING_CONVENTION = (
    "ER charting convention: In a nursing note, a timestamp opens a block of care. "
    "Every event charted after a timestamp, up to the next timestamp, belongs to that block. "
    "It happened at or shortly after that time and before the next timestamp. "
    "A claim that places an event at the time of the block it was charted in is supported."
)

# The extra instruction for the truths step.
TRUTHS_LINE = "When extracting truths, keep every event tied to the time of the block it was charted in."

# The extra instruction for the verdicts step.
VERDICTS_LINE = "Apply the charting convention when judging times."


# Subclass DeepEval's template, so the claims and reason prompts are inherited unchanged.
class ERFaithfulnessTemplate(FaithfulnessTemplate):

    # Replace the truths prompt. DeepEval calls this with the same values it would pass to its own version.
    @staticmethod
    def generate_truths(**kwargs):
        # Render DeepEval's original truths prompt, untouched.
        original = FaithfulnessTemplate.generate_truths(**kwargs)
        # Put the convention and the truths instruction above DeepEval's original prompt.
        return f"{ER_CHARTING_CONVENTION}\n{TRUTHS_LINE}\n\n{original}"

    # Replace the verdicts prompt the same way.
    @staticmethod
    def generate_verdicts(**kwargs):
        # Render DeepEval's original verdicts prompt, untouched.
        original = FaithfulnessTemplate.generate_verdicts(**kwargs)
        # Put the convention and the verdicts instruction above DeepEval's original prompt.
        return f"{ER_CHARTING_CONVENTION}\n{VERDICTS_LINE}\n\n{original}"
