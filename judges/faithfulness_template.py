from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessTemplate

# The ER charting rule, added to both the truths and verdicts prompts.
ER_CHARTING_CONVENTION = (
    "ER charting convention: In a nursing note, a timestamp opens a block of care. "
    "Every event charted after a timestamp, up to the next timestamp, belongs to that block. "
    "It happened at or shortly after that time and before the next timestamp. "
    "A claim that places an event at the time of the block it was charted in is supported."
)

# One extra instruction for each step.
TRUTHS_LINE = "When extracting truths, keep every event tied to the time of the block it was charted in."
VERDICTS_LINE = "Apply the charting convention when judging times."


# DeepEval's template with our rule added to the truths and verdicts prompts.
# The claims and reason prompts are inherited unchanged.
class ERFaithfulnessTemplate(FaithfulnessTemplate):

    @staticmethod
    def generate_truths(**kwargs):
        original = FaithfulnessTemplate.generate_truths(**kwargs)
        # Our rule goes above DeepEval's original prompt, which stays untouched.
        return f"{ER_CHARTING_CONVENTION}\n{TRUTHS_LINE}\n\n{original}"

    @staticmethod
    def generate_verdicts(**kwargs):
        original = FaithfulnessTemplate.generate_verdicts(**kwargs)
        # Our rule goes above DeepEval's original prompt, which stays untouched.
        return f"{ER_CHARTING_CONVENTION}\n{VERDICTS_LINE}\n\n{original}"


# Build the faithfulness metric. The judge is always passed in, never a default.
def make_faithfulness_metric(model, verbose=False):
    return FaithfulnessMetric(
        model=model,
        # Strict mode: any score below 1 becomes 0, and only 1 passes.
        strict_mode=True,
        # Borderline claims count against the score, the same as a "no".
        penalize_ambiguous_claims=True,
        evaluation_template=ERFaithfulnessTemplate,
        verbose_mode=verbose,
    )
