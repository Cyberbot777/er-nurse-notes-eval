# Import the shared judge definition.
from judges.model import make_judge

# Create the judge: local Qwen through Ollama, defined once in judges/model.py.
judge = make_judge()

# The one question we send to the judge.
prompt = "In one sentence, what does NKDA mean in a nursing note?"

# generate returns two values: the answer text and a cost number. We keep the text and ignore the cost.
answer, cost = judge.generate(prompt)

# Print the judge's answer so we can read it.
print(answer)
