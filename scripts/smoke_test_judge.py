# Import DeepEval's wrapper for talking to a local Ollama server.
from deepeval.models import OllamaModel

# Create the judge with the exact model name, the local server address, and temperature 0 so answers repeat.
judge = OllamaModel(model="qwen3.8:27b", base_url="http://localhost:11434", temperature=0)

# The one question we send to the judge.
prompt = "In one sentence, what does NKDA mean in a nursing note?"

# generate returns two values: the answer text and a cost number. We keep the text and ignore the cost.
answer, cost = judge.generate(prompt)

# Print the judge's answer so we can read it.
print(answer)
