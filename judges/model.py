from deepeval.models import OllamaModel


# The one judge definition. Every metric and test uses this same model and settings.
def make_judge():
    return OllamaModel(
        model="qwen3.8:27b",
        base_url="http://localhost:11434",
        # Temperature 0 so the same input gives the same answer.
        temperature=0,
        # Context window size, passed through to Ollama.
        generation_kwargs={"num_ctx": 16384},
    )
