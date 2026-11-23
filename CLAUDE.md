# Project context

## What this is
A portfolio project to learn DeepEval by building LLM judges for an ER nurse notes agent. The agent turns a raw shorthand ER note into a structured SOAPIE note, and the judges score that output.

## How we work
Richard is the domain expert and makes every decision. Go one step at a time and do only what the prompt asks. Do not create, edit or delete files unless the prompt asks for it. Do not commit or push unless asked. Ask before installing any package.

## Environment
Windows with PowerShell. Python 3.13.18 through uv: use uv run and uv add, never a system pip install. Ollama runs on localhost:11434. The judge model is Qwen 27B in Ollama. The app agent model will be a smaller model from a different family and is not chosen yet.

## Data
data/cases.json holds the raw note, the golden SOAPIE note and four judge test cases (good, bad, neutral, dropped) with expected verdicts. These are nurse reviewed answer keys. Never change them without Richard's approval. All data is synthetic; never add real patient information.

## Judges
Faithfulness compares the output against the raw note: every fact in the output must trace to the raw note. Recall compares the output against the golden note: every golden fact must appear in the output. Paraphrase counts as a match, so "high blood pressure" matches "hypertension."

## Build order
1. Faithfulness and recall judges, tested on the four cases until all eight verdicts match.
2. The SOAPIE agent.
3. Allure reports.
4. React dashboard.
Later: a charting quality judge, Langfuse, ambient input, and focused assessment by chief complaint.

## Writing style
Never use hyphens or dashes of any kind in docs or comments, including em dashes.
