# ER Nurse Notes Eval

Evaluating an ER nurse notes agent with DeepEval and local LLM judges.

## What it does

ER nurses chart fast, in shorthand. This project takes a quick ER nursing note and turns it into a structured SOAPIE note (Subjective, Objective, Assessment, Plan, Intervention, Evaluation). Two LLM judges then check the result.

The faithfulness judge checks that every fact in the SOAPIE note traces back to the raw note, which catches invented facts like a medication that was never given. The recall judge checks that every fact in the nurse reviewed golden note appears in the output, which catches dropped facts like a missing allergy status.

## Judges come first

Before the agent writes a single note, the judges are tested on four hand written cases built from one raw note. Each case has a known right answer, so the judges are proven to score correctly before they ever grade the agent.

| Case | The one change | Faithfulness | Recall |
|---|---|---|---|
| Good | None, the golden note | Pass | Pass |
| Bad | One invented fact | Fail | Pass |
| Neutral | Same facts, different wording | Pass | Pass |
| Dropped | One fact removed | Pass | Fail |

## How it runs

Everything runs locally on an RTX 5090 through Ollama, with Qwen 27B as the judge. No note ever leaves the machine. The stack is Python 3.13 managed with uv, pytest and DeepEval, with Allure reports and a small React dashboard that shows each score next to the judge's reasoning.

## Status

In progress. The raw note, golden SOAPIE note and four judge test cases are in data/cases.json. The faithfulness and recall judges are next.

## Data

All notes are synthetic. No real patient information is used anywhere in this repo.
