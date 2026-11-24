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

## Why a nurse built the judges

LLM judges are LLMs, and they can be confidently wrong about the very thing they grade. The first run proved it. DeepEval's built in faithfulness metric correctly failed the note with an invented morphine dose, but it also failed the golden note itself, scoring it 0 with a confident explanation that the note misattributed timing.

The judge was wrong. ER nurses chart in timed blocks: a timestamp opens a block of care, and everything charted under it happened at or after that time and before the next one. The judge didn't know that convention. Its verbose output showed two causes. The truths extraction step dropped the time from events in the 1436 block, and the verdict step then treated block timing as unsupported.

The easy fix would have been to loosen the threshold until the golden note passed, but that would have let the invented medication through too. Instead, the judge was taught the charting convention through DeepEval's evaluation_template override, with the metric and its strict settings unchanged. After the fix, both judges matched the nurse's answer key on all eight verdicts. The golden note passes, the invented morphine still fails faithfulness, the reworded note passes both judges, and the note missing its allergy status fails recall.

## How it runs

Everything runs locally on an RTX 5090 through Ollama, with Qwen 27B as the judge. No note ever leaves the machine. The stack is Python 3.13 managed with uv, pytest and DeepEval, with Allure reports and a small React dashboard that shows each score next to the judge's reasoning.

## Status

In progress. The raw note, golden SOAPIE note and four judge test cases are in data/cases.json. Faithfulness and recall judges are built and match the nurse's answer key on all eight verdicts. The SOAPIE agent is next.

## Data

All notes are synthetic. No real patient information is used anywhere in this repo.

## About

Built by Richard Hall, a former ER nurse and nurse manager who now works as an AI engineer. Every golden note and expected verdict is written or reviewed by a nurse.
