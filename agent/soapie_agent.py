import httpx
from ollama import Client, ResponseError

# How long to wait for the model before giving up on a try.
TIMEOUT_SECONDS = 120

# The instructions the agent model always receives.
SYSTEM_PROMPT = """You are an ER nursing documentation assistant. Convert the nurse's raw shorthand note into a SOAPIE note.

Rules:
1. Use only information written in the raw note. Never add findings, medications, doses, times or events that are not there.
2. Include every fact from the raw note. Do not drop anything, including negatives such as allergy status or denied symptoms.
3. Spell out abbreviations in plain clinical language.
4. Keep the times. A timestamp opens a block of care: every event charted after it belongs to that time until the next timestamp.
5. Use these six sections in order, each with its heading:
   S (Subjective): what the patient reports, history, allergies, home medications.
   O (Objective): observed findings, vital signs, EKG and monitor results.
   A (Assessment): the nursing assessment, based only on charted findings.
   P (Plan): the plan of care.
   I (Intervention): what was done, with times.
   E (Evaluation): how the patient responded.
6. Write only the SOAPIE note, with no introduction or commentary."""


# Turn one raw shorthand note into a SOAPIE note.
def write_soapie(raw_note):
    # The client talks to the local Ollama server. The host is set here, not left to a default.
    # The timeout is in seconds. Without it the client would wait forever.
    client = Client(host="http://localhost:11434", timeout=TIMEOUT_SECONDS)

    # One try plus one retry.
    for attempt in range(2):
        try:
            # The system prompt is the instructions. The raw note is the user message.
            response = client.chat(
                model="ministral-3:8b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": raw_note},
                ],
                # Temperature 0 for repeatable output. num_ctx is the context window size.
                options={"temperature": 0, "num_ctx": 8192},
            )
            # Return the note text only.
            return response.message.content
        except (httpx.TimeoutException, ConnectionError, ResponseError):
            # A second failure is raised to the caller.
            if attempt == 1:
                raise
