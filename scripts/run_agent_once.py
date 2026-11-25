import json
from pathlib import Path

from agent.soapie_agent import write_soapie

# Load the raw note from the cases file.
data = json.loads((Path(__file__).parent.parent / "data" / "cases.json").read_text(encoding="utf-8"))

# Run the agent on the raw note and print the SOAPIE note.
print(write_soapie(data["raw_note"]))
