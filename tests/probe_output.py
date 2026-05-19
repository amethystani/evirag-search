"""Probe Qwen3.6 output — test think:false and format:json"""
import urllib.request, json

BASE = "http://localhost:11434/api/chat"

def ask(messages, think=True, use_format_json=False, max_tokens=600, temp=0.6):
    payload_dict = {
        "model": "qwen3.6:35b-a3b",
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temp,
            "num_predict": max_tokens,
        },
    }
    if not think:
        payload_dict["think"] = False  # Ollama-level think param
        payload_dict["options"]["think"] = False
    if use_format_json:
        payload_dict["format"] = "json"

    payload = json.dumps(payload_dict).encode()
    req = urllib.request.Request(BASE, data=payload,
                                  headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read())
    msg = body["message"]
    return msg.get("content",""), msg.get("thinking","")

NLI_MSG = [
    {"role": "system", "content": "You are a scientific NLI classifier."},
    {"role": "user", "content": (
        "Classify the relationship between two scientific claims.\n\n"
        "Claim A: 'Homework improves achievement for high school students.'\n"
        "Claim B: 'No conclusive homework benefit found in meta-analyses.'\n\n"
        "Answer: SUPPORTS, CONTRADICTS, or NEUTRAL. "
        "Give JSON: {\"label\": \"...\", \"confidence\": 0.9, \"reason\": \"...\"}"
    )}
]

print("=== Test A: think=True (default) ===")
content, thinking = ask(NLI_MSG, think=True)
print(f"content: {repr(content[:300])}")
print(f"thinking: {repr(thinking[:200]) if thinking else '(none)'}")

print("\n=== Test B: think=False ===")
content, thinking = ask(NLI_MSG, think=False)
print(f"content: {repr(content[:300])}")
print(f"thinking: {repr(thinking[:200]) if thinking else '(none)'}")

print("\n=== Test C: think=False + format=json ===")
content, thinking = ask(NLI_MSG, think=False, use_format_json=True)
print(f"content: {repr(content[:400])}")
print(f"thinking: {repr(thinking[:200]) if thinking else '(none)'}")

print("\n=== Test D: simple free-text no JSON ===")
content, thinking = ask([
    {"role":"user","content":"Are these contradictory? A:'Homework helps.' B:'Homework has no benefit.' Answer yes or no."}
], think=False)
print(f"content: {repr(content[:200])}")
