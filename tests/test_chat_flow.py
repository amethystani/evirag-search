import requests
import time
import json

base_url = "http://localhost:8000/api"

print("="*60)
print("Testing Perplexity-Style EVIRAG Chat Flow")
print("="*60)

# Turn 1
q1 = "Is the earth flat or round?"
print(f"\n[Turn 1] Query: {q1}")
start = time.time()
res1 = requests.post(f"{base_url}/chat", json={
    "message": q1,
    "backend": "local",
    "agents": False,
    "vlm": False
}).json()
print(f"Latency: {time.time()-start:.2f}s")
chat_data = res1.get("chat", {})
session_id = chat_data.get("session_id")
print(f"Session ID: {session_id}")
print(f"Turn: {chat_data.get('turn')}")
print(f"Evolving Claim: {chat_data.get('claim')}")
print(f"Answer snippet: {chat_data.get('answer', '')[:150]}...")
print(f"History length: {len(chat_data.get('history', []))}")

# Turn 2
q2 = "But what about the flat earth society?"
print(f"\n[Turn 2] Query: {q2}")
start = time.time()
res2 = requests.post(f"{base_url}/chat", json={
    "message": q2,
    "session_id": session_id,
    "backend": "local",
    "agents": False,
    "vlm": False
}).json()
print(f"Latency: {time.time()-start:.2f}s")
chat_data2 = res2.get("chat", {})
print(f"Turn: {chat_data2.get('turn')}")
print(f"Evolving Claim: {chat_data2.get('claim')}")
print(f"Answer snippet: {chat_data2.get('answer', '')[:150]}...")
print(f"History length: {len(chat_data2.get('history', []))}")

# Turn 3
q3 = "So, what's the scientific consensus on its shape?"
print(f"\n[Turn 3] Query: {q3}")
start = time.time()
res3 = requests.post(f"{base_url}/chat", json={
    "message": q3,
    "session_id": session_id,
    "backend": "local",
    "agents": False,
    "vlm": False
}).json()
print(f"Latency: {time.time()-start:.2f}s")
chat_data3 = res3.get("chat", {})
print(f"Turn: {chat_data3.get('turn')}")
print(f"Evolving Claim: {chat_data3.get('claim')}")
print(f"Answer snippet: {chat_data3.get('answer', '')[:150]}...")
print(f"History length: {len(chat_data3.get('history', []))}")
print("\nSuccess.")
