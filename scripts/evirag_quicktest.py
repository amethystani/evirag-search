"""
EVIRAG Quick Sanity Check — 3 NLI pairs via Ollama (qwen3.6:35b-a3b)
Validates the model can do NLI before we launch the full 2-hour battery.
Exit code 0 = pass (≥2/3), 1 = fail.
"""
import json, re, time, sys, argparse
import urllib.request, urllib.error

MODEL = "qwen3.6:35b-a3b"
OLLAMA_URL = "http://localhost:11434/api/chat"

QUICK_NLI = [
    (
        "Homework produces significant academic benefits for students at all grade levels.",
        "No conclusive achievement gains attributable to homework can be found across the meta-analytic record.",
        "CONTRADICTS"
    ),
    (
        "Homework has a positive relationship with academic achievement in secondary school.",
        "Students in grades 7-12 benefit more from homework than younger students.",
        "SUPPORTS"
    ),
    (
        "Excessive homework assignments cause measurable wellbeing harm without commensurate academic gain.",
        "Homework significantly improves academic performance across subject areas.",
        "CONTRADICTS"
    ),
]


def ollama_chat(system_msg: str, user_msg: str, model: str = MODEL) -> tuple[str, float]:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
        "think": False,           # disable extended thinking — content is empty otherwise
        "format": "json",         # force JSON output
        "options": {"temperature": 0.1, "num_predict": 200},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    elapsed = time.time() - t0
    return body["message"]["content"].strip(), elapsed


def parse_json(text: str) -> dict:
    # Strip <think>...</think> blocks (Qwen3 chain-of-thought)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for pat in [r'\{[^{}]*\}', r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        for m in re.findall(pat, text, re.DOTALL):
            try:
                return json.loads(m)
            except Exception:
                pass
    lbl = re.search(r'"label"\s*:\s*"([A-Z]+)"', text)
    return {"label": lbl.group(1) if lbl else "PARSE_ERROR",
            "confidence": 0.5, "reason": text[:120]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    args = parser.parse_args()

    # allow model/URL override via args
    active_model = args.model
    active_url   = args.ollama_url

    print("=" * 60)
    print(f"EVIRAG Quick Sanity Check — {active_model}")
    print("=" * 60)

    # Warm-up ping
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            tags = json.loads(r.read())
        names = [m["name"] for m in tags.get("models", [])]
        print(f"[OK] Ollama running. Models: {names}")
    except Exception as e:
        print(f"[FAIL] Cannot reach Ollama: {e}")
        sys.exit(1)

    correct = 0
    for i, (ca, cb, gold) in enumerate(QUICK_NLI):
        prompt = (
            f'Classify the relationship between these two scientific claims.\n\n'
            f'Claim A: "{ca}"\n'
            f'Claim B: "{cb}"\n\n'
            f'Respond ONLY with valid JSON (no extra text, no markdown):\n'
            f'{{"label":"SUPPORTS|CONTRADICTS|NEUTRAL","confidence":0.0-1.0,"reason":"one sentence"}}'
        )
        try:
            raw, elapsed = ollama_chat(
                system_msg="You are a precise scientific NLI classifier. Output valid JSON only. No markdown fences.",
                user_msg=prompt,
            )
            parsed = parse_json(raw)
        except Exception as e:
            parsed = {"label": f"ERROR:{e}", "confidence": 0, "reason": ""}
            elapsed = 0.0

        pred = parsed.get("label", "PARSE_ERROR")
        match = (pred == gold)
        correct += int(match)
        mark = "✓" if match else "✗"
        print(f"\n[{i+1}] {mark} Gold={gold:<12} Pred={pred:<12} "
              f"Conf={parsed.get('confidence', 0):.2f}  ({elapsed:.1f}s)")
        print(f"    Reason: {parsed.get('reason','')[:90]}")

    pct = correct / len(QUICK_NLI)
    print(f"\n{'='*60}")
    print(f"RESULT: {correct}/{len(QUICK_NLI)} correct ({pct:.0%})")
    if pct >= 2/3:
        print("✓ SANITY CHECK PASSED — launch full battery with evirag_eval_qwen32b.py")
        sys.exit(0)
    else:
        print("✗ SANITY CHECK FAILED — inspect output before full run")
        sys.exit(1)


if __name__ == "__main__":
    main()
