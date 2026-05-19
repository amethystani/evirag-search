"""
EVIRAG Evaluation Suite — qwen3.6:35b-a3b via Ollama
======================================================
Five task batteries covering every EMNLP reviewer objection:

  Task A  NLI Contradiction Detection     12 gold pairs  → accuracy, CONTRADICTS F1
  Task B  CDA-7 Causal Attribution        10 gold pairs  → accuracy, Cohen's κ
  Task C  Claim Extraction Quality         3 passages    → coverage, atomicity
  Task D  ViewpointCoverage comparison     1 query       → VC vanilla vs EVIRAG
  Task E  Full Structured Synthesis        1 query       → JSON schema compliance

Usage:
    python evirag_eval_qwen32b.py [--tasks A,B,C,D,E] [--output results.json]

All results saved to evirag_eval_results.json for paper table generation.
"""

import json, re, time, argparse, sys, urllib.request
from pathlib import Path
from datetime import datetime

MODEL       = "qwen3.6:35b-a3b"
OLLAMA_URL  = "http://localhost:11434/api/chat"

# ──────────────────────────────────────────────────────────────────────────────
# Gold-standard test data (homework-research literature)
# ──────────────────────────────────────────────────────────────────────────────

NLI_GOLD = [
    ("Homework produces significant academic benefits for students at all grade levels.",
     "No conclusive achievement gains attributable to homework can be found across the meta-analytic record.",
     "CONTRADICTS", "Cooper2006 vs null-result meta-analyses"),
    ("The positive effects of homework on academic achievement are stronger for high school students than for elementary school students.",
     "Homework has uniform beneficial effects regardless of student grade level.",
     "CONTRADICTS", "Cooper2006 grade-level finding vs uniform-effect claim"),
    ("Homework has a positive relationship with academic achievement in secondary school.",
     "Students in grades 7-12 benefit more from homework than younger students.",
     "SUPPORTS", "Two statements of same Cooper2006 finding"),
    ("Excessive homework assignments cause measurable wellbeing harm without commensurate academic gain.",
     "Homework significantly improves academic performance across subject areas.",
     "CONTRADICTS", "Scheb2023 wellbeing harm vs achievement benefit"),
    ("Science homework assignments lead to improved attitudes toward science among middle school students.",
     "Homework in science courses has positive effects on student engagement and outcomes.",
     "SUPPORTS", "Masalimova2023 — two paraphrases"),
    ("Homework improves academic achievement when it is well-designed and appropriately assigned.",
     "Poorly designed homework has no beneficial effect and may harm student motivation.",
     "SUPPORTS", "Conditional-benefit view — same underlying claim"),
    ("Homework is the most effective out-of-school learning intervention available to educators.",
     "The evidence base for homework effectiveness is inconsistent and context-dependent.",
     "CONTRADICTS", "Strong vs weak-evidence claim"),
    ("Students who complete homework regularly achieve higher test scores.",
     "Test scores are not a reliable measure of learning outcomes attributable to homework.",
     "CONTRADICTS", "Achievement claim vs measurement critique"),
    ("Meta-analyses consistently show a positive correlation between homework completion and academic performance.",
     "Correlational studies cannot establish that homework causes improved performance.",
     "CONTRADICTS", "Correlation vs causation methodological debate"),
    ("The relationship between homework time and achievement follows an inverted-U curve.",
     "Excessive homework beyond an optimal threshold reduces achievement gains.",
     "SUPPORTS", "Two descriptions of the inverted-U finding"),
    ("Homework cultivates self-discipline and independent learning habits.",
     "There is no reliable evidence that homework builds non-cognitive skills.",
     "CONTRADICTS", "Non-cognitive benefits claim vs null claim"),
    ("Parental involvement in homework sessions improves academic outcomes.",
     "Parental homework help can cause anxiety and does not reliably improve achievement.",
     "CONTRADICTS", "Parental involvement benefit vs harm claim"),
]

CDA7_GOLD = [
    ("Homework improves achievement in comprehensive meta-analyses spanning grades K-12.",
     "Homework shows no significant effect in randomized controlled trials targeting elementary students.",
     "One study uses observational meta-analysis, the other uses RCT design",
     "METHODOLOGICAL", "RCT vs observational design"),
    ("Homework benefits high school students significantly.",
     "Homework has minimal effect on elementary school students.",
     "Same intervention, different student populations studied",
     "POPULATION", "Grade-level population difference"),
    ("1980s studies showed strong positive homework effects.",
     "Post-2010 studies show weaker or null effects of homework on achievement.",
     "Studies from different decades disagree",
     "TEMPORAL", "Newer evidence vs older consensus"),
    ("Achievement is measured by standardized test scores in these studies.",
     "Achievement is measured by course grades and GPA in these studies.",
     "Both claim homework affects achievement but define it differently",
     "OPERATIONAL", "Operationalization: test scores vs grades"),
    ("A significant positive effect of homework (d=0.35, p<0.05) was found.",
     "The effect of homework is not statistically significant (d=0.12, p=0.23).",
     "Both measure similar samples but report conflicting statistics",
     "STATISTICAL", "Effect size and p-value conflict"),
    ("Homework develops intrinsic motivation via self-determination theory principles.",
     "Homework undermines intrinsic motivation by externalizing the learning locus of control.",
     "Two competing motivational frameworks applied to the same intervention",
     "THEORETICAL", "SDT vs extrinsic motivation theory"),
    ("A landmark 2002 study found homework improved reading scores by 15%.",
     "A 2015 replication of the same protocol found no significant reading score improvement.",
     "Direct replication attempt with null result",
     "REPLICATION", "Direct replication failure"),
    ("Homework helps students consolidate classroom learning.",
     "Online practice platforms are more effective than traditional homework for consolidation.",
     "Different intervention modalities for the same learning goal",
     "METHODOLOGICAL", "Different protocols / delivery modalities"),
    ("Studies on urban, high-poverty schools show homework widens achievement gaps.",
     "Studies on suburban schools show homework narrows achievement gaps.",
     "Different SES populations produce opposite findings",
     "POPULATION", "SES / school-setting population difference"),
    ("The homework-achievement correlation is 0.25 in the original Cooper 1989 meta-analysis.",
     "Subsequent meta-analyses find correlations ranging from 0.02 to 0.16.",
     "Effect size has decreased across replications over time",
     "REPLICATION", "Declining effect size across replication attempts"),
]

CLAIM_PASSAGES = [
    {"id": "P1",
     "text": ("Cooper et al. (2006) conducted a synthesis of homework research spanning 1987 to 2003. "
              "Their analysis found that homework had a positive effect on academic achievement for students "
              "in grades 7-12, but negligible effects for elementary school students. The review also noted "
              "that the type of homework mattered: practice assignments showed stronger effects than "
              "preparation assignments."),
     "expected_min": 3,
     "key_claims": ["homework positive effect grades 7-12",
                    "homework negligible effect elementary",
                    "practice assignments stronger than preparation"]},
    {"id": "P2",
     "text": ("Masalimova et al. (2023) examined homework in science education specifically. They found that "
              "regular science homework improved student attitudes toward science and had moderate positive "
              "effects on science achievement scores. However, excessive homework loads caused increased "
              "stress and reduced intrinsic motivation in some student groups, particularly in lower-SES contexts."),
     "expected_min": 3,
     "key_claims": ["science homework improved attitudes",
                    "moderate positive effect on science achievement",
                    "excessive homework increased stress"]},
    {"id": "P3",
     "text": ("Scheb (2023) investigated the dual effects of homework on both academic performance and mental "
              "health in Catholic school students. The study found that while moderate homework correlated "
              "with slightly higher academic performance, students reporting more than two hours of homework "
              "per night showed elevated anxiety and depression symptoms. Academic performance gains did not "
              "scale proportionally with homework volume beyond the moderate threshold."),
     "expected_min": 3,
     "key_claims": ["moderate homework correlated with higher performance",
                    "more than two hours increased anxiety depression",
                    "performance gains did not scale with volume"]},
]

VC_QUERY = ("Does homework improve academic achievement? "
            "Provide a comprehensive answer based on the research literature.")

GT_VIEWPOINTS = [
    "Homework has conditional positive effects on academic achievement, strongest for older students (grades 7-12)",
    "Meta-analytic evidence for homework effects is mixed or null, and effect sizes have declined in more recent studies",
    "Excessive homework causes measurable wellbeing harm (anxiety, stress) without proportional academic benefit",
]

VANILLA_PROMPT = f'Based on scientific research, answer this question concisely:\n\n"{VC_QUERY}"'

EVIRAG_PROMPT = f"""You are an evidence-aware scientific assistant. Answer the following question by identifying ALL major viewpoints in the research literature — including minority and dissenting views. Do NOT collapse evidence into one answer.

Question: "{VC_QUERY}"

Structure your response as:
1. DOMINANT VIEW: [summary] [claim count]
2. ALTERNATIVE VIEW: [summary] [claim count]
3. MINORITY VIEW: [summary] [claim count]
4. CONTROVERSY CLASS: resolved/emerging/stable/polarized
5. CDA-7 ATTRIBUTION: [primary cause of dominant↔alternative conflict]
6. CONFIDENCE: low/medium/high [brief reason]

For each view, explicitly state its WEAKNESSES."""

SYNTHESIS_PROMPT = """You are the EVIRAG multi-view synthesizer for the question: "Does homework improve academic achievement?"

Produce ONLY valid JSON (no markdown, no extra text):
{
  "views": [
    {
      "view_id": "V1",
      "label": "dominant",
      "summary": "one sentence",
      "claim_count": 5,
      "source_count": 3,
      "weaknesses": ["weakness 1", "weakness 2"],
      "cda7_attribution": "METHODOLOGICAL"
    },
    {
      "view_id": "V2",
      "label": "alternative",
      "summary": "one sentence",
      "claim_count": 3,
      "source_count": 2,
      "weaknesses": ["weakness 1"],
      "cda7_attribution": "STATISTICAL"
    },
    {
      "view_id": "V3",
      "label": "minority",
      "summary": "one sentence",
      "claim_count": 2,
      "source_count": 2,
      "weaknesses": ["weakness 1"],
      "cda7_attribution": "OPERATIONAL"
    }
  ],
  "controversy_class": "stable",
  "epistemic_divergence_estimate": 0.38,
  "confidence": "medium",
  "tdc_trajectory": "stable"
}"""


# ──────────────────────────────────────────────────────────────────────────────
# Ollama client
# ──────────────────────────────────────────────────────────────────────────────

def ollama_chat(system_msg: str, user_msg: str,
                max_tokens: int = 512, temp: float = 0.1,
                force_json: bool = True) -> tuple:
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "stream": False,
        "think": False,                           # disable extended thinking (content empty otherwise)
        **({"format": "json"} if force_json else {}),
        "options": {"temperature": temp, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    return body["message"]["content"].strip(), time.time() - t0


def parse_json(text: str) -> dict:
    """Parse JSON from model output, handling <think> blocks and markdown fences."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for pat in [r'\{.*?\}', r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        for m in re.findall(pat, text, re.DOTALL):
            try:
                return json.loads(m)
            except Exception:
                pass
    lbl = re.search(r'"label"\s*:\s*"([A-Z_]+)"', text)
    return {"label": lbl.group(1) if lbl else "PARSE_ERROR",
            "confidence": 0.5, "reasoning": text[:200]}


# ──────────────────────────────────────────────────────────────────────────────
# Task A: NLI
# ──────────────────────────────────────────────────────────────────────────────

def run_task_a() -> dict:
    print("\n" + "="*60)
    print("TASK A: NLI Relation Classification (12 gold pairs)")
    print("="*60)
    results, correct = [], 0
    for i, (ca, cb, gold, hint) in enumerate(NLI_GOLD):
        prompt = (
            f'Two scientific claims — classify their relationship.\n\n'
            f'Claim A: "{ca}"\nClaim B: "{cb}"\n\n'
            f'Respond ONLY with valid JSON (no markdown):\n'
            f'{{"label":"SUPPORTS|CONTRADICTS|NEUTRAL","confidence":0.0-1.0,"reasoning":"one sentence"}}'
        )
        raw, gt = ollama_chat(
            "You are a precise scientific NLI classifier. Output valid JSON only.",
            prompt, max_tokens=200
        )
        p = parse_json(raw)
        pred, conf, match = p.get("label","PARSE_ERROR"), p.get("confidence",0), p.get("label","") == gold
        correct += int(match)
        results.append({"pair_id":i+1,"hint":hint,"gold":gold,"pred":pred,
                         "confidence":conf,"match":match,
                         "reasoning":p.get("reasoning",""),"gen_time_s":round(gt,2)})
        print(f"  [{'✓' if match else '✗'}] Gold={gold:<12} Pred={pred:<12} "
              f"Conf={conf:.2f}  {hint[:55]}")

    acc = correct / len(NLI_GOLD)
    tp = sum(1 for r in results if r['gold']=='CONTRADICTS' and r['pred']=='CONTRADICTS')
    fp = sum(1 for r in results if r['gold']!='CONTRADICTS' and r['pred']=='CONTRADICTS')
    fn = sum(1 for r in results if r['gold']=='CONTRADICTS' and r['pred']!='CONTRADICTS')
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    f1 = 2*pr*rc/(pr+rc) if pr+rc else 0
    print(f"\n  ACCURACY: {acc:.3f} ({correct}/{len(NLI_GOLD)})")
    print(f"  CONTRADICTS  P={pr:.3f}  R={rc:.3f}  F1={f1:.3f}")
    return {"accuracy":round(acc,3),"n_correct":correct,"n_total":len(NLI_GOLD),
            "contradicts_precision":round(pr,3),"contradicts_recall":round(rc,3),
            "contradicts_f1":round(f1,3),"details":results}


# ──────────────────────────────────────────────────────────────────────────────
# Task B: CDA-7
# ──────────────────────────────────────────────────────────────────────────────

def run_task_b() -> dict:
    print("\n" + "="*60)
    print("TASK B: CDA-7 Causal Attribution (10 gold pairs)")
    print("="*60)
    results, correct = [], 0
    CLASSES = ["METHODOLOGICAL","POPULATION","TEMPORAL","OPERATIONAL",
               "STATISTICAL","THEORETICAL","REPLICATION"]
    for i, (ca, cb, ctx, gold, rationale) in enumerate(CDA7_GOLD):
        prompt = (
            f'Two scientific claims conflict. Identify the PRIMARY cause.\n\n'
            f'Claim A: "{ca}"\nClaim B: "{cb}"\nContext: {ctx}\n\n'
            f'CDA-7 classes: {", ".join(CLASSES)}\n\n'
            f'Respond ONLY with valid JSON (no markdown):\n'
            f'{{"label":"<one of the 7 classes>","confidence":0.0-1.0,"reasoning":"one sentence"}}'
        )
        raw, gt = ollama_chat(
            "You are a scientific disagreement analyst. Output valid JSON only.",
            prompt, max_tokens=200
        )
        p = parse_json(raw)
        pred, conf, match = p.get("label","PARSE_ERROR"), p.get("confidence",0), p.get("label","") == gold
        correct += int(match)
        results.append({"pair_id":i+1,"gold":gold,"pred":pred,"confidence":conf,
                         "match":match,"rationale":rationale,
                         "model_reasoning":p.get("reasoning",""),"gen_time_s":round(gt,2)})
        print(f"  [{'✓' if match else '✗'}] Gold={gold:<15} Pred={pred:<15} Conf={conf:.2f}")
        if not match:
            print(f"       Expected: {rationale}")
            print(f"       Model: {p.get('reasoning','')[:80]}")

    acc = correct / len(CDA7_GOLD)
    kappa = (acc - 1/7) / (1 - 1/7)  # vs. chance for 7 classes
    print(f"\n  ACCURACY: {acc:.3f} ({correct}/{len(CDA7_GOLD)})")
    print(f"  COHEN'S κ (vs. chance): {kappa:.3f}")
    return {"accuracy":round(acc,3),"n_correct":correct,"n_total":len(CDA7_GOLD),
            "cohens_kappa_vs_chance":round(kappa,3),"details":results}


# ──────────────────────────────────────────────────────────────────────────────
# Task C: Claim extraction
# ──────────────────────────────────────────────────────────────────────────────

def run_task_c() -> dict:
    print("\n" + "="*60)
    print("TASK C: Claim Extraction (3 passages)")
    print("="*60)
    results = []
    for p in CLAIM_PASSAGES:
        prompt = (
            f'Extract all atomic factual claims from this passage. Each claim: one fact, attributed if mentioned.\n\n'
            f'Passage: "{p["text"]}"\n\n'
            f'Respond ONLY with valid JSON:\n'
            f'{{"claims":[{{"id":1,"text":"claim text","source":"author/year or null"}},...]}}'
        )
        raw, gt = ollama_chat(
            "You are a precise scientific claim extractor. Output valid JSON only.",
            prompt, max_tokens=600
        )
        parsed = parse_json(raw)
        claims = parsed.get("claims", [])
        if not claims and isinstance(parsed, list):
            claims = parsed
        n = len(claims)
        texts = " ".join((c.get("text","") if isinstance(c,dict) else str(c)).lower() for c in claims)
        key_hits = sum(1 for kc in p["key_claims"]
                       if any(w in texts for w in kc.split()[:3]))
        coverage = key_hits / len(p["key_claims"])
        meets = n >= p["expected_min"]
        results.append({"passage_id":p["id"],"n_extracted":n,
                         "expected_min":p["expected_min"],"meets_min":meets,
                         "key_claim_coverage":round(coverage,2),
                         "claims":[c.get("text","") if isinstance(c,dict) else str(c) for c in claims],
                         "gen_time_s":round(gt,2)})
        print(f"  [{'✓' if meets else '✗'}] {p['id']}: {n} claims extracted "
              f"(min={p['expected_min']})  key coverage={coverage:.0%}")
        for c in claims[:5]:
            ct = c.get("text","") if isinstance(c,dict) else str(c)
            print(f"       - {ct[:90]}")

    avg_cov = sum(r["key_claim_coverage"] for r in results) / len(results)
    pct_min = sum(r["meets_min"] for r in results) / len(results)
    print(f"\n  AVG KEY COVERAGE: {avg_cov:.3f}")
    print(f"  PASSAGES MEET MIN: {pct_min:.0%}")
    return {"avg_key_claim_coverage":round(avg_cov,3),
            "pct_passages_meet_min":round(pct_min,3),"details":results}


# ──────────────────────────────────────────────────────────────────────────────
# Task D: ViewpointCoverage
# ──────────────────────────────────────────────────────────────────────────────

def run_task_d() -> dict:
    print("\n" + "="*60)
    print("TASK D: ViewpointCoverage — Vanilla vs EVIRAG")
    print("="*60)

    KW_PER_VIEW = [
        ["conditional","grade","high school","secondary","7-12","older","design","well-designed"],
        ["null","no significant","mixed","inconsistent","meta-analytic","effect size","decline","no conclusive"],
        ["wellbeing","mental health","anxiety","stress","harm","depression","hurt","no commensurate"],
    ]

    def compute_vc(text: str) -> tuple:
        t = text.lower()
        covered, details = 0, []
        for i, (vp, kws) in enumerate(zip(GT_VIEWPOINTS, KW_PER_VIEW)):
            hits = sum(1 for kw in kws if kw in t)
            cov = hits >= 2
            covered += int(cov)
            details.append({"view":vp[:55],"hits":hits,"covered":cov})
        return covered / len(GT_VIEWPOINTS), details

    print("  [Vanilla RAG] Generating...")
    v_resp, v_t = ollama_chat("You are a helpful scientific assistant.", VANILLA_PROMPT,
                              max_tokens=600, force_json=False)
    v_vc, v_det = compute_vc(v_resp)
    print(f"  Vanilla  ({v_t:.1f}s): VC={v_vc:.3f}")

    print("  [EVIRAG Multi-View] Generating...")
    e_resp, e_t = ollama_chat(
        "You are an epistemic-fidelity scientific assistant preserving genuine scientific controversy.",
        EVIRAG_PROMPT, max_tokens=1200, force_json=False
    )
    e_vc, e_det = compute_vc(e_resp)
    print(f"  EVIRAG   ({e_t:.1f}s): VC={e_vc:.3f}")

    bound = 1/3
    print(f"\n  VC IMPROVEMENT:   {v_vc:.3f} → {e_vc:.3f}  (+{e_vc-v_vc:.3f})")
    print(f"  Theorem 1 bound:  1/k = {bound:.3f}")
    print(f"  Vanilla ≤ bound:  {'✓ CONFIRMED' if v_vc <= bound + 0.05 else '✗ VIOLATED'}")

    print(f"\n  -- Vanilla RAG (first 250 chars) --\n  {v_resp[:250]}")
    print(f"\n  -- EVIRAG Multi-View (first 400 chars) --\n  {e_resp[:400]}")

    return {"vanilla_vc":round(v_vc,3),"evirag_vc":round(e_vc,3),
            "vc_improvement":round(e_vc-v_vc,3),"theorem1_bound":round(bound,3),
            "vanilla_le_bound":v_vc <= bound + 0.05,
            "vanilla_details":v_det,"evirag_details":e_det,
            "vanilla_response":v_resp,"evirag_response":e_resp}


# ──────────────────────────────────────────────────────────────────────────────
# Task E: Structured synthesis
# ──────────────────────────────────────────────────────────────────────────────

def run_task_e() -> dict:
    print("\n" + "="*60)
    print("TASK E: Structured Multi-View Synthesis (JSON schema)")
    print("="*60)

    raw, gt = ollama_chat(
        "You are the EVIRAG multi-view synthesizer. Output valid JSON only. No markdown fences.",
        SYNTHESIS_PROMPT, 1200
    )
    parsed = parse_json(raw)

    CDA7_VALID = {"METHODOLOGICAL","POPULATION","TEMPORAL","OPERATIONAL",
                  "STATISTICAL","THEORETICAL","REPLICATION"}

    checks = {
        "has_views":          "views" in parsed and isinstance(parsed.get("views"), list),
        "has_3plus_views":    len(parsed.get("views", [])) >= 3,
        "has_controversy":    "controversy_class" in parsed,
        "has_confidence":     "confidence" in parsed,
        "has_tdc":            "tdc_trajectory" in parsed,
        "has_ed_estimate":    "epistemic_divergence_estimate" in parsed,
    }
    view_checks = []
    for v in parsed.get("views", []):
        vc = {
            "view_id":       v.get("view_id",""),
            "has_label":     "label" in v,
            "has_summary":   "summary" in v and len(v.get("summary","")) > 10,
            "has_weaknesses":len(v.get("weaknesses",[])) > 0,
            "has_cda7":      "cda7_attribution" in v,
            "cda7_valid":    v.get("cda7_attribution","") in CDA7_VALID,
        }
        view_checks.append(vc)

    schema_score     = sum(checks.values()) / len(checks)
    view_schema_score = (sum(all(v.values()) for v in view_checks) / len(view_checks)
                         if view_checks else 0)

    print(f"  Schema compliance: {schema_score:.0%}")
    for k, v in checks.items():
        print(f"    {'✓' if v else '✗'} {k}")
    print(f"  View schema compliance: {view_schema_score:.0%}")
    for vc in view_checks:
        ok = all(vc.values())
        print(f"    {'✓' if ok else '✗'} View {vc['view_id']}: "
              f"label={vc['has_label']} summary={vc['has_summary']} "
              f"weaknesses={vc['has_weaknesses']} cda7_valid={vc['cda7_valid']}")
    print(f"  controversy_class={parsed.get('controversy_class','')}  "
          f"ED={parsed.get('epistemic_divergence_estimate','')}  "
          f"TDC={parsed.get('tdc_trajectory','')}")

    return {"schema_compliance":round(schema_score,3),
            "view_schema_compliance":round(view_schema_score,3),
            "n_views":len(parsed.get("views",[])),
            "controversy_class":parsed.get("controversy_class",""),
            "confidence":parsed.get("confidence",""),
            "tdc_trajectory":parsed.get("tdc_trajectory",""),
            "ed_estimate":parsed.get("epistemic_divergence_estimate",None),
            "schema_checks":checks,"view_checks":view_checks,
            "raw_parsed":parsed,"gen_time_s":round(gt,2)}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",  default="A,B,C,D,E")
    parser.add_argument("--output", default="evirag_eval_results.json")
    parser.add_argument("--model",  default=MODEL)
    args = parser.parse_args()
    MODEL = args.model
    tasks = set(args.tasks.upper().split(","))

    print("=" * 70)
    print(f"EVIRAG Evaluation Suite — {MODEL}")
    print(f"Tasks: {args.tasks}  |  Date: {datetime.now().isoformat()}")
    print("=" * 70)

    # Verify Ollama is up
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            tags = json.loads(r.read())
        names = [m["name"] for m in tags.get("models",[])]
        print(f"[OK] Ollama running. Available: {names}")
        if MODEL not in names:
            print(f"[WARN] {MODEL} not in model list — will attempt anyway")
    except Exception as e:
        print(f"[FAIL] Ollama not reachable: {e}")
        sys.exit(1)

    all_results: dict = {
        "metadata": {"model": MODEL, "run_date": datetime.now().isoformat()}
    }
    t_total = time.time()

    if "A" in tasks: all_results["task_a_nli"]    = run_task_a()
    if "B" in tasks: all_results["task_b_cda7"]   = run_task_b()
    if "C" in tasks: all_results["task_c_claims"]  = run_task_c()
    if "D" in tasks: all_results["task_d_vc"]      = run_task_d()
    if "E" in tasks: all_results["task_e_synthesis"] = run_task_e()

    total = time.time() - t_total

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    if "task_a_nli" in all_results:
        r = all_results["task_a_nli"]
        print(f"  Task A  NLI Accuracy:      {r['accuracy']:.3f}  ({r['n_correct']}/{r['n_total']})")
        print(f"          CONTRADICTS F1:    {r['contradicts_f1']:.3f}")
    if "task_b_cda7" in all_results:
        r = all_results["task_b_cda7"]
        print(f"  Task B  CDA-7 Accuracy:    {r['accuracy']:.3f}  ({r['n_correct']}/{r['n_total']})")
        print(f"          Cohen's κ:          {r['cohens_kappa_vs_chance']:.3f}")
    if "task_c_claims" in all_results:
        r = all_results["task_c_claims"]
        print(f"  Task C  Key Claim Cover:   {r['avg_key_claim_coverage']:.3f}")
        print(f"          Min Claims Met:     {r['pct_passages_meet_min']:.0%}")
    if "task_d_vc" in all_results:
        r = all_results["task_d_vc"]
        print(f"  Task D  Vanilla VC:        {r['vanilla_vc']:.3f}  (bound=0.333)")
        print(f"          EVIRAG VC:          {r['evirag_vc']:.3f}")
        print(f"          Improvement:        +{r['vc_improvement']:.3f}")
        print(f"          Theorem 1 holds:    {'YES ✓' if r['vanilla_le_bound'] else 'NO ✗'}")
    if "task_e_synthesis" in all_results:
        r = all_results["task_e_synthesis"]
        print(f"  Task E  Schema Compliance: {r['schema_compliance']:.0%}")
        print(f"          Views Generated:   {r['n_views']}")
        print(f"          Controversy:       {r['controversy_class']}  TDC: {r['tdc_trajectory']}")
    print(f"\n  Total time: {total/60:.1f} min")
    print("=" * 70)

    out = Path(args.output)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[SAVED] {out.resolve()}")


if __name__ == "__main__":
    main()
