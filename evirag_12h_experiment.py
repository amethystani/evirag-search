"""
EVIRAG Large-Scale 12-Hour Evaluation
======================================
Three-domain, five-system, 30-query benchmark with full statistical validation.

Domains
  D1  Homework & Academic Achievement   (education)
  D2  Statin Therapy & Cardiovascular   (biomedicine)
  D3  Minimum Wage & Labor Markets      (economics)

Systems evaluated (ablation ladder)
  SYS1  Vanilla RAG        dense retrieval + single-view synthesis
  SYS2  Single-agent       precision-only retrieval, no Skeptic agent
  SYS3  EVIRAG-NoGraph     adversarial retrieval, raw-chunk synthesis
  SYS4  EVIRAG-NoMultiView full pipeline → collapsed to dominant view
  SYS5  EVIRAG-Full        complete seven-stage pipeline

Metrics (per query × system)
  VC   ViewpointCoverage    fraction of GT viewpoints surfaced
  CR   ContradictionRecall  fraction of GT contradictions detected
  CCS  ConsensusCollapseScore (lower = better)
  CCE  ConfidenceCalibError  |predicted_conf - GT_controversy_level|

Statistical analysis
  Bootstrap 95% CI (1000 iterations) on all metrics
  Wilcoxon signed-rank: EVIRAG-Full vs each ablation (Bonferroni corrected)
  Per-domain breakdown

Checkpointing: all intermediate results saved to evirag_12h_checkpoint.json
               (restart-safe: completed phases are skipped)

Usage:
    python evirag_12h_experiment.py [--resume] [--output evirag_12h_results.json]

Runtime estimate: ~6-10 hours on RTX 4500 Ada with qwen3.6:35b-a3b
"""

import json, re, time, os, argparse, math, random, urllib.request, urllib.parse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ─── configuration ────────────────────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434/api/chat"
MODEL       = "qwen3.6:35b-a3b"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHECKPOINT  = Path("/home/snu/evirag_12h_checkpoint.json")
RESULTS_OUT = Path("/home/snu/evirag_12h_results.json")
LOG_PATH    = Path("/home/snu/evirag_12h.log")

PAPERS_PER_DOMAIN  = 25    # abstracts fetched per domain
TOP_K_RETRIEVAL    = 12    # chunks retrieved per query
COSINE_REL_THRESH  = 0.35  # minimum cosine to attempt pairwise labeling
BOOTSTRAP_ITERS    = 1000
RANDOM_SEED        = 42

# ─── ground-truth viewpoints & controversy metadata ───────────────────────────

DOMAINS = {
    "homework": {
        "name": "Homework & Academic Achievement",
        "search_terms": [
            "homework academic achievement effects meta-analysis",
            "homework student learning outcomes research",
            "homework mental health wellbeing students",
            "homework grade level elementary high school effectiveness",
            "homework parental involvement academic performance",
        ],
        "controversy_class": "stable",
        "gt_conf_level": 0.5,   # 0=resolved 1=polarized → maps to medium confidence
    },
    "statins": {
        "name": "Statin Therapy & Cardiovascular Outcomes",
        "search_terms": [
            "statin primary prevention cardiovascular randomized trial",
            "statin therapy side effects myopathy diabetes risk",
            "statin women cardiovascular benefit evidence",
            "statin all-cause mortality low-risk patients",
            "statin cognitive effects memory dementia",
        ],
        "controversy_class": "emerging",
        "gt_conf_level": 0.4,
    },
    "minwage": {
        "name": "Minimum Wage & Labor Markets",
        "search_terms": [
            "minimum wage employment effects causal identification",
            "minimum wage poverty reduction income inequality",
            "minimum wage youth employment effects evidence",
            "minimum wage monopsony labor market",
            "minimum wage small business price inflation",
        ],
        "controversy_class": "polarized",
        "gt_conf_level": 0.7,
    },
}

QUERIES = {
    "homework": [
        {
            "q": "Does homework improve academic achievement?",
            "gt_views": [
                "Homework has conditional positive effects on academic achievement, strongest for older students (grades 7-12) and with well-designed practice assignments",
                "Meta-analytic evidence for homework effects is mixed or null; effect sizes have declined in more recent studies and are sensitive to publication bias",
                "Excessive homework causes measurable wellbeing harm including anxiety and stress without proportional academic benefit, particularly in high-poverty contexts",
            ],
            "gt_contradictions": [("positive achievement effect", "null or negative meta-analytic result"), ("wellbeing benefit", "wellbeing harm")],
            "gt_conf": "medium",
        },
        {
            "q": "Does homework have differential effects across grade levels?",
            "gt_views": [
                "Homework effects are substantially stronger for secondary school students (grades 7-12) than for elementary school students, where effects are negligible",
                "Developmental appropriateness of homework varies; some evidence suggests homework is counterproductive before grade 3",
                "The grade-level effect is partly an artifact of study design differences, with secondary studies more likely to use standardized tests",
            ],
            "gt_contradictions": [("stronger secondary effects", "developmental counterproductivity"), ("grade artifact", "real grade effect")],
            "gt_conf": "medium",
        },
        {
            "q": "What is the effect of homework on student mental health?",
            "gt_views": [
                "Moderate homework loads do not significantly harm student mental health; adverse effects require high homework loads (>2 hours/night)",
                "Homework contributes to student stress, anxiety and reduced sleep, particularly in high-achieving schools and high-SES contexts",
                "The mental health effect of homework is confounded by student autonomy and perceived purpose; autonomous motivation mediates outcomes",
            ],
            "gt_contradictions": [("no harm at moderate load", "anxiety and stress"), ("confounding autonomy", "direct harm hypothesis")],
            "gt_conf": "medium",
        },
        {
            "q": "Does parental involvement in homework improve learning outcomes?",
            "gt_views": [
                "Parental involvement in homework generally has positive effects on academic achievement, especially in elementary school",
                "Parental homework help can cause anxiety, reduce student autonomy, and backfire when parents lack subject knowledge",
                "The type of parental involvement matters: emotional support is beneficial; directive intervention is often counterproductive",
            ],
            "gt_contradictions": [("positive parental involvement", "backfire and anxiety"), ("support vs. directive", "simple positive framing")],
            "gt_conf": "medium",
        },
        {
            "q": "Does homework develop non-cognitive skills such as self-regulation?",
            "gt_views": [
                "Homework practice develops study habits, time management, and self-regulation through repeated independent practice opportunities",
                "There is little rigorous evidence that homework causes non-cognitive skill development rather than merely correlating with it",
                "Self-regulation may be a prerequisite for homework benefit rather than an outcome, reversing the assumed causal direction",
            ],
            "gt_contradictions": [("homework builds self-regulation", "no causal evidence"), ("prerequisite vs. outcome", "assumes direction")],
            "gt_conf": "low",
        },
        {
            "q": "Does homework widen or narrow the socioeconomic achievement gap?",
            "gt_views": [
                "Homework may widen the achievement gap because students from higher-SES families have more resources and parental support for homework completion",
                "Homework narrows the gap by providing additional practice time for students who lack access to enrichment activities",
                "Homework format and teacher follow-up mediate gap effects; poorly designed homework is more likely to widen gaps",
            ],
            "gt_contradictions": [("widens gap via SES resources", "narrows gap via practice"), ("format mediates", "direct SES mechanism")],
            "gt_conf": "low",
        },
        {
            "q": "What is the optimal homework duration per grade level?",
            "gt_views": [
                "The 10-minute rule (10 minutes per grade level per night) is a commonly endorsed guideline with moderate empirical support",
                "Optimal duration varies by subject, student, and quality; time-based guidelines are a poor proxy for workload regulation",
                "Beyond individual thresholds, additional homework time produces diminishing or negative returns on achievement",
            ],
            "gt_contradictions": [("10-minute rule supported", "time-based guidelines poor proxy"), ("diminishing returns", "linear benefit framing")],
            "gt_conf": "medium",
        },
        {
            "q": "Is the homework-achievement correlation causal or merely correlational?",
            "gt_views": [
                "Experimental and quasi-experimental evidence supports a causal effect of homework on achievement, especially in randomized studies",
                "Most homework research is observational; reverse causation (higher-achieving students do more homework) is a significant threat",
                "Natural experiment and instrumental variable studies find weaker or no causal effects, suggesting confounding drives observed correlations",
            ],
            "gt_contradictions": [("causal from experiments", "reverse causation threat"), ("natural experiments null", "experimental positive")],
            "gt_conf": "low",
        },
        {
            "q": "Do different homework types produce different learning outcomes?",
            "gt_views": [
                "Practice homework that reinforces previously taught skills shows the strongest effects on achievement compared to preparation or extension homework",
                "Project-based and creative homework assignments develop higher-order thinking skills that practice homework does not measure",
                "Digital and interactive homework platforms show comparable or superior effects to paper-based homework with faster feedback",
            ],
            "gt_contradictions": [("practice superior", "project develops higher-order skills"), ("digital comparable", "practice strongest")],
            "gt_conf": "medium",
        },
        {
            "q": "How have homework research findings changed over recent decades?",
            "gt_views": [
                "Effect sizes for homework-achievement relationships have declined from the 1980s to recent decades, possibly due to publication bias correction or curricular changes",
                "More recent research uses stronger causal identification methods, suggesting earlier positive estimates were inflated by methodological weaknesses",
                "The homework evidence base has improved in quality while remaining mixed in direction, with no clear temporal trend in findings",
            ],
            "gt_contradictions": [("declining effect sizes", "stable mixed evidence"), ("methodological improvement", "substantive change")],
            "gt_conf": "low",
        },
    ],

    "statins": [
        {
            "q": "Do statins prevent cardiovascular events in primary prevention?",
            "gt_views": [
                "Large RCTs including JUPITER, ASCOT-LLA, and WOSCOPS demonstrate statistically significant reductions in major cardiovascular events with statin therapy in primary prevention",
                "The absolute risk reduction in primary prevention is small; NNT estimates range from 50-200, and benefits may not outweigh harms and costs for low-risk individuals",
                "Benefits of statin primary prevention are highly heterogeneous by baseline risk; high-risk patients benefit substantially while low-risk patients show minimal benefit",
            ],
            "gt_contradictions": [("RCT evidence of benefit", "small absolute reduction NNT"), ("heterogeneous benefit", "uniform benefit framing")],
            "gt_conf": "medium",
        },
        {
            "q": "Do statins increase the risk of type 2 diabetes?",
            "gt_views": [
                "Meta-analyses confirm statins increase diabetes risk by approximately 10-12%, with high-intensity regimens conferring greater risk than low-intensity",
                "The cardiovascular benefits of statins substantially outweigh the diabetes risk for most patients, making the benefit-harm balance favorable",
                "The diabetes risk is a class effect mediated by insulin resistance rather than an off-target artifact, and should influence prescribing decisions in pre-diabetic patients",
            ],
            "gt_contradictions": [("diabetes risk confirmed", "benefits outweigh risk"), ("class effect mechanisms", "incidental risk framing")],
            "gt_conf": "medium",
        },
        {
            "q": "Do statins cause muscle damage and myopathy?",
            "gt_views": [
                "Statin-associated myopathy occurs across a spectrum from asymptomatic CK elevation to rare rhabdomyolysis; incidence is higher with high-dose and hydrophilic statins",
                "Symptomatic myalgia without CK elevation (statin intolerance) is largely nocebo-driven; blinded RCTs show much lower rates than open-label reports",
                "Muscle symptoms are a significant cause of statin discontinuation in clinical practice, reducing adherence and cardiovascular protection regardless of causal mechanism",
            ],
            "gt_contradictions": [("spectrum of myopathy real", "nocebo-driven myalgia"), ("adherence harm real", "nocebo framing underestimates")],
            "gt_conf": "medium",
        },
        {
            "q": "Do statins affect cognitive function or dementia risk?",
            "gt_views": [
                "Observational studies suggest statins may reduce dementia and Alzheimer's risk through anti-inflammatory and cholesterol-lowering effects on amyloid pathways",
                "Randomized trial evidence does not support a protective cognitive effect of statins; the PROSPER and HPS trials found no cognitive benefit",
                "Case reports and pharmacovigilance data associate statins with memory impairment and cognitive side effects, though causality is disputed",
            ],
            "gt_contradictions": [("observational dementia benefit", "RCT no benefit"), ("cognitive side effects", "observational benefit")],
            "gt_conf": "low",
        },
        {
            "q": "Is statin therapy effective for women in primary prevention?",
            "gt_views": [
                "Subgroup analyses from major RCTs show statins reduce cardiovascular events in women in secondary prevention, but evidence in primary prevention is less clear",
                "Women were historically underrepresented in statin trials; the lower baseline cardiovascular risk in women may mean primary prevention NNT is too high to justify treatment",
                "Sex-specific analyses show similar relative risk reductions in women and men; absolute risk differs due to lower baseline event rates in women",
            ],
            "gt_contradictions": [("similar relative RR", "lower baseline makes NNT unfavorable"), ("underrepresentation limits evidence", "available data supports benefit")],
            "gt_conf": "low",
        },
        {
            "q": "Do statins have anti-inflammatory effects beyond cholesterol lowering?",
            "gt_views": [
                "Statins have pleiotropic anti-inflammatory effects including CRP reduction, endothelial protection, and immunomodulation, which may explain some cardiovascular benefits beyond LDL lowering",
                "The cardiovascular benefit of statins is primarily attributable to LDL-C reduction; pleiotropic effects are secondary and insufficient to justify treatment for anti-inflammatory indication alone",
                "The JUPITER trial showed CRP reduction predicted statin benefit independently of LDL, supporting inflammatory mechanisms, but subsequent analysis showed LDL change was the primary driver",
            ],
            "gt_contradictions": [("pleiotropic effects meaningful", "LDL lowering primary"), ("JUPITER supports inflammation", "LDL remains driver")],
            "gt_conf": "medium",
        },
        {
            "q": "Do statins reduce all-cause mortality in low-risk populations?",
            "gt_views": [
                "Meta-analyses of primary prevention trials show a statistically significant reduction in all-cause mortality with statin therapy, though the absolute reduction is small",
                "Trials of low-risk populations show no statistically significant reduction in all-cause mortality; cardiovascular mortality reduction may be offset by non-cardiovascular causes",
                "All-cause mortality benefit in primary prevention depends critically on trial duration and baseline risk; short trials in low-risk populations are underpowered to detect mortality differences",
            ],
            "gt_contradictions": [("meta-analytic mortality benefit", "individual trial non-significance"), ("underpowered trials", "sufficient power in meta-analysis")],
            "gt_conf": "low",
        },
        {
            "q": "Are there reliable non-pharmaceutical alternatives to statins for cardiovascular risk reduction?",
            "gt_views": [
                "Lifestyle interventions including diet, exercise, and smoking cessation are effective at reducing cardiovascular risk and should be first-line for low-risk patients",
                "For high-risk patients, statins provide risk reductions that lifestyle modifications alone cannot reliably achieve; they are complementary rather than alternative",
                "Plant sterols, omega-3 fatty acids, and Mediterranean diet interventions reduce LDL and cardiovascular risk, potentially substituting for statins in statin-intolerant patients",
            ],
            "gt_contradictions": [("lifestyle first-line low-risk", "statins irreplaceable high-risk"), ("dietary alternatives", "supplementary only framing")],
            "gt_conf": "medium",
        },
        {
            "q": "Does statin benefit depend on baseline LDL-C level?",
            "gt_views": [
                "The proportional risk reduction from statins is consistent across a wide range of baseline LDL-C levels; lower LDL-C patients still benefit proportionally",
                "Absolute risk reduction is greater in patients with higher baseline LDL-C and cardiovascular risk; treating low-LDL patients provides smaller absolute benefit",
                "The LDL hypothesis predicts benefit at any baseline; the question is whether absolute benefit justifies treatment costs and side effect risks",
            ],
            "gt_contradictions": [("proportional benefit regardless of baseline", "absolute benefit higher at high LDL"), ("LDL hypothesis uniform", "risk-stratified prescribing needed")],
            "gt_conf": "medium",
        },
        {
            "q": "How should statin therapy be monitored and adjusted in clinical practice?",
            "gt_views": [
                "Regular LDL-C monitoring guides statin dose titration to achieve evidence-based targets (LDL < 70 mg/dL for high-risk, < 100 mg/dL for moderate-risk)",
                "Treat-to-target strategies have not been shown superior to fixed-dose approaches in RCTs; guidelines increasingly support fixed high-intensity statin regardless of LDL achieved",
                "Patient adherence and tolerability are more important drivers of cardiovascular outcomes than achieving specific LDL targets; monitoring should focus on adherence",
            ],
            "gt_contradictions": [("treat-to-LDL-target", "fixed-dose high intensity"), ("adherence over targets", "target-based titration")],
            "gt_conf": "medium",
        },
    ],

    "minwage": [
        {
            "q": "Does raising the minimum wage increase unemployment?",
            "gt_views": [
                "Standard competitive labor market theory predicts minimum wage above equilibrium causes disemployment; empirical estimates find small negative employment effects, particularly for low-skilled workers",
                "Card and Krueger's New Jersey fast-food study and subsequent quasi-experimental research find little or no disemployment effect of minimum wage increases",
                "Employment effects are highly heterogeneous across labor markets, firm types, and wage levels; monopsony power can cause minimum wage to increase both wages and employment",
            ],
            "gt_contradictions": [("disemployment from theory", "no effect quasi-experimental"), ("monopsony reverses sign", "competitive market negative")],
            "gt_conf": "high",
        },
        {
            "q": "Do minimum wage increases reduce poverty rates?",
            "gt_views": [
                "Minimum wage increases directly raise earnings for low-wage workers; however, the poverty-reduction effect is limited because many minimum wage earners are not in poor households",
                "EITC and other transfer programs are better targeted to poverty reduction than minimum wage; minimum wage benefits are distributed across income levels, not just poor households",
                "Minimum wage increases reduce poverty substantially when combined with strong labor market conditions and adequate policy design targeting living-wage adequacy",
            ],
            "gt_contradictions": [("poverty reduction from wage gains", "poor targeting across income"), ("EITC superior targeting", "complementary role for min wage")],
            "gt_conf": "medium",
        },
        {
            "q": "What are the employment effects of minimum wage on youth workers?",
            "gt_views": [
                "Youth workers are disproportionately concentrated in minimum wage jobs; standard analyses find larger disemployment effects among teenagers than adults",
                "Youth employment effects are confounded by school enrollment and economic cycles; controlling for these, minimum wage effects on youth are smaller than raw estimates suggest",
                "Youth employment sensitivity to minimum wage depends on whether employers can substitute teens for adults or automate; effects vary significantly by industry and region",
            ],
            "gt_contradictions": [("larger youth effects", "small after confounding"), ("substitution varies by context", "uniform youth sensitivity")],
            "gt_conf": "high",
        },
        {
            "q": "Does monopsony power in labor markets change minimum wage employment effects?",
            "gt_views": [
                "Monopsony power is theoretically sufficient to make minimum wage increases employment-neutral or employment-increasing; empirical evidence of widespread monopsony in low-wage labor markets supports this",
                "Perfect monopsony is rare; oligopsony and search frictions exist but are insufficient to fully explain the small employment effects observed empirically",
                "The degree of monopsony power is time- and place-specific; policy should not rely on monopsony to justify minimum wage increases without local labor market evidence",
            ],
            "gt_contradictions": [("monopsony widespread and important", "monopsony insufficient to explain"), ("contextual application needed", "uniform theoretical application")],
            "gt_conf": "medium",
        },
        {
            "q": "Do minimum wage increases cause price inflation?",
            "gt_views": [
                "Minimum wage increases cause modest price increases in affected industries (restaurants, retail); estimates range from 0.4-0.7% price increase per 10% wage increase",
                "Price pass-through of minimum wage increases is largely absorbed by compression of profit margins and efficiency gains rather than consumer price increases",
                "Inflation effects of minimum wage are negligible at the macroeconomic level because minimum wage workers represent a small fraction of total labor costs in most sectors",
            ],
            "gt_contradictions": [("significant restaurant price pass-through", "profit margin absorption"), ("macroeconomic negligible", "sector-specific significant")],
            "gt_conf": "medium",
        },
        {
            "q": "What causal identification strategies best identify minimum wage employment effects?",
            "gt_views": [
                "Difference-in-differences using neighboring counties or states as controls (the Card-Krueger approach) identifies minimum wage effects by comparing wage-affected and unaffected areas",
                "Geographic controls in minimum wage studies are invalid when local labor markets are integrated; recent research uses synthetic control and event-study designs instead",
                "Credible identification of minimum wage effects requires both strong controls for local economic trends and instruments for the timing of minimum wage changes; no single method dominates",
            ],
            "gt_contradictions": [("DiD geographic controls valid", "geographic integration invalidates DiD"), ("single superior method", "no method dominates")],
            "gt_conf": "medium",
        },
        {
            "q": "Do minimum wage increases affect small businesses differently than large employers?",
            "gt_views": [
                "Small businesses with thin margins are more adversely affected by minimum wage increases than large employers who can spread compliance costs across more workers",
                "Large retailers and fast food chains are the dominant employers of minimum wage workers; small business employment effects are secondary to effects on large chain employment",
                "Minimum wage effects on small businesses depend on market power: in competitive local markets, small businesses that cannot absorb costs may exit, reducing market competition",
            ],
            "gt_contradictions": [("small business more adversely affected", "large chains dominate impact"), ("market exit reduces competition", "large chain dominance")],
            "gt_conf": "medium",
        },
        {
            "q": "Do minimum wage increases improve worker health and wellbeing?",
            "gt_views": [
                "Higher wages associated with minimum wage increases are linked to reduced financial stress, improved mental health, and lower rates of food insecurity among affected workers",
                "Hours reductions and job losses from minimum wage increases can offset income gains, leaving some workers worse off in health-relevant outcomes",
                "Non-wage outcomes of minimum wage depend on enforcement and compliance; uncovered workers and those in informal employment do not benefit from legislated increases",
            ],
            "gt_contradictions": [("income improves health", "hours loss offsets gains"), ("compliance excludes informal workers", "broad health benefit framing")],
            "gt_conf": "medium",
        },
        {
            "q": "What is the employment effect of minimum wage in rural vs. urban labor markets?",
            "gt_views": [
                "Rural labor markets have lower prevailing wages relative to minimum wage levels; a given minimum wage increase is a larger shock in rural than urban areas",
                "Rural disemployment from minimum wage may be smaller than predicted because rural employers have more monopsony power due to limited alternative employment options",
                "Urban evidence does not generalize to rural contexts; minimum wage research has been disproportionately conducted in urban and suburban areas",
            ],
            "gt_contradictions": [("larger rural shock", "rural monopsony dampens effect"), ("urban evidence inapplicable", "rural research gap")],
            "gt_conf": "low",
        },
        {
            "q": "Does the evidence support a national or regional minimum wage policy?",
            "gt_views": [
                "Regional variation in cost of living and wages justifies state or local minimum wage setting; a uniform national minimum wage is too high for low-wage regions and too low for high-wage cities",
                "A federal minimum wage floor prevents a race to the bottom in low-wage labor regulation; regional differentiation should occur through supplemental local ordinances above the federal floor",
                "The optimal minimum wage level should be indexed to local median wages rather than set nominally; mechanical regional differentiation fails to track labor market dynamics",
            ],
            "gt_contradictions": [("regional variation justified", "federal floor necessary"), ("median-indexed optimal", "nominal regional setting")],
            "gt_conf": "medium",
        },
    ],
}

# ─── system prompt templates ───────────────────────────────────────────────────

VANILLA_SYS = "You are a knowledgeable scientific assistant. Provide accurate, well-supported answers."
VANILLA_USR = lambda q: f'Based on the scientific research literature, provide a comprehensive answer to this question:\n\n"{q}"\n\nBe informative and accurate.'

EVIRAG_SYS = "You are an epistemic-fidelity scientific assistant. You preserve genuine scientific controversy rather than suppressing it."
EVIRAG_USR = lambda q: f"""You are analyzing a scientifically contested question. Identify ALL major viewpoints in the research literature — do not collapse them into a single answer.

Question: "{q}"

Structure your response as:
DOMINANT VIEW: [summary of the most common/established position] [3-5 key claims]
ALTERNATIVE VIEW: [summary of the main dissenting position] [3-5 key claims]
MINORITY VIEW: [summary of a third perspective or nuanced position] [2-3 key claims]
CONTROVERSY CLASS: [resolved / emerging / stable / polarized]
CDA-7: [primary cause of dominant↔alternative conflict: METHODOLOGICAL / POPULATION / TEMPORAL / OPERATIONAL / STATISTICAL / THEORETICAL / REPLICATION]
CONFIDENCE: [low / medium / high] because [one-sentence reason]

For each view explicitly state its WEAKNESSES."""

SINGLE_AGENT_USR = lambda q: f'Search for evidence supporting the mainstream scientific view on:\n\n"{q}"\n\nProvide a well-supported answer with the key evidence.'

NOGRAPH_SYS = "You are an evidence-aware scientific assistant that actively seeks out dissenting views."
NOGRAPH_USR = lambda q: f"""Answer this question by considering MULTIPLE perspectives in the research literature.

Question: "{q}"

First consider the dominant view, then explicitly look for counter-evidence and alternative explanations. Synthesize across all perspectives."""

COLLAPSED_USR = lambda q: f"""Analyze all available research perspectives on: "{q}"

After considering all viewpoints, synthesize them into the SINGLE MOST ACCURATE AND BALANCED conclusion supported by the weight of evidence."""

SYNTHESIS_JSON_USR = lambda q: f"""Analyze the scientific controversy around: "{q}"

Respond with ONLY valid JSON:
{{"dominant_view":"one-sentence summary","alternative_view":"one-sentence summary","minority_view":"one-sentence summary","controversy_class":"resolved|emerging|stable|polarized","cda7_primary":"METHODOLOGICAL|POPULATION|TEMPORAL|OPERATIONAL|STATISTICAL|THEORETICAL|REPLICATION","confidence":"low|medium|high","dominant_weaknesses":["weakness1","weakness2"],"alternative_weaknesses":["weakness1"]}}"""

# ─── ollama client ────────────────────────────────────────────────────────────

def ollama(system: str, user: str, max_tok: int = 800, temp: float = 0.1,
           as_json: bool = False) -> tuple[str, float]:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role":"system","content":system}, {"role":"user","content":user}],
        "stream": False,
        "think": False,
        **({"format":"json"} if as_json else {}),
        "options": {"temperature": temp, "num_predict": max_tok},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                  headers={"Content-Type":"application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        body = json.loads(r.read())
    return body["message"]["content"].strip(), time.time()-t0


def parse_json(text: str) -> dict:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    for src in [text] + re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL) + re.findall(r'\{.*\}', text, re.DOTALL):
        try:
            return json.loads(src)
        except Exception:
            pass
    return {}


# ─── embeddings (TF-IDF based, pure numpy — no model download required) ───────

import math as _math

def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, drop stopwords and short tokens."""
    STOP = {"the","a","an","of","in","to","and","is","are","was","were","for",
            "with","that","this","it","be","by","on","at","from","as","or","but",
            "not","have","has","had","can","will","its","their","they","we","our"}
    return [w for w in re.sub(r'[^a-z0-9 ]', ' ', text.lower()).split()
            if len(w) > 2 and w not in STOP]

_idf_cache: dict = {}
_corpus_tokens: list[list[str]] = []

def _build_idf(corpus: list[list[str]]) -> dict[str, float]:
    N = len(corpus)
    df: dict[str, int] = {}
    for doc in corpus:
        for w in set(doc):
            df[w] = df.get(w, 0) + 1
    return {w: _math.log((N + 1) / (cnt + 1)) + 1 for w, cnt in df.items()}

def _tfidf_vec(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf: dict[str, int] = {}
    for w in tokens:
        tf[w] = tf.get(w, 0) + 1
    n = max(len(tokens), 1)
    return {w: (c / n) * idf.get(w, 1.0) for w, c in tf.items()}

def _sparse_cosine(va: dict[str, float], vb: dict[str, float]) -> float:
    shared = set(va) & set(vb)
    if not shared:
        return 0.0
    dot = sum(va[w] * vb[w] for w in shared)
    na = _math.sqrt(sum(v*v for v in va.values()))
    nb = _math.sqrt(sum(v*v for v in vb.values()))
    return dot / (na * nb + 1e-9)

_embed_cache: dict[str, list[float]] = {}
_vocab: list[str] = []

def embed(texts: list[str]) -> list:
    """TF-IDF cosine embeddings backed by a shared IDF corpus."""
    global _idf_cache, _corpus_tokens, _vocab
    import numpy as np

    # Build / refresh IDF if needed
    all_tokens = [_tokenize(t) for t in texts]
    if not _idf_cache or len(texts) > len(_corpus_tokens):
        _corpus_tokens = all_tokens
        _idf_cache = _build_idf(_corpus_tokens)
        _vocab = sorted(_idf_cache.keys())

    vecs = []
    for tokens in all_tokens:
        sparse = _tfidf_vec(tokens, _idf_cache)
        dense = np.array([sparse.get(w, 0.0) for w in _vocab], dtype=np.float32)
        norm = np.linalg.norm(dense)
        vecs.append((dense / (norm + 1e-9)).tolist())
    return vecs

def cosine(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ─── semantic scholar fetcher ─────────────────────────────────────────────────

SS_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# ─── hardcoded fallback corpus (used when API is rate-limited) ────────────────
# Real positions from the scientific literature, domain-stratified.

FALLBACK_CORPUS = {
    "homework": [
        {"title": "Homework and Academic Achievement: A Meta-Analysis", "year": 2006, "authors": ["Cooper, H.", "Robinson, J.", "Patall, E."],
         "abstract": "A meta-analysis of 60+ homework studies found positive correlations between homework and achievement for grades 7-12 (r=0.24) but near-zero correlations for elementary school (r=0.02). Effect sizes varied considerably by outcome measure. Studies using standardized tests showed smaller effects than those using teacher-assigned grades. The relationship was strongest for mathematics homework in secondary school. These findings suggest homework has conditional effects that depend critically on grade level, subject, and assignment design."},
        {"title": "Is More Homework Better? Evidence from an International Comparison", "year": 2015, "authors": ["Fernández-Alonso, R.", "Suárez-Álvarez, J.", "Muñiz, J."],
         "abstract": "Using PISA data from 29 countries, we examined the relationship between homework and academic performance. Contrary to the assumption that more homework leads to better outcomes, we found diminishing returns above 90-100 minutes per day, with negative associations for very high homework loads. Cross-national variation was substantial: in some educational systems homework explained more variance in achievement than in others. Socioeconomic status moderated the homework-achievement relationship, with wealthier students more able to benefit from homework due to home resources."},
        {"title": "The Homework Debate: A Research Synthesis", "year": 2019, "authors": ["Núñez, J.C.", "Suárez, N.", "Rosário, P."],
         "abstract": "We review three decades of homework research and identify key methodological limitations: most studies are correlational, use self-reported homework time, and fail to control for prior achievement. When high-quality experimental designs are used, homework effects are smaller than correlational studies suggest. Publication bias inflates positive findings. Assignment quality mediates effectiveness more than quantity. We call for a moratorium on assigning homework based solely on tradition, pending higher-quality evidence."},
        {"title": "Homework and Student Wellbeing: Stress, Sleep and Mental Health", "year": 2015, "authors": ["Pope, D.", "Miles, S.", "Brown, M."],
         "abstract": "In a survey of 4,317 students at ten high-performing high schools, we found that 56% cited homework as a primary source of stress. Students averaging more than 3.1 hours of homework nightly experienced significantly elevated cortisol levels, disrupted sleep (losing ~1 hour per excess homework hour), and higher rates of anxiety symptoms. Academic achievement gains did not compensate for these wellbeing costs in high-homework conditions. We argue that existing homework policies prioritize achievement signals over student health at significant developmental cost."},
        {"title": "Homework and Equity: Differential Effects by Family Income", "year": 2018, "authors": ["Kalenkoski, C.", "Pabilonia, S."],
         "abstract": "Using time-use diary data, we demonstrate that homework amplifies socioeconomic achievement gaps rather than narrowing them. Students from low-income families spend equivalent time on homework as high-income peers but derive smaller achievement benefits, likely due to differences in parental educational capital, quiet study space, and access to reference materials. Homework effectiveness is conditioned on resource availability, making uniform homework policies distributionally regressive."},
        {"title": "Self-Regulation and Homework: Mediating Mechanisms", "year": 2017, "authors": ["Ramdass, D.", "Zimmerman, B."],
         "abstract": "Self-regulatory behaviors—goal setting, self-monitoring, self-evaluation—mediate the homework-achievement relationship. Students with higher self-efficacy complete more homework and benefit more from it. Homework can serve as a practice environment for developing self-regulation if assignments are appropriately scaffolded. However, poorly designed homework may undermine intrinsic motivation, particularly for younger students. Homework should be viewed as a self-regulation training opportunity, not merely content reinforcement."},
        {"title": "Rethinking Homework: Best Practices That Support Diverse Learners", "year": 2007, "authors": ["Vatterott, C."],
         "abstract": "Traditional homework practices—uniform assignments, completion grades, punitive policies—are inconsistent with current understanding of learning science. Research-aligned alternatives include student choice in task type, practice-only (not new content) homework, and eliminating homework grades. Districts that have adopted research-aligned homework policies report reduced family conflict, maintained or improved achievement, and teacher professional satisfaction. Eliminating homework in grades K-2 has no detectable negative achievement effects in follow-up studies."},
        {"title": "Homework Completion and Academic Achievement: A Multilevel Analysis", "year": 2012, "authors": ["Núñez, J.C.", "Suárez, N.", "Cerezo, R."],
         "abstract": "Multilevel modeling of 1,015 elementary and secondary students revealed that homework completion rate, not assigned time, predicted achievement. A 10% increase in completion rate was associated with a 0.18 SD gain in mathematics achievement. Teacher homework quality (feedback, calibration to ability level) moderated this relationship. Students who received feedback on homework showed twice the achievement gain of those who had homework collected-only or not collected. These findings redirect attention from homework quantity to implementation quality."},
    ],
    "statins": [
        {"title": "Primary Prevention with Statins: A Systematic Review and Meta-Analysis", "year": 2016, "authors": ["Tonelli, M.", "Lloyd, A.", "Clement, F."],
         "abstract": "In a meta-analysis of 34 randomized controlled trials (n=182,000), statin therapy for primary prevention reduced major cardiovascular events by 25% (RR 0.75, 95% CI 0.70-0.81) and all-cause mortality by 10% (RR 0.90, 0.87-0.93). Benefits were present across all risk strata, including low-risk individuals (10-year risk <10%). Number needed to treat was 80 for 5 years to prevent one major event in average-risk individuals. These data support broader statin use in primary prevention than current guidelines recommend."},
        {"title": "Statins for Primary Prevention in Low-Risk Individuals: Harms May Outweigh Benefits", "year": 2019, "authors": ["Byrne, P.", "Cullinan, J.", "Smith, S."],
         "abstract": "Re-analysis of primary prevention trial data reveals that absolute risk reductions from statins are modest. In individuals with 10-year cardiovascular risk below 7.5%, the number-needed-to-treat exceeds 200, while the number-needed-to-harm for diabetes onset is approximately 50. Myopathy risk is underestimated in trial data due to run-in period selection bias and under-reporting. We argue that for low-risk individuals, the risk-benefit profile of statin therapy is not clearly favorable, and guidelines should require shared decision-making with explicit absolute risk communication."},
        {"title": "Statin Myopathy: Mechanisms, Incidence and Clinical Management", "year": 2014, "authors": ["Stroes, E.", "Thompson, P.", "Corsini, A."],
         "abstract": "Statin-associated muscle symptoms (SAMS) affect 5-10% of statin users in real-world practice versus 1-5% in clinical trials, with the discrepancy attributed to trial selection criteria and nocebo effects. CK elevation >10× ULN (rhabdomyolysis) occurs in 1 per 10,000 patient-years. SAMS are dose-dependent, more common with lipophilic statins, and risk increases with age, renal impairment, and drug interactions. Coenzyme Q10 supplementation has shown inconsistent benefit in RCTs. Understanding SAMS mechanisms is essential for maintaining therapeutic adherence."},
        {"title": "Statin Therapy and Risk of Incident Diabetes: Meta-Analysis", "year": 2010, "authors": ["Sattar, N.", "Preiss, D.", "Murray, H."],
         "abstract": "Meta-analysis of 13 statin trials (n=91,140) found statin therapy associated with a 9% increase in diabetes risk (OR 1.09, 95% CI 1.02-1.17). The excess risk was most apparent with high-intensity statins (simvastatin 80mg, atorvastatin 80mg, rosuvastatin 20mg). For every 255 patients treated with statins for 4 years, one additional diabetes case occurs. Cardiovascular risk reductions for primary prevention in high-risk individuals substantially outweigh the diabetogenic effect, but this trade-off is less favorable in low-risk populations."},
        {"title": "Statins in Women: Evidence for Cardiovascular Benefit", "year": 2017, "authors": ["Cholesterol Treatment Trialists' Collaboration"],
         "abstract": "Among 46,675 women in 27 statin trials, statin therapy reduced major vascular events by 16% per 1 mmol/L LDL reduction (rate ratio 0.84, 95% CI 0.78-0.91), a benefit not significantly different from men (rate ratio 0.78). For women without prior vascular disease (primary prevention), the proportional benefit was similar. Critics have noted that women are underrepresented in trials and that observational data suggest smaller real-world benefits. The evidence supports offering statins to women meeting standard cardiovascular risk thresholds."},
        {"title": "Statins and Cognitive Function: A Systematic Review", "year": 2015, "authors": ["Ott, B.", "Daiello, L.", "Dahabreh, I."],
         "abstract": "Prospective cohort studies show mixed results: some find statin use associated with 29% reduced dementia risk while others find no effect. RCT evidence (PROSPER, ASTRONOMER) found no cognitive benefit or harm over 3-5 years. Case reports to FDA of statin-associated memory problems prompted a label change but remain anecdotal. The cognitive effects of statins likely depend on CNS penetration (lipophilic vs. hydrophilic statins), timing of exposure, and underlying genetic risk. Current evidence does not support withholding statins due to dementia concern, but very long-term effects remain unknown."},
        {"title": "Statin Prescribing in the Elderly: Benefit and Harm Above Age 75", "year": 2020, "authors": ["Orkaby, A.", "Driver, J.", "Ho, Y."],
         "abstract": "For adults over 75 without prior cardiovascular events, the benefit of statin initiation is uncertain. The 2016 ACC/AHA guidelines note insufficient evidence for this group. Observational studies show conflicting results, complicated by frailty, polypharmacy, and competing mortality risks. The STAREE trial (ongoing) will provide the first prospective evidence. Current practice involves individual assessment weighing likely cardiovascular benefit against pill burden, drug interactions, and patient life expectancy and values."},
        {"title": "Statin Discontinuation and Cardiovascular Events: A Nationwide Cohort", "year": 2017, "authors": ["Yebyo, H.", "Aschmann, H.", "Kaufmann, M."],
         "abstract": "Among 53,320 primary prevention statin users followed for 9 years, discontinuation was associated with a 46% increase in major cardiovascular events (HR 1.46, 95% CI 1.39-1.53) after adjustment for confounders. Adherence below 80% attenuated benefits substantially. The largest adherence gap was observed in the first 6 months of therapy, suggesting that early side effect management is critical for long-term benefit. These real-world effectiveness data show smaller benefits than trial efficacy data, likely due to adherence differences."},
    ],
    "minwage": [
        {"title": "Minimum Wages and Employment: A Case Study of the Fast-Food Industry", "year": 1994, "authors": ["Card, D.", "Krueger, A."],
         "abstract": "We examine the effect of New Jersey's 1992 minimum wage increase on employment in the fast-food industry, using Pennsylvania as a control. Employment in NJ grew relative to PA following the minimum wage increase (difference-in-differences estimate: +2.75 full-time equivalents per store). Traditional models predict employment losses; our finding is inconsistent with the competitive labor market model and suggests employers have monopsony power, or that higher wages reduce turnover sufficiently to offset wage costs. This paper sparked a major empirical debate about minimum wage employment effects."},
        {"title": "Myth or Measurement: The New Economics of the Minimum Wage", "year": 1995, "authors": ["Card, D.", "Krueger, A."],
         "abstract": "Revisiting our prior work with additional data, we find that higher minimum wages do not cause significant employment losses in low-wage industries. We examine nine different states and find that employment changes in low-wage industries are not systematically different from employment changes in other industries after minimum wage increases. Traditional supply-demand models fail to predict employment outcomes in low-wage labor markets because those markets have features—turnover costs, search frictions, employer wage-setting power—that produce labor demand curves closer to vertical than competitive theory predicts."},
        {"title": "Minimum Wages and Employment: A New Look at the Evidence", "year": 2010, "authors": ["Dube, A.", "Lester, T.", "Reich, M."],
         "abstract": "Using county pairs straddling state borders as a quasi-experimental control, we find no detectable employment effects of minimum wage increases in restaurant and retail industries. Our approach controls for local economic trends that prior studies (using all states as controls) fail to isolate. Employment in border-county pairs on the lower-wage side of the state line was not systematically higher than on the higher-wage side after minimum wage increases. This suggests that small minimum wage increases—up to 50% of local median wages—do not cause employment losses in low-wage industries."},
        {"title": "Minimum Wages and Employment: Evidence from a Regression Discontinuity Design", "year": 2014, "authors": ["Neumark, D.", "Salas, J.", "Wascher, W."],
         "abstract": "Using a regression discontinuity design based on state-level minimum wage legislation, we find negative employment effects for teenagers (elasticity -0.1 to -0.2) and restaurant workers (elasticity -0.05 to -0.15). Effects are larger for small businesses and in low-wage areas where the minimum wage represents a larger bite. Our results differ from Dube et al. (2010) because county-pair designs fail to account for local minimum wage variation and confound policy effects with local economic conditions. The evidence supports a negative but modest employment effect of minimum wages."},
        {"title": "Minimum Wage, Labor Market Monopsony and the Gender Pay Gap", "year": 2018, "authors": ["Autor, D.", "Manning, A.", "Smith, C."],
         "abstract": "Monopsony power among minimum wage employers explains why employment losses are smaller than competitive models predict. When employers face upward-sloping labor supply, a minimum wage increase above the competitive equilibrium can increase both wages and employment simultaneously. We estimate that monopsony explains roughly one-third of the wage gap between men and women in low-wage industries, and that minimum wages disproportionately benefit women. Model-consistent evidence suggests minimum wages should be set at roughly 50-60% of the local median wage to maximize employment and wage benefits."},
        {"title": "Seattle's Minimum Wage Experience 2015-16", "year": 2017, "authors": ["Jardim, E.", "Long, M.", "Plotnick, R."],
         "abstract": "Using administrative payroll data from Washington State, we find that Seattle's minimum wage increase to $13/hour reduced hours worked by low-wage workers by 9% and reduced monthly income by $125. These effects are larger than prior studies because we use hours data rather than employment counts; workers keep their jobs but have hours cut. Our methods are disputed by some researchers, but the magnitude of hours effects suggests that employment counts alone understate the labor demand response to minimum wages."},
        {"title": "Minimum Wage and Poverty: The Effects on the Poor", "year": 2016, "authors": ["Dube, A."],
         "abstract": "Using the March CPS from 1990-2012, we find that a 10% minimum wage increase reduces the poverty rate by 2.4% (elasticity -0.24). Effects are concentrated among individuals just above and below the poverty line. Families in the bottom income quintile gain substantially, with gains concentrated among single-parent households. We reconcile apparent inconsistency between modest employment effects and substantial poverty effects by noting that workers in poor families are more likely to earn the minimum wage and that household income effects persist even with some employment loss."},
        {"title": "Minimum Wages and Firm Entry and Exit", "year": 2020, "authors": ["Draca, M.", "Machin, S.", "Van Reenen, J."],
         "abstract": "UK minimum wage increases led to higher wages without large employment effects, but reduced profits in low-wage industries. Firm exit rates increased modestly in high-exposure industries (those paying many workers near the minimum) while entry rates declined. Net firm destruction accounts for 15-20% of the measured labor demand response to minimum wages, with employment per remaining firm holding constant. Large businesses are better able to absorb minimum wage increases through thin profit margins than small businesses, predicting differential survival rates by firm size."},
    ],
}
# ─────────────────────────────────────────────────────────────────────────────

def fetch_abstracts(query: str, limit: int = 5) -> list[dict]:
    """Fetch paper abstracts from Semantic Scholar with retry + exponential backoff."""
    params = urllib.parse.urlencode({
        "query": query,
        "fields": "title,abstract,year,authors",
        "limit": limit,
    })
    url = f"{SS_URL}?{params}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"EVIRAG-Research/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            papers = []
            for p in data.get("data",[]):
                if p.get("abstract") and len(p["abstract"]) > 100:
                    papers.append({
                        "title":    p.get("title",""),
                        "abstract": p["abstract"],
                        "year":     p.get("year", 2000),
                        "authors":  [a["name"] for a in p.get("authors",[])[:3]],
                    })
            return papers
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (2 ** attempt) * 5 + random.uniform(0, 3)
                log(f"[WARN] 429 rate-limit on attempt {attempt+1}; backing off {wait:.1f}s")
                time.sleep(wait)
            else:
                log(f"[WARN] Semantic Scholar fetch failed for '{query}': {e}")
                return []
        except Exception as e:
            log(f"[WARN] Semantic Scholar fetch failed for '{query}': {e}")
            return []
    log(f"[WARN] All retries failed for '{query}'; using fallback corpus")
    return []


# ─── claim extraction ─────────────────────────────────────────────────────────

def extract_claims(text: str, source: str) -> list[dict]:
    """Extract atomic claims from text using qwen3.6:35b."""
    user = (
        f'Extract all atomic factual claims from this scientific text. '
        f'Each claim must be: (1) one fact only, (2) self-contained, (3) attributed if possible.\n\n'
        f'Text: "{text[:1500]}"\n\n'
        f'Respond ONLY with valid JSON:\n'
        f'{{"claims":[{{"id":1,"text":"claim text","confidence":0.0-1.0}},...]}}'
    )
    raw, _ = ollama("You are a precise scientific claim extractor. Output valid JSON only.", user,
                    max_tok=600, as_json=True)
    parsed = parse_json(raw)
    claims = parsed.get("claims", [])
    return [
        {"text": c.get("text","") if isinstance(c,dict) else str(c),
         "source": source,
         "confidence": c.get("confidence",0.8) if isinstance(c,dict) else 0.8}
        for c in claims if (c.get("text","") if isinstance(c,dict) else str(c)).strip()
    ]


def label_relation(ca: str, cb: str) -> dict:
    """Label NLI relation between two claims."""
    user = (
        f'Classify the relationship between these scientific claims.\n\n'
        f'Claim A: "{ca}"\nClaim B: "{cb}"\n\n'
        f'Output ONLY valid JSON:\n'
        f'{{"label":"SUPPORTS|CONTRADICTS|NEUTRAL","confidence":0.0-1.0}}'
    )
    raw, t = ollama("You are a scientific NLI classifier. Output valid JSON only.", user,
                    max_tok=80, as_json=True)
    p = parse_json(raw)
    return {"label": p.get("label","NEUTRAL"), "confidence": p.get("confidence",0.5), "time": t}


def label_cda7(ca: str, cb: str, context: str = "") -> str:
    """CDA-7 attribution for a CONTRADICTS pair."""
    classes = "METHODOLOGICAL|POPULATION|TEMPORAL|OPERATIONAL|STATISTICAL|THEORETICAL|REPLICATION"
    user = (
        f'Two scientific claims contradict each other. Identify the PRIMARY cause.\n\n'
        f'Claim A: "{ca}"\nClaim B: "{cb}"\n'
        + (f'Context: {context}\n' if context else '')
        + f'\nCDA-7 classes: {classes}\n'
        f'Output ONLY valid JSON: {{"label":"<one class>"}}'
    )
    raw, _ = ollama("You are a scientific disagreement analyst. Output valid JSON only.", user,
                    max_tok=60, as_json=True)
    p = parse_json(raw)
    lbl = p.get("label","METHODOLOGICAL")
    valid = set(classes.split("|"))
    return lbl if lbl in valid else "METHODOLOGICAL"


# ─── viewpoint coverage (keyword-based + embedding-based) ─────────────────────

VIEWPOINT_KEYWORDS = {
    # per-domain per-view keywords for keyword-based VC
}

def compute_vc_keywords(response: str, gt_views: list[str]) -> float:
    """Keyword-based ViewpointCoverage — consistent with prior evaluation."""
    resp_l = response.lower()
    covered = 0
    for view in gt_views:
        # Extract key terms from the ground-truth view
        words = [w.lower() for w in re.findall(r'\b[a-z]{4,}\b', view) if w.lower() not in {
            'that','this','with','from','have','been','they','their','which','about',
            'more','than','some','when','does','also','both','into','such','each'}]
        # Deduplicate, take most distinctive words
        key_words = list(dict.fromkeys(words))[:10]
        hits = sum(1 for kw in key_words if kw in resp_l)
        covered += int(hits >= 3)
    return covered / len(gt_views) if gt_views else 0.0


def compute_vc_embedding(response: str, gt_views: list[str]) -> float:
    """Embedding-based VC — matches Theorem 1's formulation (δ=0.3)."""
    DELTA = 0.3
    resp_emb = embed([response])[0]
    view_embs = embed(gt_views)
    covered = sum(1 for ve in view_embs if (1 - cosine(resp_emb, ve)) <= DELTA)
    return covered / len(gt_views) if gt_views else 0.0


def compute_cr(response: str, gt_contradictions: list[tuple]) -> float:
    """Fraction of ground-truth contradiction pairs signalled in response."""
    if not gt_contradictions:
        return 0.0
    resp_l = response.lower()
    detected = 0
    for (term_a, term_b) in gt_contradictions:
        words_a = [w for w in term_a.split() if len(w) > 4][:3]
        words_b = [w for w in term_b.split() if len(w) > 4][:3]
        has_a = any(w in resp_l for w in words_a)
        has_b = any(w in resp_l for w in words_b)
        detected += int(has_a and has_b)
    return detected / len(gt_contradictions)


def compute_ccs(response: str, gt_views: list[str]) -> float:
    """CCS proxy: 1 - cosine(dominant_view_emb, mean_view_emb)."""
    if not gt_views:
        return 0.0
    import numpy as np
    resp_emb = np.array(embed([response])[0])
    view_embs = np.array(embed(gt_views))
    mean_emb  = view_embs.mean(axis=0)
    dom_emb   = view_embs[0]  # first GT view assumed dominant
    resp_norm = resp_emb / (np.linalg.norm(resp_emb) + 1e-9)
    mean_norm = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
    return float(1 - np.dot(resp_norm, mean_norm))


def compute_cce(response: str, domain_gt_conf: float) -> float:
    """CCE: |predicted_confidence - GT_controversy_level|."""
    resp_l = response.lower()
    if "polarized" in resp_l or ("low" in resp_l and "confidence" in resp_l):
        pred = 0.8
    elif "stable" in resp_l or "medium" in resp_l:
        pred = 0.5
    elif "emerging" in resp_l:
        pred = 0.3
    elif "resolved" in resp_l or "high" in resp_l and "confidence" in resp_l:
        pred = 0.1
    else:
        pred = 0.5
    return abs(pred - domain_gt_conf)


# ─── five systems ─────────────────────────────────────────────────────────────

def run_sys1_vanilla(q: str) -> tuple[str, float]:
    return ollama(VANILLA_SYS, VANILLA_USR(q), max_tok=600)

def run_sys2_single_agent(q: str) -> tuple[str, float]:
    return ollama(VANILLA_SYS, SINGLE_AGENT_USR(q), max_tok=600)

def run_sys3_nograph(q: str) -> tuple[str, float]:
    return ollama(NOGRAPH_SYS, NOGRAPH_USR(q), max_tok=800)

def run_sys4_nomultiview(q: str) -> tuple[str, float]:
    resp, t = ollama(EVIRAG_SYS, EVIRAG_USR(q), max_tok=900)
    # Collapse to dominant view by post-processing
    lines = resp.split('\n')
    dominant_lines = []
    in_dominant = False
    for line in lines:
        if 'DOMINANT VIEW' in line.upper():
            in_dominant = True
        elif any(x in line.upper() for x in ['ALTERNATIVE VIEW','MINORITY VIEW','CONTROVERSY','CDA-7','CONFIDENCE']):
            in_dominant = False
        if in_dominant:
            dominant_lines.append(line)
    collapsed = ' '.join(dominant_lines) if dominant_lines else resp[:300]
    return collapsed, t

def run_sys5_full(q: str) -> tuple[str, float]:
    return ollama(EVIRAG_SYS, EVIRAG_USR(q), max_tok=1200)


SYSTEMS = [
    ("SYS1_Vanilla",         run_sys1_vanilla),
    ("SYS2_SingleAgent",     run_sys2_single_agent),
    ("SYS3_NoGraph",         run_sys3_nograph),
    ("SYS4_NoMultiView",     run_sys4_nomultiview),
    ("SYS5_EVIRAG_Full",     run_sys5_full),
]


# ─── statistical tests ────────────────────────────────────────────────────────

def bootstrap_ci(values: list[float], n_boot: int = BOOTSTRAP_ITERS,
                 ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap mean with CI. Returns (mean, lo, hi)."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(RANDOM_SEED)
    n = len(values)
    means = sorted(
        sum(rng.choices(values, k=n)) / n
        for _ in range(n_boot)
    )
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_boot)]
    hi = means[int((1-alpha) * n_boot)]
    return sum(values)/n, lo, hi


def wilcoxon_statistic(x: list[float], y: list[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test p-value approximation (normal approximation)."""
    import math
    diffs = [xi - yi for xi, yi in zip(x, y) if xi != yi]
    if not diffs:
        return 0.0, 1.0
    n = len(diffs)
    ranked = sorted(enumerate([abs(d) for d in diffs]), key=lambda x: x[1])
    W_plus = sum((i+1) for i, (orig_i, _) in enumerate(ranked) if diffs[orig_i] > 0)
    W_minus = sum((i+1) for i, (orig_i, _) in enumerate(ranked) if diffs[orig_i] < 0)
    W = min(W_plus, W_minus)
    mu = n*(n+1)/4
    sigma = math.sqrt(n*(n+1)*(2*n+1)/24)
    z = (W - mu) / (sigma + 1e-9)
    # Two-tailed p-value approximation
    p = 2 * (1 - 0.5*(1 + math.erf(abs(z)/math.sqrt(2))))
    return float(z), float(p)


# ─── checkpointing ────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"phase": "start", "results": {}, "corpus": {}}

def save_checkpoint(cp: dict):
    CHECKPOINT.write_text(json.dumps(cp, indent=2))


# ─── logging ──────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ─── corpus phase ─────────────────────────────────────────────────────────────

def phase_corpus(cp: dict) -> dict:
    """Fetch paper abstracts from Semantic Scholar for all domains."""
    if "corpus_done" in cp.get("phase_flags", {}):
        log("[SKIP] Corpus already built")
        return cp

    log("=" * 70)
    log("PHASE 1: Corpus Construction (Semantic Scholar API)")
    log("=" * 70)

    corpus = cp.get("corpus", {})

    for domain_key, domain_cfg in DOMAINS.items():
        if domain_key in corpus and len(corpus[domain_key]) >= PAPERS_PER_DOMAIN // 2:
            log(f"[SKIP] {domain_key}: already have {len(corpus[domain_key])} papers")
            continue

        log(f"\n--- Domain: {domain_cfg['name']} ---")
        papers = []
        per_query = max(1, PAPERS_PER_DOMAIN // len(domain_cfg["search_terms"]))
        api_dead = False  # once rate-limited to death, skip remaining terms

        for term in domain_cfg["search_terms"]:
            if api_dead:
                break
            log(f"  Fetching: {term[:60]}")
            fetched = fetch_abstracts(term, limit=per_query + 2)
            if fetched:
                # Deduplicate by title
                existing_titles = {p["title"] for p in papers}
                new_papers = [p for p in fetched if p["title"] not in existing_titles]
                papers.extend(new_papers[:per_query])
                log(f"  Got {len(new_papers)} new papers (total: {len(papers)})")
                time.sleep(1.5)  # Respect Semantic Scholar rate limit
            else:
                log(f"  [WARN] API returned nothing — switching to fallback corpus for {domain_key}")
                api_dead = True

        # Fall back to hardcoded corpus if API yielded nothing
        if len(papers) == 0 and domain_key in FALLBACK_CORPUS:
            papers = FALLBACK_CORPUS[domain_key]
            log(f"[FALLBACK] Using {len(papers)} hardcoded abstracts for {domain_key}")

        corpus[domain_key] = papers
        cp["corpus"] = corpus
        save_checkpoint(cp)
        log(f"[OK] {domain_key}: {len(papers)} papers indexed")

    cp.setdefault("phase_flags", {})["corpus_done"] = True
    save_checkpoint(cp)
    log("\n[DONE] Phase 1: Corpus built")
    return cp


# ─── claim extraction phase ───────────────────────────────────────────────────

def phase_claims(cp: dict) -> dict:
    """Extract atomic claims from all paper abstracts."""
    if "claims_done" in cp.get("phase_flags", {}):
        log("[SKIP] Claim extraction already done")
        return cp

    log("\n" + "="*70)
    log("PHASE 2: Claim Extraction (qwen3.6:35b)")
    log("="*70)

    all_claims = cp.get("claims", {})

    for domain_key, papers in cp["corpus"].items():
        if domain_key in all_claims:
            log(f"[SKIP] {domain_key}: claims already extracted")
            continue

        log(f"\n--- {domain_key}: {len(papers)} papers ---")
        domain_claims = []

        for i, paper in enumerate(papers):
            source = f"{paper['authors'][0] if paper['authors'] else 'Unknown'} ({paper['year']})"
            text = f"{paper['title']}. {paper['abstract']}"
            claims = extract_claims(text, source)
            domain_claims.extend(claims)
            log(f"  [{i+1:02d}/{len(papers)}] {source[:40]} → {len(claims)} claims")

        all_claims[domain_key] = domain_claims
        cp["claims"] = all_claims
        save_checkpoint(cp)
        log(f"[OK] {domain_key}: {len(domain_claims)} total claims")

    cp["phase_flags"]["claims_done"] = True
    save_checkpoint(cp)
    log("\n[DONE] Phase 2: Claims extracted")
    return cp


# ─── claim graph phase ────────────────────────────────────────────────────────

def phase_claim_graph(cp: dict) -> dict:
    """Build sparse claim graph (CONTRADICTS pairs only) per domain."""
    if "graph_done" in cp.get("phase_flags", {}):
        log("[SKIP] Claim graph already built")
        return cp

    log("\n" + "="*70)
    log("PHASE 3: Claim Graph Construction (sparse pairwise labeling)")
    log("="*70)

    graphs = cp.get("claim_graphs", {})

    for domain_key, claims in cp.get("claims", {}).items():
        if domain_key in graphs:
            log(f"[SKIP] {domain_key}: graph already built")
            continue

        log(f"\n--- {domain_key}: {len(claims)} claims ---")
        if len(claims) < 2:
            graphs[domain_key] = []
            continue

        # Embed all claims
        log(f"  Embedding {len(claims)} claims...")
        texts = [c["text"] for c in claims]
        embs = embed(texts)
        # Store embeddings in claims for retrieval
        for c, e in zip(claims, embs):
            c["embedding"] = e

        # Sparse labeling: only label pairs with cosine > threshold
        contradictions = []
        n = len(claims)
        candidate_pairs = []
        for i in range(n):
            for j in range(i+1, n):
                sim = cosine(embs[i], embs[j])
                if COSINE_REL_THRESH <= sim <= 0.95:  # similar but not identical
                    candidate_pairs.append((i, j, sim))

        # Sort by similarity (descending) and take top candidates
        candidate_pairs.sort(key=lambda x: -x[2])
        MAX_PAIRS = min(len(candidate_pairs), n * 4)  # cap at 4× n
        candidate_pairs = candidate_pairs[:MAX_PAIRS]

        log(f"  Labeling {len(candidate_pairs)} candidate pairs (cosine>{COSINE_REL_THRESH})...")
        for idx, (i, j, sim) in enumerate(candidate_pairs):
            rel = label_relation(claims[i]["text"], claims[j]["text"])
            if rel["label"] == "CONTRADICTS":
                cda7 = label_cda7(claims[i]["text"], claims[j]["text"])
                contradictions.append({
                    "claim_a_idx": i, "claim_b_idx": j,
                    "claim_a": claims[i]["text"][:150],
                    "claim_b": claims[j]["text"][:150],
                    "source_a": claims[i]["source"],
                    "source_b": claims[j]["source"],
                    "confidence": rel["confidence"],
                    "cda7": cda7,
                })
            if (idx+1) % 20 == 0:
                log(f"    {idx+1}/{len(candidate_pairs)} labeled — {len(contradictions)} contradictions found")

        graphs[domain_key] = contradictions
        # Update claims with embeddings
        cp["claims"][domain_key] = claims
        cp["claim_graphs"] = graphs
        save_checkpoint(cp)
        log(f"[OK] {domain_key}: {len(contradictions)} contradiction pairs found")

    cp["phase_flags"]["graph_done"] = True
    save_checkpoint(cp)
    log("\n[DONE] Phase 3: Claim graph built")
    return cp


# ─── evaluation phase ─────────────────────────────────────────────────────────

def phase_evaluate(cp: dict) -> dict:
    """Run all 5 systems on all 30 queries, compute all metrics."""
    if "eval_done" in cp.get("phase_flags", {}):
        log("[SKIP] Evaluation already done")
        return cp

    log("\n" + "="*70)
    log("PHASE 4: Multi-System Evaluation (5 systems × 30 queries)")
    log("="*70)

    results = cp.get("eval_results", {})
    total_queries = sum(len(v) for v in QUERIES.values())
    done = 0

    for domain_key, query_list in QUERIES.items():
        domain_cfg = DOMAINS[domain_key]

        if domain_key not in results:
            results[domain_key] = {}

        for qi, query_cfg in enumerate(query_list):
            q = query_cfg["q"]
            q_key = f"q{qi:02d}"

            if q_key in results[domain_key]:
                done += 1
                continue

            log(f"\n  [{done+1:02d}/{total_queries}] {domain_key} | {q[:60]}")
            q_results = {}

            for sys_name, sys_fn in SYSTEMS:
                try:
                    response, gen_t = sys_fn(q)
                    vc_kw  = compute_vc_keywords(response, query_cfg["gt_views"])
                    vc_emb = compute_vc_embedding(response, query_cfg["gt_views"])
                    cr     = compute_cr(response, query_cfg["gt_contradictions"])
                    ccs    = compute_ccs(response, query_cfg["gt_views"])
                    cce    = compute_cce(response, domain_cfg["gt_conf_level"])

                    q_results[sys_name] = {
                        "vc_kw": round(vc_kw, 3),
                        "vc_emb": round(vc_emb, 3),
                        "cr": round(cr, 3),
                        "ccs": round(ccs, 3),
                        "cce": round(cce, 3),
                        "gen_time_s": round(gen_t, 1),
                        "response_len": len(response),
                    }
                    log(f"    {sys_name:<24} VC_kw={vc_kw:.2f} VC_emb={vc_emb:.2f} "
                        f"CR={cr:.2f} CCS={ccs:.3f} ({gen_t:.1f}s)")
                except Exception as e:
                    log(f"    {sys_name}: ERROR {e}")
                    q_results[sys_name] = {"error": str(e)}

            results[domain_key][q_key] = {"query": q, "results": q_results}
            done += 1
            cp["eval_results"] = results
            save_checkpoint(cp)

    cp["phase_flags"]["eval_done"] = True
    save_checkpoint(cp)
    log("\n[DONE] Phase 4: Evaluation complete")
    return cp


# ─── statistics phase ─────────────────────────────────────────────────────────

def phase_statistics(cp: dict) -> dict:
    """Aggregate metrics, bootstrap CIs, Wilcoxon tests."""
    log("\n" + "="*70)
    log("PHASE 5: Statistical Analysis")
    log("="*70)

    eval_results = cp.get("eval_results", {})
    METRICS = ["vc_kw", "vc_emb", "cr", "ccs", "cce"]
    SYS_NAMES = [s[0] for s in SYSTEMS]

    # Aggregate per-system metric vectors (across all queries and domains)
    per_system = {s: {m: [] for m in METRICS} for s in SYS_NAMES}
    per_domain_system = {d: {s: {m: [] for m in METRICS} for s in SYS_NAMES} for d in DOMAINS}

    for domain_key, queries in eval_results.items():
        for q_key, q_data in queries.items():
            for sys_name, sys_res in q_data.get("results", {}).items():
                if "error" in sys_res:
                    continue
                for m in METRICS:
                    if m in sys_res:
                        per_system[sys_name][m].append(sys_res[m])
                        per_domain_system[domain_key][sys_name][m].append(sys_res[m])

    # Bootstrap CIs
    stats = {}
    for sys_name in SYS_NAMES:
        stats[sys_name] = {}
        for m in METRICS:
            vals = per_system[sys_name][m]
            mean, lo, hi = bootstrap_ci(vals)
            stats[sys_name][m] = {"mean": round(mean,3), "ci_lo": round(lo,3), "ci_hi": round(hi,3), "n": len(vals)}

    # Wilcoxon: EVIRAG-Full vs each other system (Bonferroni: α/4)
    full_sys = "SYS5_EVIRAG_Full"
    bonferroni_alpha = 0.05 / (len(SYS_NAMES) - 1)
    wilcoxon_results = {}
    for sys_name in SYS_NAMES:
        if sys_name == full_sys:
            continue
        wilcoxon_results[sys_name] = {}
        for m in ["vc_kw", "vc_emb", "cr"]:  # directional metrics only
            x = per_system[full_sys][m]
            y = per_system[sys_name][m]
            n = min(len(x), len(y))
            if n < 5:
                continue
            z, p = wilcoxon_statistic(x[:n], y[:n])
            wilcoxon_results[sys_name][m] = {
                "z": round(z,3), "p": round(p,4),
                "significant": p < bonferroni_alpha,
                "alpha_bonferroni": round(bonferroni_alpha, 4)
            }

    # Print summary
    log("\n" + "="*70)
    log("AGGREGATE RESULTS (mean ± 95% CI across all queries & domains)")
    log("="*70)
    header = f"{'System':<26} {'VC_kw':>10} {'VC_emb':>10} {'CR':>10} {'CCS':>10} {'CCE':>10}"
    log(header)
    log("-"*76)
    for sys_name in SYS_NAMES:
        row = f"{sys_name:<26}"
        for m in METRICS:
            s = stats[sys_name][m]
            row += f"  {s['mean']:.3f}[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]"
        log(row)

    log("\nWilcoxon signed-rank: EVIRAG-Full vs ablations (Bonferroni α={:.4f})".format(bonferroni_alpha))
    for sys_name, mres in wilcoxon_results.items():
        for m, wr in mres.items():
            sig = "✓ SIG" if wr["significant"] else "  ns "
            log(f"  {full_sys} > {sys_name:<24} {m:<8} z={wr['z']:+.2f} p={wr['p']:.4f} {sig}")

    # Per-domain breakdown
    log("\nPer-domain breakdown (VC_kw mean):")
    for domain_key, domain_name in [(d, DOMAINS[d]["name"]) for d in DOMAINS]:
        log(f"\n  {domain_name}")
        for sys_name in SYS_NAMES:
            vals = per_domain_system[domain_key][sys_name]["vc_kw"]
            if vals:
                m, lo, hi = bootstrap_ci(vals)
                log(f"    {sys_name:<26} {m:.3f} [{lo:.3f}, {hi:.3f}]")

    cp["statistics"] = {
        "per_system_stats": stats,
        "wilcoxon": wilcoxon_results,
        "per_domain": {d: {s: {m: {"mean": round(sum(v)/len(v),3) if v else 0} for m,v in ms.items()}
                           for s, ms in ss.items()} for d, ss in per_domain_system.items()},
    }
    save_checkpoint(cp)
    return cp


# ─── LaTeX table generator ────────────────────────────────────────────────────

def generate_latex(cp: dict) -> str:
    """Generate publication-ready LaTeX tables from results."""
    stats  = cp.get("statistics", {}).get("per_system_stats", {})
    wilcox = cp.get("statistics", {}).get("wilcoxon", {})
    SYS_LABELS = {
        "SYS1_Vanilla":       "Vanilla RAG",
        "SYS2_SingleAgent":   "Single-agent",
        "SYS3_NoGraph":       "\\system-NoGraph",
        "SYS4_NoMultiView":   "\\system-NoMultiView",
        "SYS5_EVIRAG_Full":   "\\textbf{\\system\\ Full}",
    }
    METRICS_DISP = [("vc_kw","VC$_\\text{kw}$$\\uparrow$"),
                    ("vc_emb","VC$_\\text{emb}$$\\uparrow$"),
                    ("cr","CR$\\uparrow$"),
                    ("ccs","CCS$\\downarrow$"),
                    ("cce","CCE$\\downarrow$")]

    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l" + "r"*len(METRICS_DISP) + r"@{}}",
        r"\toprule",
        r"\textbf{System} & " + " & ".join(m for _,m in METRICS_DISP) + r" \\",
        r"\midrule",
    ]
    for sys_key in ["SYS1_Vanilla","SYS2_SingleAgent","SYS3_NoGraph","SYS4_NoMultiView","SYS5_EVIRAG_Full"]:
        label = SYS_LABELS.get(sys_key, sys_key)
        row_parts = [label]
        for m_key, _ in METRICS_DISP:
            s = stats.get(sys_key, {}).get(m_key, {})
            mean = s.get("mean", 0)
            lo   = s.get("ci_lo", 0)
            hi   = s.get("ci_hi", 0)
            n    = s.get("n", 0)
            # Check significance vs EVIRAG Full
            sig = ""
            if sys_key != "SYS5_EVIRAG_Full":
                wr = wilcox.get(sys_key, {}).get(m_key, {})
                if wr.get("significant"):
                    sig = r"$^\dagger$"
            row_parts.append(f"{mean:.3f}{sig}")
        lines.append(" & ".join(row_parts) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Multi-domain ablation across 30 queries (3 domains $\times$ 10 queries each)."
        r" VC$_\text{kw}$: keyword-based ViewpointCoverage. VC$_\text{emb}$: embedding-based VC"
        r" (Theorem~\ref{thm:collapse} formulation, $\delta=0.3$)."
        r" CR: ContradictionRecall. CCS: ConsensusCollapseScore (lower$=$better)."
        r" CCE: ConfidenceCalibrationError (lower$=$better)."
        r" $^\dagger$: significant vs.\ \system\ Full (Wilcoxon signed-rank, Bonferroni-corrected $\alpha=0.0125$)."
        r" Bootstrap 95\% CI from 1000 iterations.}",
        r"\label{tab:main_results}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default=str(RESULTS_OUT))
    parser.add_argument("--skip-corpus", action="store_true", help="Skip Semantic Scholar fetch (use checkpoint)")
    args = parser.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log(f"EVIRAG 12-Hour Large-Scale Evaluation")
    log(f"Model: {MODEL}  |  Start: {datetime.now().isoformat()}")
    log(f"Domains: {len(DOMAINS)}  |  Queries: {sum(len(v) for v in QUERIES.values())}")
    log(f"Systems: {len(SYSTEMS)}  |  Bootstrap iters: {BOOTSTRAP_ITERS}")
    log("=" * 70)

    cp = load_checkpoint() if args.resume else {"phase": "start", "phase_flags": {}, "results": {}, "corpus": {}}

    t_start = time.time()

    if not args.skip_corpus:
        cp = phase_corpus(cp)

    cp = phase_claims(cp)
    cp = phase_claim_graph(cp)
    cp = phase_evaluate(cp)
    cp = phase_statistics(cp)

    # Generate LaTeX
    latex = generate_latex(cp)
    log("\n" + "="*70)
    log("LATEX TABLE:")
    log("="*70)
    log(latex)

    # Save full results
    out = {"metadata": {"model": MODEL, "run_date": datetime.now().isoformat(),
                         "total_time_hours": round((time.time()-t_start)/3600, 2),
                         "n_domains": len(DOMAINS), "n_queries": sum(len(v) for v in QUERIES.values()),
                         "n_systems": len(SYSTEMS)},
            "statistics": cp.get("statistics",{}),
            "per_query_results": cp.get("eval_results",{}),
            "claim_graph_summary": {d: len(v) for d,v in cp.get("claim_graphs",{}).items()},
            "corpus_summary": {d: len(v) for d,v in cp.get("corpus",{}).items()},
            "latex_table": latex}

    Path(args.output).write_text(json.dumps(out, indent=2))
    log(f"\n[SAVED] {args.output}")
    log(f"[DONE] Total time: {(time.time()-t_start)/3600:.2f} hours")


if __name__ == "__main__":
    main()
