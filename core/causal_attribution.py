"""
EVIRAG: Causal Disagreement Attribution Module
================================================
Novel contribution: When two scientific papers disagree, WHY do they disagree?

Existing systems (RARR, FActScore, SciFact) detect THAT papers contradict each other.
None attribute the CAUSE of the disagreement. This module fills that gap.

Attribution Taxonomy (CDA-7):
    1. METHODOLOGICAL  — Different experimental designs, measurement tools, or protocols
    2. POPULATION      — Different study populations, domains, or application settings
    3. TEMPORAL        — Knowledge evolved; newer evidence supersedes older claims
    4. OPERATIONAL     — Different operationalizations/definitions of key terms
    5. STATISTICAL     — Conflicting statistical interpretations (p-values, effect sizes)
    6. THEORETICAL     — Competing theoretical frameworks or assumptions
    7. REPLICATION     — Failure to replicate; original finding was spurious

This is inspired by the causal inference literature (Pearl 2009) and the scientific
disagreement taxonomy from Ramirez et al. (2019), but applied for the first time
in a retrieval-augmented generation pipeline.

For each detected contradiction in the disagreement graph, we:
  1. Extract contextual signals from both conflicting claims + their source sections
  2. Use structured NLI prompting to assign attribution category + confidence
  3. Build a CausalAttribution map that explains the disagreement landscape

This attribution is then surfaced in the multi-view synthesis and used to improve
confidence calibration (methodological disagreements are more resolvable than
theoretical ones).
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import Counter

from epistemic_engine import OllamaClient, AtomicClaim, ClaimRelationship
from config import get_model_config


# ---------------------------------------------------------------------------
# Attribution taxonomy
# ---------------------------------------------------------------------------

ATTRIBUTION_TYPES = [
    "METHODOLOGICAL",
    "POPULATION",
    "TEMPORAL",
    "OPERATIONAL",
    "STATISTICAL",
    "THEORETICAL",
    "REPLICATION",
    "UNKNOWN",
]

# Resolvability prior: how likely is this type of disagreement to be resolved
# by future research? Used to modulate confidence calibration.
RESOLVABILITY_PRIOR = {
    "METHODOLOGICAL": 0.75,   # High — better methods can resolve
    "POPULATION":     0.70,   # High — broader studies can clarify scope
    "TEMPORAL":       0.85,   # Very high — time resolves most temporal conflicts
    "OPERATIONAL":    0.60,   # Medium — definitional fights are often persistent
    "STATISTICAL":    0.65,   # Medium — replication studies help
    "THEORETICAL":    0.30,   # Low — theoretical paradigms can persist for decades
    "REPLICATION":    0.50,   # Medium — depends on reproducibility efforts
    "UNKNOWN":        0.40,   # Uncertain
}

# Confidence penalty for each attribution type applied to overall answer confidence
CONFIDENCE_PENALTY = {
    "METHODOLOGICAL": 0.10,
    "POPULATION":     0.15,
    "TEMPORAL":       0.05,   # Low penalty — temporal disagreements are interpretable
    "OPERATIONAL":    0.20,
    "STATISTICAL":    0.15,
    "THEORETICAL":    0.25,   # High penalty — theoretical disagreements are deep
    "REPLICATION":    0.30,   # Highest — a replication failure is serious
    "UNKNOWN":        0.20,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DisagreementAttribution:
    """Attribution for a single pair of contradicting claims."""
    claim1_id: str
    claim2_id: str
    claim1_text: str
    claim2_text: str
    attribution_type: str        # One of ATTRIBUTION_TYPES
    confidence: float            # 0-1, how confident is the attribution
    evidence: str                # Textual evidence for the attribution
    resolvability: float         # 0-1, how likely to be resolved by future research
    confidence_penalty: float    # Applied to overall answer confidence

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CausalAttributionReport:
    """Full causal attribution analysis for a query's contradiction set."""
    query: str
    total_contradictions: int
    attributions: List[DisagreementAttribution]

    # Aggregate statistics
    attribution_distribution: Dict[str, int]   # type -> count
    dominant_cause: str                         # Most frequent attribution type
    mean_resolvability: float                   # Avg resolvability across all contradictions
    aggregate_confidence_penalty: float         # Total confidence adjustment
    controversy_depth: str                      # 'shallow' / 'moderate' / 'deep'

    # Research implications
    research_recommendations: List[str]

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "total_contradictions": self.total_contradictions,
            "attributions": [a.to_dict() for a in self.attributions],
            "attribution_distribution": self.attribution_distribution,
            "dominant_cause": self.dominant_cause,
            "mean_resolvability": self.mean_resolvability,
            "aggregate_confidence_penalty": self.aggregate_confidence_penalty,
            "controversy_depth": self.controversy_depth,
            "research_recommendations": self.research_recommendations,
        }


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CAUSAL_ATTRIBUTION_SYSTEM_PROMPT = """You are a scientific epistemologist specializing in
analyzing the CAUSES of disagreements between research papers.

Given two contradicting claims from different scientific papers, your task is to identify
the MOST LIKELY cause of their disagreement from this taxonomy:

1. METHODOLOGICAL  — They used different experimental designs, measurement instruments,
   data collection methods, or analytical protocols.
   Signal words: "our study used...", "unlike X who used Y...", "RCT vs observational"

2. POPULATION      — They studied different populations, domains, or application settings.
   Signal words: "in children vs adults", "cross-cultural", "in clinical vs community settings"

3. TEMPORAL        — The papers are from different time periods; knowledge evolved.
   Signal words: publication years differ significantly, "early studies showed... recent..."

4. OPERATIONAL     — They define the key concept differently.
   Signal words: different definitions of the core term, "what we mean by X...", varying thresholds

5. STATISTICAL     — Conflicting statistical interpretations.
   Signal words: "effect size d=...", "p-value threshold", "statistical vs practical significance"

6. THEORETICAL     — They operate within competing theoretical frameworks.
   Signal words: "from a cognitive perspective vs behavioral...", different theoretical models

7. REPLICATION     — One paper failed to replicate the other's findings.
   Signal words: "we could not replicate", "failed to reproduce", "inconsistent with X"

8. UNKNOWN         — Cannot determine the cause from available text.

Respond ONLY with valid JSON:
{
  "attribution_type": "<type from list above>",
  "confidence": <0.0-1.0>,
  "evidence": "<1-2 sentences citing specific signals in the claims that led to this classification>",
  "key_signal": "<the specific phrase or feature that most clearly indicates this type>"
}"""

CAUSAL_ATTRIBUTION_USER_PROMPT = """Claim A (from Paper 1):
"{claim1}"

Claim B (from Paper 2):
"{claim2}"

Context A: {context1}
Context B: {context2}

These claims CONTRADICT each other. What is the most likely CAUSE of this disagreement?"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CausalDisagreementAttributor:
    """
    Attributes the causes of scientific disagreements in the EVIRAG pipeline.

    This is a novel contribution: no existing RAG or scientific QA system
    explains WHY retrieved sources disagree, only THAT they disagree.

    Usage:
        attributor = CausalDisagreementAttributor(backend="local")
        report = attributor.attribute(
            query="Does homework improve academic performance?",
            claims=claim_list,
            relationships=relationship_list,
            claim_map=claim_map,
        )
    """

    def __init__(self, backend: str = "local"):
        self.backend = backend
        self.client = OllamaClient()
        # Use the most capable available model for attribution
        self.model_config = get_model_config("precision", "agent", backend)
        self.model = (
            self.model_config.name if self.model_config else "llama3:latest"
        )

    def attribute(
        self,
        query: str,
        claims: List[AtomicClaim],
        relationships: List[ClaimRelationship],
        claim_map: Optional[Dict[str, AtomicClaim]] = None,
        max_contradictions: int = 20,
    ) -> CausalAttributionReport:
        """
        Attribute causes to all detected contradictions in the claim graph.

        Args:
            query: The original user query
            claims: All extracted atomic claims
            relationships: All claim relationships (NLI output)
            claim_map: Dict mapping claim_id -> AtomicClaim (for fast lookup)
            max_contradictions: Maximum contradictions to attribute (cost control)

        Returns:
            CausalAttributionReport with full attribution analysis
        """
        if claim_map is None:
            claim_map = {c.claim_id: c for c in claims}

        # Filter to contradictory relationships only
        contradictions = [
            r for r in relationships
            if r.relationship in ("contradiction", "contradicts")
            and r.confidence >= 0.6
        ]

        # Sort by confidence descending, take top N
        contradictions.sort(key=lambda r: r.confidence, reverse=True)
        contradictions = contradictions[:max_contradictions]

        if not contradictions:
            return self._empty_report(query)

        # Attribute each contradiction
        attributions: List[DisagreementAttribution] = []
        for rel in contradictions:
            c1 = claim_map.get(rel.claim1_id)
            c2 = claim_map.get(rel.claim2_id)
            if c1 is None or c2 is None:
                continue
            attr = self._attribute_single(c1, c2, rel)
            if attr:
                attributions.append(attr)

        return self._build_report(query, attributions, len(contradictions))

    def _attribute_single(
        self,
        claim1: AtomicClaim,
        claim2: AtomicClaim,
        rel: ClaimRelationship,
    ) -> Optional[DisagreementAttribution]:
        """Attribute the cause of a single contradiction."""
        prompt = CAUSAL_ATTRIBUTION_USER_PROMPT.format(
            claim1=claim1.text,
            claim2=claim2.text,
            context1=claim1.context or "No additional context",
            context2=claim2.context or "No additional context",
        )

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                system=CAUSAL_ATTRIBUTION_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=512,
            )
            parsed = self._parse_json(response)
            if not parsed:
                return self._fallback_attribution(claim1, claim2, rel)

            attr_type = parsed.get("attribution_type", "UNKNOWN").upper()
            if attr_type not in ATTRIBUTION_TYPES:
                attr_type = "UNKNOWN"

            confidence = float(parsed.get("confidence", 0.5))
            evidence = parsed.get("evidence", rel.reasoning or "")

            return DisagreementAttribution(
                claim1_id=claim1.claim_id,
                claim2_id=claim2.claim_id,
                claim1_text=claim1.text,
                claim2_text=claim2.text,
                attribution_type=attr_type,
                confidence=round(confidence, 3),
                evidence=evidence,
                resolvability=RESOLVABILITY_PRIOR[attr_type],
                confidence_penalty=CONFIDENCE_PENALTY[attr_type],
            )

        except Exception as e:
            print(f"  CausalAttribution error for pair ({claim1.claim_id[:8]}, "
                  f"{claim2.claim_id[:8]}): {e}")
            return self._fallback_attribution(claim1, claim2, rel)

    def _fallback_attribution(
        self,
        claim1: AtomicClaim,
        claim2: AtomicClaim,
        rel: ClaimRelationship,
    ) -> DisagreementAttribution:
        """Rule-based fallback attribution using keyword signals."""
        combined = (claim1.text + " " + claim2.text).lower()

        # Keyword-based heuristics
        if any(w in combined for w in [
            "rct", "randomized", "trial", "observational", "cohort",
            "experimental", "control group", "survey", "longitudinal"
        ]):
            attr_type = "METHODOLOGICAL"
        elif any(w in combined for w in [
            "children", "adults", "patients", "population", "sample",
            "clinical", "community", "cross-cultural", "western"
        ]):
            attr_type = "POPULATION"
        elif any(w in combined for w in [
            "define", "definition", "what we mean", "operationalize",
            "measure", "indicator", "proxy"
        ]):
            attr_type = "OPERATIONAL"
        elif any(w in combined for w in [
            "p-value", "significance", "effect size", "confidence interval",
            "statistically", "clinically", "practical significance"
        ]):
            attr_type = "STATISTICAL"
        elif any(w in combined for w in [
            "replicate", "reproduce", "failed", "unable to confirm",
            "inconsistent with"
        ]):
            attr_type = "REPLICATION"
        elif any(w in combined for w in [
            "theory", "framework", "paradigm", "model", "assume",
            "cognitive", "behavioral", "theoretical"
        ]):
            attr_type = "THEORETICAL"
        else:
            attr_type = "UNKNOWN"

        return DisagreementAttribution(
            claim1_id=claim1.claim_id,
            claim2_id=claim2.claim_id,
            claim1_text=claim1.text,
            claim2_text=claim2.text,
            attribution_type=attr_type,
            confidence=0.5,   # lower confidence for rule-based
            evidence=f"Rule-based attribution: keyword signals in claim texts.",
            resolvability=RESOLVABILITY_PRIOR[attr_type],
            confidence_penalty=CONFIDENCE_PENALTY[attr_type],
        )

    def _build_report(
        self,
        query: str,
        attributions: List[DisagreementAttribution],
        total_contradictions: int,
    ) -> CausalAttributionReport:
        """Aggregate attributions into a full report."""
        if not attributions:
            return self._empty_report(query)

        type_counts = Counter(a.attribution_type for a in attributions)
        dominant_cause = type_counts.most_common(1)[0][0]

        mean_resolvability = sum(a.resolvability for a in attributions) / len(attributions)

        # Aggregate confidence penalty: use max rather than sum to avoid over-penalizing
        agg_penalty = max(a.confidence_penalty for a in attributions)

        controversy_depth = self._classify_depth(
            dominant_cause, mean_resolvability, len(attributions)
        )

        recommendations = self._generate_recommendations(type_counts, dominant_cause)

        return CausalAttributionReport(
            query=query,
            total_contradictions=total_contradictions,
            attributions=attributions,
            attribution_distribution=dict(type_counts),
            dominant_cause=dominant_cause,
            mean_resolvability=round(mean_resolvability, 3),
            aggregate_confidence_penalty=round(agg_penalty, 3),
            controversy_depth=controversy_depth,
            research_recommendations=recommendations,
        )

    def _classify_depth(
        self,
        dominant_cause: str,
        mean_resolvability: float,
        n_contradictions: int,
    ) -> str:
        """
        Classify how 'deep' the controversy is.
          - 'shallow': methodological, easily resolved
          - 'moderate': operational or statistical, needs more data
          - 'deep': theoretical or replication failures, paradigm-level
        """
        if dominant_cause in ("TEMPORAL", "METHODOLOGICAL", "POPULATION") and mean_resolvability > 0.65:
            return "shallow"
        elif dominant_cause in ("OPERATIONAL", "STATISTICAL") or mean_resolvability > 0.45:
            return "moderate"
        else:  # THEORETICAL, REPLICATION, or low resolvability
            return "deep"

    def _generate_recommendations(
        self, type_counts: Counter, dominant_cause: str
    ) -> List[str]:
        """Generate research recommendations based on attribution pattern."""
        recs = []
        if "METHODOLOGICAL" in type_counts:
            recs.append(
                "Conduct a systematic review standardizing methodological protocols "
                "across studies to resolve methodological disagreements."
            )
        if "POPULATION" in type_counts:
            recs.append(
                "Larger cross-population studies or meta-analyses with subgroup "
                "analyses could clarify scope conditions."
            )
        if "TEMPORAL" in type_counts:
            recs.append(
                "Prioritize recent evidence (post-2020); older findings may reflect "
                "a superseded understanding."
            )
        if "OPERATIONAL" in type_counts:
            recs.append(
                "Adopt consensus definitions for key constructs — definitional "
                "disagreements block cumulative scientific progress."
            )
        if "STATISTICAL" in type_counts:
            recs.append(
                "Shift from p-value significance to effect size reporting (Cohen's d, "
                "odds ratios) to make comparisons tractable."
            )
        if "THEORETICAL" in type_counts:
            recs.append(
                "This is a paradigm-level disagreement. Seek empirical tests that "
                "uniquely distinguish between competing theoretical predictions."
            )
        if "REPLICATION" in type_counts:
            recs.append(
                "Prioritize pre-registered replication studies before further "
                "theoretical development on this topic."
            )
        if not recs:
            recs.append(
                "The nature of this disagreement is unclear. A structured literature "
                "review with explicit comparison criteria is recommended."
            )
        return recs

    def _empty_report(self, query: str) -> CausalAttributionReport:
        """Return an empty report when no contradictions are found."""
        return CausalAttributionReport(
            query=query,
            total_contradictions=0,
            attributions=[],
            attribution_distribution={},
            dominant_cause="NONE",
            mean_resolvability=1.0,
            aggregate_confidence_penalty=0.0,
            controversy_depth="shallow",
            research_recommendations=[
                "No significant contradictions detected in the corpus for this query. "
                "Either strong consensus exists or the corpus lacks diversity of viewpoints."
            ],
        )

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict]:
        """Extract and parse JSON from LLM response."""
        # Try to find JSON block
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # Try full text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None
