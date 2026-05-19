"""
EVIRAG: Temporal Epistemic Drift Module
=========================================
Novel contribution: Track how scientific disagreements evolve over TIME.

No existing RAG system is time-aware in the sense of modeling how
epistemic consensus changes across publication years. This module:

  1. Extracts publication years from document metadata
  2. Assigns claims to time-bands (e.g., pre-2015, 2015-2020, 2020+)
  3. Computes TemporalDisagreementCurve (TDC): disagreement intensity per year band
  4. Classifies the disagreement trajectory:
       - 'converging': disagreement decreases over time (consensus forming)
       - 'diverging':  disagreement increases (emerging controversy)
       - 'stable':     disagreement stays constant
       - 'oscillating': non-monotonic, cyclical debate
  5. Detects "epistemic turning points" — years where disagreement changed sharply

This is critical for scientific RAG because a dominant view from 1990
may have been overturned by 2020 consensus. Standard RAG ignores this.

Inspired by:
  - Temporal Question Answering (Chen et al. 2021)
  - Science Map of scientific consensus evolution (Park et al. 2023)
  - Our novel contribution: temporal disagreement trajectory in RAG
"""

import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TemporalBand:
    """A time window of scientific claims."""
    start_year: int
    end_year: int
    label: str
    claims: List[Dict] = field(default_factory=list)
    contradiction_count: int = 0
    support_count: int = 0
    neutral_count: int = 0
    disagreement_density: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "claim_count": len(self.claims),
            "contradiction_count": self.contradiction_count,
            "support_count": self.support_count,
            "neutral_count": self.neutral_count,
            "disagreement_density": round(self.disagreement_density, 4),
        }


@dataclass
class EpistemicTurningPoint:
    """A year where disagreement changed sharply."""
    year: int
    delta_disagreement: float   # change in disagreement density
    direction: str              # 'spike' or 'drop'
    likely_cause: str           # heuristic explanation

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TemporalDriftReport:
    """Full temporal epistemic drift analysis."""
    query: str
    year_range: Tuple[int, int]
    time_bands: List[TemporalBand]
    trajectory: str             # 'converging', 'diverging', 'stable', 'oscillating'
    trajectory_confidence: float
    turning_points: List[EpistemicTurningPoint]
    recency_bias_warning: bool  # True if old papers dominate
    dominant_view_shift: bool   # True if dominant view changed over time
    temporal_confidence_bonus: float  # Bonus if recent papers strongly agree
    interpretation: str

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "year_range": list(self.year_range),
            "time_bands": [b.to_dict() for b in self.time_bands],
            "trajectory": self.trajectory,
            "trajectory_confidence": round(self.trajectory_confidence, 3),
            "turning_points": [tp.to_dict() for tp in self.turning_points],
            "recency_bias_warning": self.recency_bias_warning,
            "dominant_view_shift": self.dominant_view_shift,
            "temporal_confidence_bonus": round(self.temporal_confidence_bonus, 3),
            "interpretation": self.interpretation,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TemporalEpistemicDriftAnalyzer:
    """
    Analyzes how scientific disagreement evolves over time within a corpus.

    Core idea: if recent papers converge while older ones diverge,
    confidence should increase (converging trajectory). If recent papers
    are MORE divided than older ones, it's an emerging controversy
    and confidence should decrease.

    The TemporalDisagreementCurve (TDC) plots disagreement_density vs. time,
    and we fit a slope to classify the trajectory.
    """

    # Default time band boundaries
    DEFAULT_BANDS = [
        (1990, 2000, "pre-2000"),
        (2000, 2010, "2000-2010"),
        (2010, 2015, "2010-2015"),
        (2015, 2018, "2015-2018"),
        (2018, 2020, "2018-2020"),
        (2020, 2022, "2020-2022"),
        (2022, 2030, "2022+"),
    ]

    def __init__(self):
        self.current_year = 2025

    def analyze(
        self,
        query: str,
        claims: List[Any],           # List[AtomicClaim]
        relationships: List[Any],    # List[ClaimRelationship]
        doc_metadata: Optional[Dict[str, Dict]] = None,  # doc_id -> metadata with 'year'
    ) -> TemporalDriftReport:
        """
        Analyze temporal epistemic drift.

        Args:
            query: User query
            claims: All extracted atomic claims
            relationships: All claim relationships
            doc_metadata: Optional dict of doc metadata including publication year
        """
        # Extract year for each claim
        claim_years = self._extract_claim_years(claims, doc_metadata)

        if not claim_years:
            return self._empty_report(query)

        years = list(claim_years.values())
        year_min, year_max = min(years), max(years)

        # Build time bands
        bands = self._build_time_bands(claims, relationships, claim_years, year_min, year_max)

        # Filter empty bands
        active_bands = [b for b in bands if len(b.claims) > 0]

        if len(active_bands) < 2:
            return self._minimal_report(query, active_bands, year_min, year_max)

        # Analyze trajectory
        trajectory, traj_conf = self._classify_trajectory(active_bands)

        # Detect turning points
        turning_points = self._detect_turning_points(active_bands)

        # Recency bias check
        recent_fraction = self._compute_recent_fraction(claim_years)
        recency_bias_warning = bool(recent_fraction < 0.2)  # less than 20% recent papers

        # Dominant view shift detection
        dominant_view_shift = self._detect_view_shift(active_bands)

        # Temporal confidence adjustment
        temporal_bonus = self._compute_temporal_bonus(active_bands, trajectory)

        interpretation = self._interpret(
            trajectory, traj_conf, turning_points, recency_bias_warning,
            dominant_view_shift, active_bands
        )

        return TemporalDriftReport(
            query=query,
            year_range=(year_min, year_max),
            time_bands=active_bands,
            trajectory=trajectory,
            trajectory_confidence=traj_conf,
            turning_points=turning_points,
            recency_bias_warning=recency_bias_warning,
            dominant_view_shift=dominant_view_shift,
            temporal_confidence_bonus=temporal_bonus,
            interpretation=interpretation,
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _extract_claim_years(
        self,
        claims: List[Any],
        doc_metadata: Optional[Dict[str, Dict]],
    ) -> Dict[str, int]:
        """Extract publication year for each claim_id."""
        claim_years = {}

        for claim in claims:
            year = None

            # Try explicit metadata
            if doc_metadata and hasattr(claim, 'source_doc_id'):
                meta = doc_metadata.get(claim.source_doc_id, {})
                year = meta.get("year") or meta.get("publication_year")

            # Try to extract from doc_title (e.g., "Smith et al. (2019)")
            if year is None and hasattr(claim, 'source_doc_title'):
                year = self._extract_year_from_text(claim.source_doc_title)

            # Try context
            if year is None and hasattr(claim, 'context') and claim.context:
                year = self._extract_year_from_text(claim.context)

            if year and 1950 <= year <= self.current_year:
                claim_years[claim.claim_id] = year

        return claim_years

    def _extract_year_from_text(self, text: str) -> Optional[int]:
        """Extract a 4-digit year from text."""
        if not text:
            return None
        matches = re.findall(r'\b(19[5-9]\d|20[0-2]\d)\b', text)
        if matches:
            # Return the most recent plausible year
            years = [int(y) for y in matches if 1950 <= int(y) <= self.current_year]
            return max(years) if years else None
        return None

    def _build_time_bands(
        self,
        claims: List[Any],
        relationships: List[Any],
        claim_years: Dict[str, int],
        year_min: int,
        year_max: int,
    ) -> List[TemporalBand]:
        """Assign claims and relationships to time bands."""
        bands = []
        for start, end, label in self.DEFAULT_BANDS:
            if end < year_min or start > year_max:
                continue
            band = TemporalBand(start_year=start, end_year=end, label=label)

            # Claims in this band
            band_claim_ids = set()
            for claim in claims:
                year = claim_years.get(claim.claim_id)
                if year and start <= year < end:
                    band.claims.append({"id": claim.claim_id, "text": claim.text})
                    band_claim_ids.add(claim.claim_id)

            # Relationships where BOTH claims are in this band
            if band_claim_ids:
                for rel in relationships:
                    if rel.claim1_id in band_claim_ids and rel.claim2_id in band_claim_ids:
                        rt = rel.relationship.lower()
                        if "contradict" in rt:
                            band.contradiction_count += 1
                        elif "entail" in rt or "support" in rt:
                            band.support_count += 1
                        else:
                            band.neutral_count += 1

                total = band.contradiction_count + band.support_count + band.neutral_count
                if total > 0:
                    band.disagreement_density = band.contradiction_count / total

            bands.append(band)

        return bands

    def _classify_trajectory(
        self, active_bands: List[TemporalBand]
    ) -> Tuple[str, float]:
        """
        Classify the TemporalDisagreementCurve (TDC) trajectory.

        Fit a linear regression to (band_index, disagreement_density) and
        classify by slope sign and magnitude.
        """
        if len(active_bands) < 2:
            return "stable", 0.5

        x = np.arange(len(active_bands), dtype=float)
        y = np.array([b.disagreement_density for b in active_bands])

        # Linear regression
        if np.std(y) < 1e-6:
            return "stable", 0.9

        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])

        # R-squared for confidence
        y_pred = np.polyval(coeffs, x)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = float(1 - ss_res / ss_tot) if ss_tot > 1e-8 else 0.0

        # Detect oscillation
        sign_changes = sum(
            1 for i in range(1, len(y)) if (y[i] - y[i-1]) * (y[i-1] - y[i-2] if i > 1 else y[i] - y[i-1]) < 0
        )
        if sign_changes >= 2 and abs(slope) < 0.05:
            return "oscillating", float(max(0.4, r_squared))

        if slope < -0.02:
            return "converging", float(min(1.0, abs(slope) * 10 * r_squared + 0.5))
        elif slope > 0.02:
            return "diverging", float(min(1.0, abs(slope) * 10 * r_squared + 0.5))
        else:
            return "stable", float(max(0.5, r_squared))

    def _detect_turning_points(
        self, active_bands: List[TemporalBand]
    ) -> List[EpistemicTurningPoint]:
        """Detect years where disagreement density changed sharply."""
        turning_points = []
        if len(active_bands) < 3:
            return turning_points

        densities = [b.disagreement_density for b in active_bands]

        for i in range(1, len(densities) - 1):
            delta = densities[i] - densities[i - 1]
            if abs(delta) > 0.15:  # threshold for "sharp" change
                direction = "spike" if delta > 0 else "drop"
                year = (active_bands[i].start_year + active_bands[i].end_year) // 2
                cause = self._infer_turning_point_cause(
                    direction, active_bands[i], active_bands[i-1]
                )
                turning_points.append(EpistemicTurningPoint(
                    year=year,
                    delta_disagreement=round(delta, 4),
                    direction=direction,
                    likely_cause=cause,
                ))

        return turning_points

    def _infer_turning_point_cause(
        self,
        direction: str,
        current_band: TemporalBand,
        prev_band: TemporalBand,
    ) -> str:
        """Heuristic cause for a turning point."""
        if direction == "spike":
            return (
                f"Sharp increase in contradictions in {current_band.label} suggests "
                "a new wave of challenge studies or a methodological debate emerging."
            )
        else:
            return (
                f"Sharp decrease in contradictions in {current_band.label} suggests "
                "emerging consensus, possibly driven by a landmark study or meta-analysis."
            )

    def _compute_recent_fraction(self, claim_years: Dict[str, int]) -> float:
        """Fraction of claims from recent years (last 5 years)."""
        recent_cutoff = self.current_year - 5
        recent = sum(1 for y in claim_years.values() if y >= recent_cutoff)
        return recent / len(claim_years) if claim_years else 0.0

    def _detect_view_shift(self, active_bands: List[TemporalBand]) -> bool:
        """
        Detect if the dominant view changed over time.
        Proxy: disagreement density in first half vs second half changes substantially.
        """
        if len(active_bands) < 4:
            return False
        half = len(active_bands) // 2
        first_half_density = np.mean([b.disagreement_density for b in active_bands[:half]])
        second_half_density = np.mean([b.disagreement_density for b in active_bands[half:]])
        # Cast to Python bool — numpy.bool_ is not JSON-serialisable by FastAPI
        return bool(abs(second_half_density - first_half_density) > 0.2)

    def _compute_temporal_bonus(
        self, active_bands: List[TemporalBand], trajectory: str
    ) -> float:
        """
        Temporal confidence bonus/penalty:
          - converging trajectory → small bonus (recent consensus forming)
          - diverging trajectory → small penalty (emerging controversy)
          - oscillating → no change
        """
        if trajectory == "converging":
            return 0.05
        elif trajectory == "diverging":
            return -0.05
        else:
            return 0.0

    def _interpret(
        self,
        trajectory: str,
        traj_conf: float,
        turning_points: List[EpistemicTurningPoint],
        recency_bias_warning: bool,
        dominant_view_shift: bool,
        active_bands: List[TemporalBand],
    ) -> str:
        """Generate human-readable interpretation."""
        parts = []

        traj_descriptions = {
            "converging": "Scientific consensus is FORMING over time — recent papers show less disagreement.",
            "diverging": "This is an EMERGING controversy — disagreement is INCREASING in recent literature.",
            "stable": "Disagreement has remained STABLE over time — this is a persistent, unresolved debate.",
            "oscillating": "Disagreement OSCILLATES — periods of consensus and challenge alternate cyclically.",
        }
        parts.append(traj_descriptions.get(trajectory, "Trajectory unclear."))
        parts.append(f"Trajectory confidence: {traj_conf:.0%}.")

        if turning_points:
            tp = turning_points[0]
            parts.append(
                f"Key turning point around {tp.year}: {tp.likely_cause}"
            )

        if recency_bias_warning:
            parts.append(
                "WARNING: The corpus is dominated by older papers. "
                "Recent developments may not be well-represented."
            )

        if dominant_view_shift:
            parts.append(
                "The dominant view appears to have SHIFTED over the study period. "
                "Earlier and later claims may represent different states of knowledge."
            )

        return " ".join(parts)

    def _empty_report(self, query: str) -> TemporalDriftReport:
        return TemporalDriftReport(
            query=query,
            year_range=(0, 0),
            time_bands=[],
            trajectory="unknown",
            trajectory_confidence=0.0,
            turning_points=[],
            recency_bias_warning=True,
            dominant_view_shift=False,
            temporal_confidence_bonus=0.0,
            interpretation="No publication year information available in corpus metadata.",
        )

    def _minimal_report(
        self, query: str, bands: List[TemporalBand], year_min: int, year_max: int
    ) -> TemporalDriftReport:
        return TemporalDriftReport(
            query=query,
            year_range=(year_min, year_max),
            time_bands=bands,
            trajectory="stable",
            trajectory_confidence=0.5,
            turning_points=[],
            recency_bias_warning=False,
            dominant_view_shift=False,
            temporal_confidence_bonus=0.0,
            interpretation="Insufficient temporal diversity for trajectory analysis.",
        )
