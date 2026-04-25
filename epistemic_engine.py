"""
EVIRAG Epistemic Engine
Handles intent analysis, hypothesis generation, claim extraction, and NLI reasoning
"""

import json
import uuid
import requests
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from config import (
    get_model_config, get_prompt, INTENT_DIMENSIONS,
    SLM_MODELS, LLM_MODELS
)


@dataclass
class EpistemicIntent:
    """Classification of query epistemic characteristics"""
    factual_vs_disputed: str
    empirical_vs_theoretical: str
    disagreement_level: str
    reasoning: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExplanationHypothesis:
    """Structured explanation hypothesis"""
    central_hypothesis: str
    supporting_claims: List[str]
    assumptions: List[str]
    expected_counterclaims: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AtomicClaim:
    """Atomic, normalized claim extracted from text"""
    claim_id: str
    text: str
    source_chunk_id: str
    source_doc_id: str
    source_doc_title: str
    context: Optional[str] = None
    section: Optional[str] = None
    source_path: Optional[str] = None
    page_num: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ClaimRelationship:
    """Relationship between two claims"""
    claim1_id: str
    claim2_id: str
    relationship: str  # 'entailment', 'contradiction', 'neutral'
    confidence: float
    reasoning: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class OllamaClient:
    """Client for Ollama API — tuned for qwen3:0.6b (thinking model)."""

    # qwen3:0.6b always generates ~150 thinking tokens before the response.
    # num_predict must exceed that or the response field is empty.
    THINKING_TOKEN_OVERHEAD = 200

    def __init__(self, endpoint: str = "http://localhost:11434"):
        self.endpoint = endpoint

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system: Optional[str] = None,
    ) -> str:
        """Generate text using Ollama.

        For qwen3 thinking models, num_predict is automatically bumped by
        THINKING_TOKEN_OVERHEAD so the response field is never empty.
        The `thinking` field (chain-of-thought) is discarded — only the
        final `response` is returned.
        """
        url = f"{self.endpoint}/api/generate"

        # Guarantee we clear the thinking budget
        effective_tokens = max_tokens + self.THINKING_TOKEN_OVERHEAD

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
            "options": {
                "num_predict": effective_tokens,
                "num_ctx": 2048,   # keep context small for 0.6B
            },
        }

        if system:
            payload["system"] = system

        try:
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            text = result.get("response", "").strip()
            # Strip any residual <think>...</think> tags some models emit
            import re as _re
            text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
            return text
        except Exception as e:
            print(f"Ollama error ({model}): {e}")
            return ""
    
    def extract_json(self, text: str) -> Optional[Any]:
        """Extract JSON (object or array) from model response."""
        import re

        # 1. ```json ... ``` block — object or array
        for pattern in [
            r'```json\s*([\[{].*?[\]}])\s*```',
            r'```\s*([\[{].*?[\]}])\s*```',
        ]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass

        # 2. Bare JSON array [...]
        m = re.search(r'(\[.*\])', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        
        # Look for raw JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        return None


class EpistemicIntentAnalyzer:
    """Analyze epistemic intent of scientific queries"""
    
    def __init__(self, backend="local"):
        self.backend = backend
        self.model_config = get_model_config("intent_analyzer", "slm", backend)
        
        # Auto-select client based on endpoint
        if self.model_config and "groq.com" in self.model_config.endpoint:
            from groq_client import GroqClient
            self.client = GroqClient()
        else:
            self.client = OllamaClient()
    
    def analyze(self, query: str) -> EpistemicIntent:
        """Classify query along epistemic dimensions"""
        
        prompt = get_prompt("intent_analysis", query=query)
        
        system_prompt = """You are an expert at analyzing scientific questions. 
Your task is to classify queries along epistemic dimensions to guide retrieval strategy.
Always respond with valid JSON."""
        
        response = self.client.generate(
            model=self.model_config.name,
            prompt=prompt,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_tokens,
            system=system_prompt
        )
        
        # Parse JSON response
        result = self.client.extract_json(response)
        
        if result:
            return EpistemicIntent(
                factual_vs_disputed=result.get("factual_vs_disputed", "somewhat_disputed"),
                empirical_vs_theoretical=result.get("empirical_vs_theoretical", "mixed"),
                disagreement_level=result.get("disagreement_level", "medium"),
                reasoning=result.get("reasoning", "")
            )
        else:
            # Default to conservative assumptions
            return EpistemicIntent(
                factual_vs_disputed="somewhat_disputed",
                empirical_vs_theoretical="mixed",
                disagreement_level="medium",
                reasoning="Failed to parse intent"
            )


class HypothesisGenerator:
    """Generate explanation hypotheses (Explanation-First Verification)"""
    
    def __init__(self, backend="local"):
        self.backend = backend
        self.slm_config = get_model_config("hypothesis_generator", "slm", backend)
        self.llm_config = get_model_config("hypothesis_generator", "llm", backend)
        
        # Auto-select client
        config_to_check = self.llm_config or self.slm_config
        if config_to_check and "groq.com" in config_to_check.endpoint:
            from groq_client import GroqClient
            self.client = GroqClient()
        else:
            self.client = OllamaClient()
    
    def generate(
        self, 
        query: str, 
        intent: EpistemicIntent,
        use_llm: bool = False
    ) -> ExplanationHypothesis:
        """Generate explanation hypothesis"""
        
        # Model selection: prefer LLM if requested OR if no SLM available
        if use_llm or self.slm_config is None:
            model_config = self.llm_config
        else:
            model_config = self.slm_config
        
        # If still None, raise error
        if model_config is None:
            raise ValueError(f"No model config available for hypothesis generator (backend: {self.backend})")
        
        prompt = get_prompt(
            "hypothesis_generation",
            query=query,
            intent=intent.to_dict()
        )
        
        system_prompt = """You are an expert at generating testable scientific hypotheses.
Given a query, generate a structured hypothesis that can be verified against literature.
Each claim should be atomic and falsifiable."""
        
        response = self.client.generate(
            model=model_config.name,
            prompt=prompt,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            system=system_prompt
        )
        
        result = self.client.extract_json(response)
        
        if result:
            return ExplanationHypothesis(
                central_hypothesis=result.get("central_hypothesis", query),
                supporting_claims=result.get("supporting_claims", []),
                assumptions=result.get("assumptions", []),
                expected_counterclaims=result.get("expected_counterclaims", [])
            )
        else:
            # Fallback
            return ExplanationHypothesis(
                central_hypothesis=query,
                supporting_claims=[],
                assumptions=[],
                expected_counterclaims=[]
            )


class ClaimExtractor:
    """Extract atomic claims from retrieved chunks"""
    
    def __init__(self, backend="local"):
        self.backend = backend
        self.filter_config = get_model_config("claim_filter", "slm", backend)
        self.extractor_config = get_model_config("claim_extractor", "llm", backend)
        
        # Auto-select client
        if self.extractor_config and "groq.com" in self.extractor_config.endpoint:
            from groq_client import GroqClient
            self.client = GroqClient()
        else:
            self.client = OllamaClient()
    
    def extract_from_chunk(self, chunk, chunk_id_counter: int = 0) -> List[AtomicClaim]:
        """Extract claims from a single chunk"""
        
        # Step 1: Filter sentences likely to contain claims (SLM)
        filtered_sentences = self._filter_claim_sentences(chunk.text, chunk.metadata)
        
        if not filtered_sentences:
            return []
        
        # Step 2: Extract atomic claims (LLM)
        claims = self._extract_atomic_claims(
            filtered_sentences,
            chunk.chunk_id,
            chunk.doc_id,
            chunk.metadata.get("title", "Unknown")
        )
        
        # Assign globally unique IDs using UUID — prevents cross-agent/cross-chunk collisions
        for claim in claims:
            claim.claim_id = f"claim_{uuid.uuid4().hex[:12]}"
            claim.section = chunk.section
            claim.source_path = chunk.metadata.get("path")
            claim.page_num = getattr(chunk, "page_num", None)

        return claims
    
    def _filter_claim_sentences(self, text: str, metadata: Dict) -> List[str]:
        """Filter sentences likely to contain factual claims (SLM)"""
        
        prompt = get_prompt(
            "claim_filtering",
            text=text,
            source=metadata.get("title", "Unknown")
        )
        
        system_prompt = """You filter text to identify sentences containing factual scientific claims.
Ignore background, motivation, and non-assertive statements.
Return a JSON list of claim sentences."""
        
        response = self.client.generate(
            model=self.filter_config.name,
            prompt=prompt,
            temperature=self.filter_config.temperature,
            max_tokens=self.filter_config.max_tokens,
            system=system_prompt
        )
        
        # Try to parse JSON list
        result = self.client.extract_json(response)

        def _to_str(item) -> str:
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                return item.get("sentence") or item.get("text") or item.get("claim") or str(item)
            return str(item)

        if result and isinstance(result, list):
            return [s for s in (_to_str(x) for x in result) if len(s.strip()) > 10]
        elif result and isinstance(result, dict) and "sentences" in result:
            raw = result["sentences"]
            return [s for s in (_to_str(x) for x in raw) if len(s.strip()) > 10]

        # Fallback: split the raw text into sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20][:10]
    
    def _extract_atomic_claims(
        self,
        sentences: List[str],
        chunk_id: str,
        doc_id: str,
        doc_title: str
    ) -> List[AtomicClaim]:
        """Extract atomic claims from filtered sentences (LLM)"""
        
        prompt = get_prompt(
            "claim_extraction",
            sentences="\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences)),
            source=doc_title
        )
        
        system_prompt = """You extract atomic, normalized claims from scientific text.
Each claim should be self-contained, verifiable, and properly attributed.
Normalize terminology but preserve meaning. Always return valid JSON."""
        
        response = self.client.generate(
            model=self.extractor_config.name,
            prompt=prompt,
            temperature=self.extractor_config.temperature,
            max_tokens=self.extractor_config.max_tokens,
            system=system_prompt
        )
        
        result = self.client.extract_json(response)

        claims = []
        if result and isinstance(result, dict) and "claims" in result:
            for claim_data in result["claims"]:
                text = claim_data.get("claim", "") or claim_data.get("text", "")
                if text.strip():
                    claims.append(AtomicClaim(
                        claim_id="",
                        text=text.strip(),
                        source_chunk_id=chunk_id,
                        source_doc_id=doc_id,
                        source_doc_title=doc_title,
                        context=claim_data.get("context"),
                        source_path=None,
                        page_num=None,
                    ))
        elif result and isinstance(result, list):
            # Model returned a plain list of claim strings
            for item in result:
                text = item if isinstance(item, str) else item.get("claim", "") if isinstance(item, dict) else ""
                if text.strip():
                    claims.append(AtomicClaim(
                        claim_id="",
                        text=text.strip(),
                        source_chunk_id=chunk_id,
                        source_doc_id=doc_id,
                        source_doc_title=doc_title,
                        context=None,
                        source_path=None,
                        page_num=None,
                    ))

        # Hard fallback: if JSON parsing failed entirely, treat each sentence as a claim
        if not claims:
            import re as _re
            # Normalise sentences to plain strings regardless of what the filter returned
            str_sentences = []
            for item in sentences:
                if isinstance(item, str):
                    str_sentences.append(item)
                elif isinstance(item, dict):
                    str_sentences.append(item.get("sentence") or item.get("text") or item.get("claim") or "")
            raw_sentences = _re.split(r'(?<=[.!?])\s+', " ".join(str_sentences))
            for s in raw_sentences:
                s = s.strip()
                if len(s) > 30:   # skip very short fragments
                    claims.append(AtomicClaim(
                        claim_id="",
                        text=s,
                        source_chunk_id=chunk_id,
                        source_doc_id=doc_id,
                        source_doc_title=doc_title,
                        context=None,
                        source_path=None,
                        page_num=None,
                    ))
            claims = claims[:5]  # cap fallback claims per chunk

        return claims


class NLIReasoner:
    """Natural Language Inference for claim relationships"""
    
    def __init__(self, backend="local"):
        self.backend = backend
        self.prefilter_config = get_model_config("nli_prefilter", "slm", backend)
        self.verifier_config = get_model_config("nli_verifier", "llm", backend)
        
        # Auto-select client
        if self.verifier_config and "groq.com" in self.verifier_config.endpoint:
            from groq_client import GroqClient
            self.client = GroqClient()
        else:
            self.client = OllamaClient()
    
    def analyze_relationship(
        self,
        claim1: AtomicClaim,
        claim2: AtomicClaim,
        use_prefilter: bool = True
    ) -> Optional[ClaimRelationship]:
        """Determine relationship between two claims"""
        
        # Optional: prefilter with SLM for efficiency
        if use_prefilter:
            should_verify = self._prefilter_pair(claim1, claim2)
            if not should_verify:
                return ClaimRelationship(
                    claim1_id=claim1.claim_id,
                    claim2_id=claim2.claim_id,
                    relationship="neutral",
                    confidence=0.5,
                    reasoning="Prefiltered as likely neutral"
                )
        
        # Full NLI verification with LLM
        return self._verify_relationship(claim1, claim2)
    
    def _prefilter_pair(self, claim1: AtomicClaim, claim2: AtomicClaim) -> bool:
        """Quick SLM check if pair is worth detailed verification"""
        
        prompt = f"""Are these two claims likely to support or contradict each other?

Claim 1: {claim1.text}
Claim 2: {claim2.text}

Answer: yes (worth checking) or no (likely neutral)"""
        
        response = self.client.generate(
            model=self.prefilter_config.name,
            prompt=prompt,
            temperature=0.1,
            max_tokens=10
        )
        
        return "yes" in response.lower()
    
    def _verify_relationship(
        self,
        claim1: AtomicClaim,
        claim2: AtomicClaim
    ) -> ClaimRelationship:
        """Perform detailed NLI reasoning"""
        
        prompt = get_prompt(
            "nli_verification",
            claim1=claim1.text,
            source1=claim1.source_doc_title,
            claim2=claim2.text,
            source2=claim2.source_doc_title
        )
        
        system_prompt = """You are an expert at determining logical relationships between scientific claims.

IMPORTANT: Be sensitive to contradictions - if claims present genuinely opposing views, conflicting results, or contradictory findings (even partially), classify as 'contradiction'.

Examples of contradictions to detect:
- "RAG improves performance" vs "RAG hurts performance"  
- "Model X outperforms Y" vs "Model Y outperforms X"
- "Method shows significant gains" vs "Method shows no improvement"
- "Approach is effective" vs "Approach is ineffective"

Classify as:
- entailment: claim2 supports, agrees with, or reinforces claim1
- contradiction: claims present opposing, conflicting, or contradictory information
- neutral: claims are unrelated or neither support nor oppose

Be precise and provide reasoning. Always return valid JSON."""
        
        response = self.client.generate(
            model=self.verifier_config.name,
            prompt=prompt,
            temperature=self.verifier_config.temperature,
            max_tokens=self.verifier_config.max_tokens,
            system=system_prompt
        )
        
        result = self.client.extract_json(response)

        if result:
            relationship = result.get("relationship", "neutral")

            # Calibrated confidence: small models (0.6B) always output 1.0 for raw floats.
            # Use categorical certainty + relationship-type priors instead.
            CERTAINTY_MAP = {"certain": 0.88, "likely": 0.70, "uncertain": 0.52}
            # Relationship-type base priors (contradictions are harder to detect reliably)
            REL_PRIOR = {"contradiction": 0.78, "entailment": 0.82, "neutral": 0.65}

            certainty_str = str(result.get("certainty", "likely")).lower()
            certainty_val = CERTAINTY_MAP.get(certainty_str, 0.70)
            rel_prior = REL_PRIOR.get(relationship, 0.70)
            # Blend: 60% certainty label, 40% relationship prior
            confidence = round(0.6 * certainty_val + 0.4 * rel_prior, 3)

            return ClaimRelationship(
                claim1_id=claim1.claim_id,
                claim2_id=claim2.claim_id,
                relationship=relationship,
                confidence=confidence,
                reasoning=result.get("reasoning", "")
            )
        else:
            return ClaimRelationship(
                claim1_id=claim1.claim_id,
                claim2_id=claim2.claim_id,
                relationship="neutral",
                confidence=0.5,
                reasoning="Failed to parse NLI result"
            )


class EpistemicEngine:
    """Main epistemic processing pipeline"""
    
    def __init__(self, backend="local"):
        self.backend = backend
        self.intent_analyzer = EpistemicIntentAnalyzer(backend)
        self.hypothesis_generator = HypothesisGenerator(backend)
        self.claim_extractor = ClaimExtractor(backend)
        self.nli_reasoner = NLIReasoner(backend)
    
    def process_query(self, query: str) -> Tuple[EpistemicIntent, ExplanationHypothesis]:
        """Process query: analyze intent and generate hypothesis"""
        
        # Analyze intent
        intent = self.intent_analyzer.analyze(query)
        
        # Generate hypothesis (escalate to LLM if high disagreement)
        use_llm = intent.disagreement_level == "high"
        hypothesis = self.hypothesis_generator.generate(query, intent, use_llm)
        
        return intent, hypothesis
    
    def extract_claims_from_chunks(self, chunks: List) -> List[AtomicClaim]:
        """Extract claims from multiple chunks"""
        all_claims = []
        
        for i, chunk in enumerate(chunks):
            claims = self.claim_extractor.extract_from_chunk(chunk, chunk_id_counter=i)
            all_claims.extend(claims)
        
        return all_claims
    
    def build_claim_relationships(
        self,
        claims: List[AtomicClaim],
        max_pairs: int = 1000
    ) -> List[ClaimRelationship]:
        """Build relationships between claims"""
        
        relationships = []
        
        # Limit number of pairs for efficiency
        n = len(claims)
        pairs_to_check = min(max_pairs, n * (n - 1) // 2)
        
        print(f"Analyzing relationships between {n} claims ({pairs_to_check} pairs)...")
        
        checked = 0
        contradictions_found = 0
        entailments_found = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if checked >= pairs_to_check:
                    break
                
                rel = self.nli_reasoner.analyze_relationship(
                    claims[i], claims[j], use_prefilter=False  # Disable aggressive prefilter
                )
                
                # DEBUG: Track what NLI is actually returning
                if rel:
                    if rel.relationship == "contradiction":
                        contradictions_found += 1
                    elif rel.relationship == "entailment":
                        entailments_found += 1
                
                if rel and rel.relationship != "neutral":
                    relationships.append(rel)
                    print(f"  Found: {rel.relationship} (conf: {rel.confidence:.3f})")
                
                checked += 1
            
            if checked >= pairs_to_check:
                break
        
        print(f"Relationships analyzed: {checked} pairs | Contradictions: {contradictions_found} | Entailments: {entailments_found} | Neutral: {checked - contradictions_found - entailments_found}")
        
        return relationships


if __name__ == "__main__":
    # Test epistemic engine
    print("Testing Epistemic Engine")
    print("=" * 60)
    
    engine = EpistemicEngine()
    
    # Test query
    test_query = "Is overparameterization necessary for generalization in deep neural networks?"
    
    print(f"\nQuery: {test_query}\n")
    
    # Process query
    intent, hypothesis = engine.process_query(test_query)
    
    print("Intent Analysis:")
    print(json.dumps(intent.to_dict(), indent=2))
    
    print("\nHypothesis:")
    print(json.dumps(hypothesis.to_dict(), indent=2))
