"""
EVIRAG Multi-Agent Deliberative Retrieval (MAD-RAG)
Role-specialized epistemic agents with constrained autonomy
"""

import statistics
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from config import AGENT_CONFIG, SYSTEM_CONSTRAINTS, get_prompt, get_model_config
from epistemic_engine import OllamaClient, AtomicClaim, ExplanationHypothesis
from groq_client import GroqClient


@dataclass
class AgentState:
    """Shared state observable by all agents"""
    hypothesis: ExplanationHypothesis
    current_evidence: List[AtomicClaim]
    retrieval_history: List[str]
    disagreement_level: str
    
    def to_dict(self) -> Dict:
        return {
            "hypothesis": self.hypothesis.to_dict(),
            "evidence_count": len(self.current_evidence),
            "retrieval_count": len(self.retrieval_history),
            "disagreement_level": self.disagreement_level
        }


@dataclass
class AgentOutput:
    """Output from an agent"""
    agent_name: str
    retrieval_query: str
    retrieved_chunks: List  # DocumentChunks
    extracted_claims: List[AtomicClaim]
    stance: str  # 'support', 'oppose', 'neutral'
    confidence: float
    reasoning: str
    status: str = "ok"
    duration_ms: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "agent_name": self.agent_name,
            "retrieval_query": self.retrieval_query,
            "num_chunks": len(self.retrieved_chunks),
            "num_claims": len(self.extracted_claims),
            "stance": self.stance,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "status": self.status,
            "duration_ms": self.duration_ms
        }


class BaseAgent(ABC):
    """Base class for all epistemic agents"""
    
    def __init__(self, name: str, objective: str, backend="local"):
        self.name = name
        self.objective = objective
        self.backend = backend
        self.config = AGENT_CONFIG.get(name, {})
        self.model_config = get_model_config(name, "agent", backend)
        
        # Auto-select client
        if self.model_config and "groq.com" in self.model_config.endpoint:
            self.client = GroqClient()
        else:
            self.client = OllamaClient()
        
        if not self.model_config:
            raise ValueError(f"No model config found for agent: {name} (backend: {backend})")
        
        self.retrieval_k = self.config.get("retrieval_k", 5) # Default to 5 if not specified
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7) # Default to 0.7
    
    @abstractmethod
    def generate_retrieval_query(
        self, 
        hypothesis: ExplanationHypothesis,
        state: AgentState
    ) -> str:
        """Generate a retrieval query based on agent's objective"""
        pass
    
    @abstractmethod
    def evaluate_stance(
        self,
        hypothesis: ExplanationHypothesis,
        claims: List[AtomicClaim]
    ) -> Tuple[str, float, str]:
        """Evaluate stance on hypothesis given claims"""
        pass
    
    def execute(
        self,
        hypothesis: ExplanationHypothesis,
        state: AgentState,
        retrieval_func,  # Function to retrieve chunks
        claim_extraction_func  # Function to extract claims
    ) -> AgentOutput:
        """Execute agent's retrieval and reasoning pipeline"""
        
        # 1. Generate retrieval query
        query = self.generate_retrieval_query(hypothesis, state)
        
        # 2. Retrieve chunks
        chunks = retrieval_func(query, k=self.retrieval_k)
        
        # 3. Extract claims from chunks
        # Use a globally unique counter so claim IDs don't collide across agents/chunks
        claims = []
        chunk_counter = getattr(self, '_global_chunk_counter', 0)
        for chunk, score in chunks:
            chunk_claims = claim_extraction_func(chunk, chunk_counter)
            chunk_counter += 1
            claims.extend(chunk_claims)
        self._global_chunk_counter = chunk_counter
        
        # 4. Evaluate stance
        stance, confidence, reasoning = self.evaluate_stance(hypothesis, claims)
        
        return AgentOutput(
            agent_name=self.name,
            retrieval_query=query,
            retrieved_chunks=[c for c, _ in chunks],
            extracted_claims=claims,
            stance=stance,
            confidence=confidence,
            reasoning=reasoning
        )


class PrecisionAgent(BaseAgent):
    """High-confidence support agent"""
    
    def __init__(self, name: str, objective: str, backend="local"):
        super().__init__(name, objective, backend)
    
    def generate_retrieval_query(
        self,
        hypothesis: ExplanationHypothesis,
        state: AgentState
    ) -> str:
        """Generate query seeking high-confidence supporting evidence"""
        
        prompt = get_prompt(
            "agent_retrieval",
            agent_name="Precision Agent",
            objective="Find high-confidence evidence that strongly supports the hypothesis",
            hypothesis=hypothesis.central_hypothesis,
            evidence_state=json.dumps(state.to_dict(), indent=2),
            strategy="strict_semantic: focus on claims with strong empirical backing"
        )
        
        system_prompt = f"""You are the Precision Agent. Your goal is to find the strongest, 
most reliable evidence that supports the hypothesis. Focus on well-established findings 
with strong empirical support. Return only a retrieval query."""
        
        response = self.client.generate(
            model=self.model_config.name,
            prompt=prompt,
            temperature=0.1,
            max_tokens=200,
            system=system_prompt
        )
        
        # Extract query from response
        result = self.client.extract_json(response)
        if result and "query" in result:
            return result["query"]
        
        # Fallback
        return hypothesis.central_hypothesis
    
    def evaluate_stance(
        self,
        hypothesis: ExplanationHypothesis,
        claims: List[AtomicClaim]
    ) -> Tuple[str, float, str]:
        """Evaluate if claims provide strong support"""
        
        if not claims:
            return "neutral", 0.0, "No claims found"
        
        # Simple heuristic: check if claims support hypothesis
        supporting_count = 0
        for claim in claims:
            # Use LLM to check support
            support_check = self._check_claim_support(hypothesis.central_hypothesis, claim.text)
            if support_check:
                supporting_count += 1
        
        support_ratio = supporting_count / len(claims)
        
        if support_ratio >= 0.7:
            stance = "support"
            confidence = min(support_ratio * 1.2, 1.0)
        elif support_ratio <= 0.3:
            stance = "oppose"
            confidence = (1 - support_ratio) * 0.8
        else:
            stance = "neutral"
            confidence = 0.5
        
        reasoning = f"{supporting_count}/{len(claims)} claims support hypothesis"
        
        return stance, confidence, reasoning
    
    def _check_claim_support(self, hypothesis: str, claim: str) -> bool:
        """Quick check if claim supports hypothesis"""
        prompt = f"""Does this claim support the hypothesis?

Hypothesis: {hypothesis}
Claim: {claim}

Answer: yes or no"""
        
        response = self.client.generate(
            model=self.model_config.name,
            prompt=prompt,
            temperature=0.1,
            max_tokens=10
        )
        
        return "yes" in response.lower()


class RecallAgent(BaseAgent):
    """Broad coverage agent"""
    
    def __init__(self, name: str, objective: str, backend="local"):
        super().__init__(name, objective, backend)
    
    def generate_retrieval_query(
        self,
        hypothesis: ExplanationHypothesis,
        state: AgentState
    ) -> str:
        """Generate query for broad evidence coverage"""
        
        prompt = get_prompt(
            "agent_retrieval",
            agent_name="Recall Agent",
            objective="Achieve broad coverage of all relevant evidence",
            hypothesis=hypothesis.central_hypothesis,
            evidence_state=json.dumps(state.to_dict(), indent=2),
            strategy="expansive_semantic: cast a wide net to capture diverse perspectives"
        )
        
        system_prompt = """You are the Recall Agent. Your goal is to find all relevant evidence,
including edge cases and alternative perspectives. Prioritize breadth over precision.
Return only a retrieval query."""
        
        response = self.client.generate(
            model=self.model_config.name,
            prompt=prompt,
            temperature=0.3,
            max_tokens=200,
            system=system_prompt
        )
        
        result = self.client.extract_json(response)
        if result and "query" in result:
            return result["query"]
        
        # Fallback: broader query
        return f"{hypothesis.central_hypothesis} related evidence"
    
    def evaluate_stance(
        self,
        hypothesis: ExplanationHypothesis,
        claims: List[AtomicClaim]
    ) -> Tuple[str, float, str]:
        """Evaluate overall coverage and stance"""
        
        if not claims:
            return "neutral", 0.0, "No claims found"
        
        # Recall agent doesn't take strong stance, focuses on coverage
        reasoning = f"Retrieved {len(claims)} claims for broad coverage"
        return "neutral", 0.6, reasoning


class SkepticAgent(BaseAgent):
    """Actively find contradictions agent"""
    
    def __init__(self, name: str, objective: str, backend="local"):
        super().__init__(name, objective, backend)
    
    def generate_retrieval_query(
        self,
        hypothesis: ExplanationHypothesis,
        state: AgentState
    ) -> str:
        """Generate query seeking contradictory evidence"""
        
        # Look for negations and alternatives
        prompt = get_prompt(
            "agent_retrieval",
            agent_name="Skeptic Agent",
            objective="Find evidence that contradicts or challenges the hypothesis",
            hypothesis=hypothesis.central_hypothesis,
            evidence_state=json.dumps(state.to_dict(), indent=2),
            strategy="adversarial_semantic: seek disconfirming evidence and alternative explanations"
        )
        
        system_prompt = """You are the Skeptic Agent. Your goal is to find evidence that
contradicts the hypothesis or supports alternative explanations. Be actively adversarial
in your search. Return only a retrieval query."""
        
        response = self.client.generate(
            model=self.model_config.name,
            prompt=prompt,
            temperature=0.2,
            max_tokens=200,
            system=system_prompt
        )
        
        result = self.client.extract_json(response)
        if result and "query" in result:
            return result["query"]
        
        # Fallback: negation
        return f"challenges to {hypothesis.central_hypothesis}"
    
    def evaluate_stance(
        self,
        hypothesis: ExplanationHypothesis,
        claims: List[AtomicClaim]
    ) -> Tuple[str, float, str]:
        """Evaluate if claims contradict hypothesis"""
        
        if not claims:
            return "neutral", 0.0, "No contradictory claims found"
        
        contradicting_count = 0
        for claim in claims:
            contradicts = self._check_claim_contradiction(hypothesis.central_hypothesis, claim.text)
            if contradicts:
                contradicting_count += 1
        
        contradict_ratio = contradicting_count / len(claims)
        
        if contradict_ratio >= 0.5:
            stance = "oppose"
            confidence = min(contradict_ratio * 1.3, 1.0)
        else:
            stance = "neutral"
            confidence = 0.5
        
        reasoning = f"{contradicting_count}/{len(claims)} claims contradict hypothesis"
        
        return stance, confidence, reasoning
    
    def _check_claim_contradiction(self, hypothesis: str, claim: str) -> bool:
        """Quick check if claim contradicts hypothesis"""
        prompt = f"""Does this claim contradict the hypothesis?

Hypothesis: {hypothesis}
Claim: {claim}

Answer: yes or no"""
        
        response = self.client.generate(
            model=self.model_config.name,
            prompt=prompt,
            temperature=0.1,
            max_tokens=10
        )
        
        return "yes" in response.lower()


class CounterfactualAgent(BaseAgent):
    """Attempt disproof agent"""
    
    def __init__(self, name: str, objective: str, backend="local"):
        super().__init__(name, objective, backend)
    
    def generate_retrieval_query(
        self,
        hypothesis: ExplanationHypothesis,
        state: AgentState
    ) -> str:
        """Generate query for counterfactual reasoning"""
        
        prompt = get_prompt(
            "agent_retrieval",
            agent_name="Counterfactual Agent",
            objective="Find evidence for alternative explanations and attempt to disprove hypothesis",
            hypothesis=hypothesis.central_hypothesis,
            evidence_state=json.dumps(state.to_dict(), indent=2),
            strategy="counterfactual_reasoning: explore what-if scenarios and alternative causal mechanisms"
        )
        
        system_prompt = """You are the Counterfactual Agent. Your goal is to explore
alternative explanations and attempt to disprove the hypothesis through counterfactual
reasoning. Return only a retrieval query."""
        
        response = self.client.generate(
            model=self.model_config.name,
            prompt=prompt,
            temperature=0.3,
            max_tokens=200,
            system=system_prompt
        )
        
        result = self.client.extract_json(response)
        if result and "query" in result:
            return result["query"]
        
        # Fallback
        return f"alternative explanations for {hypothesis.central_hypothesis}"
    
    def evaluate_stance(
        self,
        hypothesis: ExplanationHypothesis,
        claims: List[AtomicClaim]
    ) -> Tuple[str, float, str]:
        """Evaluate counterfactual evidence"""
        
        if not claims:
            return "neutral", 0.0, "No counterfactual evidence found"
        
        # Counterfactual agent looks for alternatives
        reasoning = f"Found {len(claims)} claims related to alternative explanations"
        return "oppose", 0.6, reasoning


class MultiAgentOrchestrator:
    """Orchestrate multiple epistemic agents for deliberative retrieval"""
    
    def __init__(self, enabled_agents: List[str] = None, backend="local"):
        self.backend = backend
        self.enabled_agents = enabled_agents or ["precision", "recall", "skeptic", "counterfactual"]
        self.agents = self._initialize_agents()
    
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize agents based on enabled list"""
        agents = {}
        
        if "precision" in self.enabled_agents:
            agents["precision"] = PrecisionAgent(
                name="precision",
                objective="Find high-confidence evidence",
                backend=self.backend
            )
        
        if "recall" in self.enabled_agents:
            agents["recall"] = RecallAgent(
                name="recall",
                objective="Achieve broad coverage",
                backend=self.backend
            )
        
        if "skeptic" in self.enabled_agents:
            agents["skeptic"] = SkepticAgent(
                name="skeptic",
                objective="Find contradictory evidence",
                backend=self.backend
            )
        
        if "counterfactual" in self.enabled_agents:
            agents["counterfactual"] = CounterfactualAgent(
                name="counterfactual",
                objective="Explore alternative explanations",
                backend=self.backend
            )
        
        return agents
    
    def deliberate(
        self,
        hypothesis: ExplanationHypothesis,
        disagreement_level: str,
        retrieval_func,
        claim_extraction_func,
        enabled_agents: Optional[List[str]] = None
    ) -> List[AgentOutput]:
        """Execute multi-agent deliberative retrieval"""
        
        # Filter agents
        if enabled_agents:
            active_agents = [self.agents[name] for name in enabled_agents if name in self.agents]
        else:
            active_agents = list(self.agents.values())
        
        # Initialize shared state
        state = AgentState(
            hypothesis=hypothesis,
            current_evidence=[],
            retrieval_history=[],
            disagreement_level=disagreement_level
        )

        max_workers = max(1, min(
            SYSTEM_CONSTRAINTS.get("max_concurrent_agents", len(active_agents)),
            len(active_agents)
        ))
        print(f"Executing {len(active_agents)} agents with max concurrency {max_workers}...")

        def run_agent(agent: BaseAgent) -> AgentOutput:
            print(f"Executing {agent.name} agent...")
            started = time.perf_counter()
            # All agents see the same hypothesis and initial evidence state. Their
            # disagreement is resolved downstream by relationship building.
            local_state = AgentState(
                hypothesis=hypothesis,
                current_evidence=list(state.current_evidence),
                retrieval_history=list(state.retrieval_history),
                disagreement_level=disagreement_level,
            )
            try:
                output = agent.execute(
                    hypothesis=hypothesis,
                    state=local_state,
                    retrieval_func=retrieval_func,
                    claim_extraction_func=claim_extraction_func
                )
                output.duration_ms = int((time.perf_counter() - started) * 1000)
                output.status = "ok"
                return output
            except Exception as exc:
                return AgentOutput(
                    agent_name=agent.name,
                    retrieval_query="",
                    retrieved_chunks=[],
                    extracted_claims=[],
                    stance="neutral",
                    confidence=0.0,
                    reasoning=f"Agent failed: {exc}",
                    status="error",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )

        output_by_name: Dict[str, AgentOutput] = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="evirag-agent") as executor:
            futures = {executor.submit(run_agent, agent): agent.name for agent in active_agents}
            for future in as_completed(futures):
                output = future.result()
                output_by_name[output.agent_name] = output
                print(
                    f"{output.agent_name} agent finished: "
                    f"{output.status}, {len(output.extracted_claims)} claims, "
                    f"{output.duration_ms}ms"
                )

        agent_outputs = [
            output_by_name[agent.name]
            for agent in active_agents
            if agent.name in output_by_name
        ]

        state.current_evidence = [
            claim
            for output in agent_outputs
            for claim in output.extracted_claims
        ]
        state.retrieval_history = [
            output.retrieval_query
            for output in agent_outputs
            if output.retrieval_query
        ]

        return agent_outputs
    
    def get_agent_summary(self, agent_outputs: List[AgentOutput]) -> Dict:
        """Get summary of agent outputs"""
        if not agent_outputs:
            return {
                "total_agents": 0,
                "total_claims": 0,
                "stances": {"support": 0, "oppose": 0, "neutral": 0},
                "avg_confidence": 0.0,
                "median_confidence": 0.0,
                "parallel": False,
                "max_concurrency": 0,
                "runtime_ms": 0,
                "novelty_pressure": 0.0,
                "agents": [],
            }

        confidences = [o.confidence for o in agent_outputs]
        total_claims = sum(len(o.extracted_claims) for o in agent_outputs)
        unique_claims = len({claim.text for output in agent_outputs for claim in output.extracted_claims})
        oppose_agents = len([o for o in agent_outputs if o.stance == "oppose"])
        novelty_pressure = 0.0
        if total_claims:
            novelty_pressure = (unique_claims / total_claims + oppose_agents / len(agent_outputs)) / 2

        return {
            "total_agents": len(agent_outputs),
            "total_claims": total_claims,
            "stances": {
                "support": len([o for o in agent_outputs if o.stance == "support"]),
                "oppose": len([o for o in agent_outputs if o.stance == "oppose"]),
                "neutral": len([o for o in agent_outputs if o.stance == "neutral"])
            },
            "avg_confidence": sum(confidences) / len(confidences),
            "median_confidence": statistics.median(confidences),
            "parallel": SYSTEM_CONSTRAINTS.get("max_concurrent_agents", 1) > 1 and len(agent_outputs) > 1,
            "max_concurrency": min(SYSTEM_CONSTRAINTS.get("max_concurrent_agents", 1), len(agent_outputs)),
            "runtime_ms": max(o.duration_ms for o in agent_outputs),
            "novelty_pressure": novelty_pressure,
            "agents": [o.to_dict() for o in agent_outputs]
        }


if __name__ == "__main__":
    # Test multi-agent system
    print("Testing Multi-Agent System")
    print("=" * 60)
    
    from epistemic_engine import ExplanationHypothesis
    
    # Mock hypothesis
    hypothesis = ExplanationHypothesis(
        central_hypothesis="Overparameterization improves generalization in deep learning",
        supporting_claims=["Double descent phenomenon", "Implicit regularization"],
        assumptions=["Sufficient data available"],
        expected_counterclaims=["Overfitting concerns", "Classical bias-variance tradeoff"]
    )
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    print(f"Initialized {len(orchestrator.agents)} agents:")
    for agent in orchestrator.agents:
        print(f"  - {agent.name}: {agent.objective}")
