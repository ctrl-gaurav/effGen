"""
Intelligent routing system for sub-agent decisions in effGen.

The router analyzes tasks and makes intelligent decisions about:
- Whether to use sub-agents at all
- How many sub-agents are needed
- What specializations are required
- Optimal execution strategy (parallel/sequential/hierarchical/hybrid)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ._i18n import fold
from .complexity_analyzer import ComplexityAnalyzer, ComplexityScore
from .decomposition_engine import DecompositionEngine, TaskStructure
from .task import SubTask

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Available routing strategies for task execution."""
    SINGLE_AGENT = "single_agent"
    PARALLEL_SUB_AGENTS = "parallel_sub_agents"
    SEQUENTIAL_SUB_AGENTS = "sequential_sub_agents"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"


@dataclass
class RoutingDecision:
    """
    Result of routing analysis.

    Attributes:
        use_sub_agents: Whether to use sub-agents
        strategy: Chosen routing strategy
        num_sub_agents: Number of sub-agents needed
        specializations: Required specializations
        decomposition: List of subtasks (if applicable)
        confidence_score: Confidence in this decision (0-1)
        complexity_score: Full complexity analysis
        reasoning: Explanation of decision
        metadata: Additional routing metadata
    """
    use_sub_agents: bool
    strategy: RoutingStrategy
    num_sub_agents: int = 0
    specializations: list[str] = field(default_factory=list)
    decomposition: list[SubTask] = field(default_factory=list)
    confidence_score: float = 0.0
    complexity_score: ComplexityScore | None = None
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "use_sub_agents": self.use_sub_agents,
            "strategy": self.strategy.value,
            "num_sub_agents": self.num_sub_agents,
            "specializations": self.specializations,
            "decomposition": [
                {
                    "id": st.id,
                    "description": st.description,
                    "specialization": st.required_specialization,
                    "depends_on": st.depends_on
                } for st in self.decomposition
            ],
            "confidence_score": round(self.confidence_score, 3),
            "complexity_score": self.complexity_score.to_dict() if self.complexity_score else None,
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }


class SubAgentRouter:
    """
    Intelligent router to decide sub-agent usage strategy.

    The router analyzes tasks and makes decisions about:
    - Should we use sub-agents at all?
    - How many sub-agents are needed?
    - What specializations are required?
    - Should execution be parallel or sequential?
    - What's the optimal decomposition strategy?
    """

    DEFAULT_CONFIG = {
        "complexity_threshold": 7.0,
        "confidence_threshold": 0.7,
        "min_subtasks": 2,
        "max_subtasks": 5,
        "prefer_parallel": True,
        "max_parallel_agents": 5,
        "allow_hierarchical": True,
        "max_depth": 2,
        "hybrid_threshold": 8.5,
        "keyword_triggers": [
            "research and analyze",
            "compare multiple",
            "comprehensive",
            "gather data from various",
            "create a report on",
            "investigate and summarize",
            "build and test",
            "analyze trends",
            "multi-step",
            "end-to-end",
            # Spanish equivalents. Compared against text folded to
            # unaccented lowercase (see effgen.core._i18n), so they are
            # written folded too. No language-detection step: see
            # docs/i18n-notes.md for why that was rejected.
            "investiga y analiza",
            "compara multiples",
            "integral",
            "recopila datos de varias",
            "crea un informe sobre",
            "investiga y resume",
            "construye y prueba",
            "analiza tendencias",
            "multi-paso",
            "de principio a fin"
        ]
    }

    def __init__(self, config: dict[str, Any] | None = None,
                 llm_client: Any | None = None) -> None:
        """
        Initialize router.

        Args:
            config: Optional configuration overrides
            llm_client: Optional LLM client for decomposition
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.complexity_analyzer = ComplexityAnalyzer(
            config=self.config.get("complexity_analyzer", {})
        )
        self.decomposition_engine = DecompositionEngine(
            llm_client=llm_client,
            config={
                "max_subtasks": self.config["max_subtasks"],
                "min_subtasks": self.config["min_subtasks"]
            }
        )
        self.history = []  # Track routing decisions for learning

    def should_use_sub_agents(self, task: str, context: dict[str, Any] | None = None) -> RoutingDecision:
        """
        Public API to determine if sub-agents should be used.

        Returns a simplified RoutingDecision with just the use_sub_agents flag,
        strategy, and complexity analysis.

        Args:
            task: Task description
            context: Optional context for analysis

        Returns:
            RoutingDecision with use_sub_agents determination
        """
        context = context or {}

        # Analyze complexity
        complexity_score = self.complexity_analyzer.analyze(task, context)

        # Check if sub-agents should be used
        use_sub_agents = self._should_use_sub_agents(complexity_score, task)

        # Determine strategy
        if use_sub_agents:
            task_structure = self.decomposition_engine.analyze_task_structure(task)
            strategy = self._determine_strategy(task_structure, complexity_score)
        else:
            strategy = RoutingStrategy.SINGLE_AGENT

        # Generate reasoning
        num_requirements = complexity_score.breakdown.get("num_requirements", 1)
        if use_sub_agents:
            reasoning = (
                f"Complexity score: {complexity_score.overall:.2f}. Sub-agents "
                f"recommended: an explicit request, a multi-step keyword phrase, "
                f"or {num_requirements} distinct requirements indicate a compound task."
            )
        else:
            reasoning = (
                f"Complexity score: {complexity_score.overall:.2f}. Single agent "
                f"sufficient: no explicit request, multi-step keyword phrase, or "
                f"multiple distinct requirements — the task is a single unit of work."
            )

        return RoutingDecision(
            use_sub_agents=use_sub_agents,
            strategy=strategy,
            complexity_score=complexity_score,
            reasoning=reasoning,
            confidence_score=0.8 if use_sub_agents else 0.9
        )

    def route(self, task: str, context: dict[str, Any] | None = None) -> RoutingDecision:
        """
        Main routing decision function.

        Args:
            task: Task description (a plain string).
            context: Optional context for routing

        Returns:
            RoutingDecision with strategy and decomposition

        Example::

            from effgen import SubAgentRouter

            decision = SubAgentRouter().route("Research and analyze AI safety trends.")
            print(decision.use_sub_agents, decision.strategy.value)
        """
        if not isinstance(task, str):
            raise TypeError(
                f"SubAgentRouter.route() expects the task as a string; "
                f"got {type(task).__name__}."
            )
        context = context or {}

        # Step 1: Analyze task complexity
        complexity_score = self.complexity_analyzer.analyze(task, context)

        # Step 2: Check if complexity warrants sub-agents
        if not self._should_use_sub_agents(complexity_score, task):
            decision = RoutingDecision(
                use_sub_agents=False,
                strategy=RoutingStrategy.SINGLE_AGENT,
                num_sub_agents=0,
                confidence_score=0.9,
                complexity_score=complexity_score,
                reasoning="Task complexity below threshold or single-agent sufficient"
            )
            self._record_decision(task, decision)
            return decision

        # Step 3: Analyze task structure
        task_structure = self.decomposition_engine.analyze_task_structure(task)

        # Step 4: Determine optimal strategy
        strategy = self._determine_strategy(task_structure, complexity_score)

        # Step 5: Decompose task
        try:
            decomposition = self.decomposition_engine.decompose(
                task,
                strategy.value,
                task_structure,
                context
            )
        except Exception as e:
            # If decomposition fails, fall back to single agent
            logger.warning(f"Decomposition failed: {e}. Falling back to single agent.")
            decision = RoutingDecision(
                use_sub_agents=False,
                strategy=RoutingStrategy.SINGLE_AGENT,
                num_sub_agents=0,
                confidence_score=0.5,
                complexity_score=complexity_score,
                reasoning=f"Decomposition failed: {str(e)}"
            )
            self._record_decision(task, decision)
            return decision

        # Step 6: Identify required specializations
        specializations = self._identify_specializations(decomposition)

        # Step 7: Calculate confidence score
        confidence = self._calculate_confidence(task_structure, complexity_score, decomposition)

        # Step 8: Generate reasoning
        reasoning = self._generate_reasoning(
            complexity_score,
            task_structure,
            strategy,
            len(decomposition)
        )

        # Create decision
        decision = RoutingDecision(
            use_sub_agents=True,
            strategy=strategy,
            num_sub_agents=len(decomposition),
            specializations=specializations,
            decomposition=decomposition,
            confidence_score=confidence,
            complexity_score=complexity_score,
            reasoning=reasoning,
            metadata={
                "task_structure": task_structure.to_dict(),
                "triggers_matched": self._get_matched_triggers(task)
            }
        )

        self._record_decision(task, decision)
        return decision

    def _should_use_sub_agents(self, complexity_score: ComplexityScore, task: str) -> bool:
        """
        Decision criteria for using sub-agents.

        Factors considered, in order:
        1. User-explicit request (always honored)
        2. Presence of keyword triggers (an explicit multi-step phrase)
        3. Number of distinct requirements (structural compoundness)

        The raw complexity ``overall`` score is not used as an independent
        trigger: it is weighted by task length and domain/tool vocabulary
        breadth as much as by actual task structure, so a single dense
        instruction — or a short instruction plus a long pasted reference
        document (a paper abstract, a log excerpt) — can score as "complex"
        without describing more than one thing to do. Decomposition is
        warranted by evidence the task is genuinely *compound*, not merely
        long or information-dense.

        Args:
            complexity_score: Calculated complexity score
            task: Task description

        Returns:
            True if sub-agents should be used
        """
        # User-explicit trigger check — always honor user's request
        if self._check_user_explicit_triggers(task):
            logger.debug("User explicitly requested sub-agents — honoring request.")
            return True

        # Keyword trigger check — an explicit multi-step phrase ("research
        # and analyze", "compare multiple", "comprehensive", ...) is a
        # reliable compound-task signal regardless of the raw score.
        if self._check_keyword_triggers(task):
            return True

        # Multiple requirements check — several questions, numbered/bulleted
        # items, semicolon-separated asks, or "and"-joined requests are a
        # structural signal the task actually has multiple parts.
        num_requirements = complexity_score.breakdown.get("num_requirements", 1)
        if num_requirements >= self.config["min_subtasks"]:
            return True

        return False

    def _check_keyword_triggers(self, task: str) -> bool:
        """Check if task contains keyword triggers."""
        task_lower = fold(task)
        triggers = self.config["keyword_triggers"]
        return any(trigger in task_lower for trigger in triggers)

    def _check_user_explicit_triggers(self, task: str) -> bool:
        """
        Check if user explicitly requested sub-agents in their prompt.

        Uses fuzzy regex matching to handle spelling variations, hyphenation,
        and different phrasings. Matches patterns like:
        - "use sub-agents" / "use subagents" / "use sub agents"
        - "launch 3 more agents" / "launch two agents"
        - "enable sub-agent mode" / "enable subagent routing"
        - "split into sub-tasks" / "break into subtasks"
        - "deploy multiple agents" / "spawn agents"
        """
        task_lower = fold(task)

        # Action verbs that indicate user intent to use sub-agents
        action_verbs = r"(?:use|enable|launch|spawn|deploy|activate|start|trigger|run|create)"
        # Agent-related nouns (flexible hyphenation/spacing)
        agent_nouns = r"(?:sub[\s\-]?agents?|subagents?|multiple\s+agents?|more\s+agents?|additional\s+agents?)"
        # Number patterns (for "use 2 agents", "launch three agents")
        num_pattern = r"(?:\d+|two|three|four|five|several|a\s+few|some)\s+"

        # Pattern 1: action verb + (optional number +) agent noun (sub-agents or multiple agents)
        # e.g., "use sub-agents", "enable subagent", "deploy multiple agents"
        pattern1 = rf"\b{action_verbs}\s+(?:{num_pattern})?{agent_nouns}\b"

        # Pattern 2: action verb + number + bare "agents" (e.g., "launch 3 agents", "use 2 agents")
        pattern2 = rf"\b{action_verbs}\s+{num_pattern}agents?\b"

        # Pattern 3: split/break/decompose into sub-tasks/agents
        pattern3 = r"\b(?:split|break|decompose|divide)\s+(?:\w+\s+)?(?:into|to)\s+(?:sub[\s\-]?tasks?|subtasks?|sub[\s\-]?agents?|agents?|parts?)"

        # Pattern 4: "sub-agent mode/routing/system"
        pattern4 = r"\bsub[\s\-]?agent\s+(?:mode|routing|system|execution)"

        # Pattern 5: explicit number of agents ("with 3 agents", "using two agents")
        pattern5 = rf"\b(?:with|using)\s+{num_pattern}agents?\b"

        # Pattern 6: action verb + bare "agents" (e.g., "spawn agents")
        pattern6 = rf"\b{action_verbs}\s+agents?\b"

        # Spanish equivalents of patterns 1-6 (bilingual by design, see
        # DOMAINS note in complexity_analyzer.py). Word order differs from
        # English in pattern4_es (Spanish commonly puts the noun before the
        # qualifier: "modo subagente" as well as "subagente modo"), so both
        # orderings are matched rather than a single direct translation.
        action_verbs_es = r"(?:usa|activa|lanza|despliega|inicia|dispara|ejecuta|crea)"
        agent_nouns_es = r"(?:sub[\s\-]?agentes?|subagentes?|m[uú]ltiples?\s+agentes?|m[aá]s\s+agentes?|agentes?\s+adicionales)"
        num_pattern_es = r"(?:\d+|dos|tres|cuatro|cinco|varios|unos?\s+pocos|algunos)\s+"

        pattern1_es = rf"\b{action_verbs_es}\s+(?:{num_pattern_es})?{agent_nouns_es}\b"
        pattern2_es = rf"\b{action_verbs_es}\s+{num_pattern_es}agentes?\b"
        pattern3_es = r"\b(?:divide|separa|descompón|descompon|parte)\s+(?:\w+\s+)?en\s+(?:sub[\s\-]?tareas?|subtareas?|sub[\s\-]?agentes?|agentes?|partes?)"
        pattern4_es = r"\b(?:sub[\s\-]?agente\s+(?:modo|enrutamiento|sistema|ejecuci[oó]n)|modo\s+sub[\s\-]?agente)\b"
        pattern5_es = rf"\b(?:con|usando)\s+{num_pattern_es}agentes?\b"
        pattern6_es = rf"\b{action_verbs_es}\s+agentes?\b"

        for pattern in [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6,
                        pattern1_es, pattern2_es, pattern3_es, pattern4_es, pattern5_es, pattern6_es]:
            if re.search(pattern, task_lower):
                return True

        return False

    def _get_matched_triggers(self, task: str) -> list[str]:
        """Get list of matched keyword triggers."""
        # Folded exactly as _check_keyword_triggers folds it, or an accented
        # task would route to sub-agents while reporting no trigger for it.
        task_lower = fold(task)
        triggers = self.config["keyword_triggers"]
        matched = [trigger for trigger in triggers if trigger in task_lower]
        # Also check user-explicit triggers
        if self._check_user_explicit_triggers(task):
            matched.append("[user-explicit] sub-agent request detected")
        return matched

    def _determine_strategy(self,
                           structure: TaskStructure,
                           complexity: ComplexityScore) -> RoutingStrategy:
        """
        Determine the optimal sub-agent strategy.

        Strategies:
        - parallel_sub_agents: Independent tasks executed simultaneously
        - sequential_sub_agents: Dependent tasks executed in order
        - hierarchical: Complex tasks with sub-sub-agents
        - hybrid: Mix of parallel and sequential

        Args:
            structure: Task structure analysis
            complexity: Complexity score

        Returns:
            Chosen routing strategy
        """
        # Check for hierarchical (extremely complex tasks)
        if complexity.overall > 9 and self.config["allow_hierarchical"]:
            return RoutingStrategy.HIERARCHICAL

        # Check for hybrid (very complex with synthesis)
        if (complexity.overall > self.config["hybrid_threshold"] and
            structure.has_synthesis and
            structure.has_data_gathering):
            return RoutingStrategy.HYBRID

        # Check for sequential (has dependencies)
        if structure.has_dependencies:
            return RoutingStrategy.SEQUENTIAL_SUB_AGENTS

        # Check for parallel (multiple questions, no dependencies)
        if structure.parallelizable and structure.has_multiple_questions:
            if self.config["prefer_parallel"]:
                return RoutingStrategy.PARALLEL_SUB_AGENTS

        # Default to parallel for multi-requirement tasks
        if structure.has_multiple_questions:
            return RoutingStrategy.PARALLEL_SUB_AGENTS

        # Default to sequential for complex single tasks
        return RoutingStrategy.SEQUENTIAL_SUB_AGENTS

    def _identify_specializations(self, decomposition: list[SubTask]) -> list[str]:
        """
        Identify which specialized sub-agents are needed.

        Args:
            decomposition: List of subtasks

        Returns:
            List of specialization names (research, coding, analysis, synthesis)
        """
        specializations = set()

        for subtask in decomposition:
            # Use explicit specialization if available
            if subtask.required_specialization:
                specializations.add(subtask.required_specialization)
            else:
                # Infer from description
                desc_lower = subtask.description.lower()

                if any(word in desc_lower for word in
                       ["search", "find", "research", "gather", "investigate"]):
                    specializations.add("research")

                if any(word in desc_lower for word in
                       ["code", "program", "implement", "script", "debug"]):
                    specializations.add("coding")

                if any(word in desc_lower for word in
                       ["analyze", "calculate", "compute", "evaluate", "assess"]):
                    specializations.add("analysis")

                if any(word in desc_lower for word in
                       ["summarize", "combine", "integrate", "synthesize", "report"]):
                    specializations.add("synthesis")

        # If no specialization identified, default to general
        if not specializations:
            specializations.add("general")

        return sorted(specializations)

    def _calculate_confidence(self,
                             structure: TaskStructure,
                             complexity: ComplexityScore,
                             decomposition: list[SubTask]) -> float:
        """
        Calculate confidence in routing decision.

        Args:
            structure: Task structure
            complexity: Complexity score
            decomposition: Generated decomposition

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.7  # Base confidence

        # Boost confidence for clear indicators
        if structure.has_multiple_questions:
            confidence += 0.1

        if structure.parallelizable:
            confidence += 0.05

        if complexity.overall > 8.0:
            confidence += 0.1

        # Reduce confidence for edge cases
        if len(decomposition) < self.config["min_subtasks"]:
            confidence -= 0.1

        if complexity.overall < self.config["complexity_threshold"] * 1.1:
            confidence -= 0.05

        # Ensure valid range
        return max(0.0, min(1.0, confidence))

    def _generate_reasoning(self,
                           complexity: ComplexityScore,
                           structure: TaskStructure,
                           strategy: RoutingStrategy,
                           num_subtasks: int) -> str:
        """
        Generate human-readable reasoning for decision.

        Args:
            complexity: Complexity score
            structure: Task structure
            strategy: Chosen strategy
            num_subtasks: Number of subtasks

        Returns:
            Reasoning text
        """
        reasons = []

        # Complexity reason
        reasons.append(f"Task complexity: {complexity.overall:.1f}/10")

        # Structure reasons
        if structure.has_multiple_questions:
            reasons.append("Multiple distinct requirements detected")

        if structure.parallelizable:
            reasons.append("Tasks can be parallelized")
        elif structure.has_dependencies:
            reasons.append("Sequential dependencies detected")

        # Strategy reason
        strategy_reasons = {
            RoutingStrategy.PARALLEL_SUB_AGENTS: "Using parallel execution for efficiency",
            RoutingStrategy.SEQUENTIAL_SUB_AGENTS: "Using sequential execution due to dependencies",
            RoutingStrategy.HIERARCHICAL: "Using hierarchical decomposition for complex task",
            RoutingStrategy.HYBRID: "Using hybrid approach for optimal performance"
        }
        reasons.append(strategy_reasons.get(strategy, "Using selected strategy"))

        # Subtask count
        reasons.append(f"Decomposed into {num_subtasks} subtasks")

        return ". ".join(reasons) + "."

    def _record_decision(self, task: str, decision: RoutingDecision):
        """
        Record routing decision for learning and analysis.

        Args:
            task: Original task
            decision: Routing decision
        """
        self.history.append({
            "task": task,
            "decision": decision.to_dict(),
            "timestamp": None  # Could add timestamp
        })

        # Keep history bounded
        max_history = 1000
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]

    def get_decision_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get routing decision history.

        Args:
            limit: Optional limit on number of results

        Returns:
            List of historical decisions
        """
        if limit:
            return self.history[-limit:]
        return self.history

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about routing decisions.

        Returns:
            Statistics dictionary
        """
        if not self.history:
            return {
                "total_decisions": 0,
                "sub_agent_usage_rate": 0.0,
                "avg_complexity": 0.0,
                "strategy_distribution": {}
            }

        total = len(self.history)
        sub_agent_count = sum(
            1 for h in self.history
            if h["decision"]["use_sub_agents"]
        )

        complexities = [
            h["decision"]["complexity_score"]["overall"]
            for h in self.history
            if h["decision"].get("complexity_score")
        ]

        strategies = {}
        for h in self.history:
            strategy = h["decision"]["strategy"]
            strategies[strategy] = strategies.get(strategy, 0) + 1

        return {
            "total_decisions": total,
            "sub_agent_usage_rate": sub_agent_count / total if total > 0 else 0.0,
            "avg_complexity": sum(complexities) / len(complexities) if complexities else 0.0,
            "strategy_distribution": strategies,
            "avg_subtasks": sum(
                h["decision"]["num_sub_agents"]
                for h in self.history
                if h["decision"]["use_sub_agents"]
            ) / max(sub_agent_count, 1)
        }

    def analyze_routing_patterns(self) -> dict[str, Any]:
        """
        Analyze historical routing patterns to identify trends.

        Returns:
            Analysis of routing patterns
        """
        if not self.history:
            return {"error": "No routing history available"}

        stats = self.get_statistics()

        # Analyze decision patterns
        patterns = {
            "total_decisions": len(self.history),
            "statistics": stats,
            "trends": self._identify_trends(),
            "accuracy_metrics": self._calculate_accuracy_metrics(),
            "optimization_suggestions": self._generate_optimization_suggestions()
        }

        return patterns

    def _identify_trends(self) -> dict[str, Any]:
        """Identify trends in routing decisions."""
        if len(self.history) < 5:
            return {"message": "Insufficient data for trend analysis"}

        recent_decisions = self.history[-20:]

        # Calculate trend in complexity scores
        complexities = [
            d["decision"]["complexity_score"]["overall"]
            for d in recent_decisions
            if d["decision"].get("complexity_score")
        ]

        # Calculate trend in sub-agent usage
        sub_agent_usage = [
            1 if d["decision"]["use_sub_agents"] else 0
            for d in recent_decisions
        ]

        return {
            "average_complexity_recent": sum(complexities) / len(complexities) if complexities else 0,
            "sub_agent_usage_rate_recent": sum(sub_agent_usage) / len(sub_agent_usage) if sub_agent_usage else 0,
            "trend_direction": self._calculate_trend_direction(complexities),
            "sample_size": len(recent_decisions)
        }

    def _calculate_trend_direction(self, values: list[float]) -> str:
        """Calculate if trend is increasing, decreasing, or stable."""
        if len(values) < 3:
            return "insufficient_data"

        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)

        diff = second_half - first_half

        if abs(diff) < 0.5:
            return "stable"
        elif diff > 0:
            return "increasing"
        else:
            return "decreasing"

    def _calculate_accuracy_metrics(self) -> dict[str, Any]:
        """Calculate routing accuracy metrics."""
        # Placeholder for actual accuracy calculation
        # Would require ground truth or feedback data
        return {
            "note": "Accuracy metrics require feedback data",
            "confidence_average": sum(
                d["decision"].get("confidence_score", 0)
                for d in self.history
            ) / len(self.history) if self.history else 0
        }

    def _generate_optimization_suggestions(self) -> list[str]:
        """Generate suggestions for optimizing routing decisions."""
        suggestions = []
        stats = self.get_statistics()

        # Check if threshold might be too low/high
        usage_rate = stats.get("sub_agent_usage_rate", 0)

        if usage_rate > 0.8:
            suggestions.append(
                "Sub-agent usage rate is very high (>80%). Consider increasing "
                "complexity threshold to reduce overhead for simpler tasks."
            )
        elif usage_rate < 0.2:
            suggestions.append(
                "Sub-agent usage rate is very low (<20%). Consider decreasing "
                "complexity threshold to better leverage parallel processing."
            )

        # Check strategy distribution
        strategy_dist = stats.get("strategy_distribution", {})
        if len(strategy_dist) == 1:
            suggestions.append(
                "Only one routing strategy is being used. Ensure task variety "
                "or review routing logic for proper strategy selection."
            )

        # Check average subtasks
        avg_subtasks = stats.get("avg_subtasks", 0)
        if avg_subtasks > 4:
            suggestions.append(
                "Average subtask count is high. Consider if tasks are being "
                "over-decomposed, which may increase coordination overhead."
            )
        elif avg_subtasks < 2 and usage_rate > 0.5:
            suggestions.append(
                "Low subtask count despite high sub-agent usage. Review "
                "decomposition logic for better task breakdown."
            )

        if not suggestions:
            suggestions.append("Routing patterns appear optimal based on current data.")

        return suggestions

    def export_decision_log(self, filepath: str, format: str = "json") -> None:
        """
        Export routing decision history to file.

        Args:
            filepath: Path to export file
            format: Export format (json, csv)
        """
        import csv
        import json

        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)

        elif format == "csv":
            if not self.history:
                return

            with open(filepath, 'w', newline='') as f:
                fieldnames = ["task", "use_sub_agents", "strategy", "num_sub_agents",
                             "complexity_score", "confidence_score"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for entry in self.history:
                    decision = entry["decision"]
                    writer.writerow({
                        "task": entry["task"][:100],
                        "use_sub_agents": decision["use_sub_agents"],
                        "strategy": decision["strategy"],
                        "num_sub_agents": decision["num_sub_agents"],
                        "complexity_score": decision.get("complexity_score", {}).get("overall", 0),
                        "confidence_score": decision["confidence_score"]
                    })
        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear_history(self, keep_recent: int = 0) -> None:
        """
        Clear routing history.

        Args:
            keep_recent: Number of recent entries to keep (0 = clear all)
        """
        if keep_recent > 0:
            self.history = self.history[-keep_recent:]
        else:
            self.history = []

    def get_strategy_recommendations(self, task_characteristics: dict[str, Any]) -> list[str]:
        """
        Get strategy recommendations based on task characteristics.

        Args:
            task_characteristics: Dict with task properties

        Returns:
            List of recommended strategies
        """
        recommendations = []

        has_dependencies = task_characteristics.get("has_dependencies", False)
        num_requirements = task_characteristics.get("num_requirements", 1)
        complexity = task_characteristics.get("complexity", 5.0)
        parallelizable = task_characteristics.get("parallelizable", True)

        # Parallel recommendations
        if parallelizable and num_requirements >= 3 and not has_dependencies:
            recommendations.append({
                "strategy": "parallel_sub_agents",
                "confidence": "high",
                "reason": "Multiple independent requirements can be processed simultaneously"
            })

        # Sequential recommendations
        if has_dependencies or not parallelizable:
            recommendations.append({
                "strategy": "sequential_sub_agents",
                "confidence": "high",
                "reason": "Dependencies require sequential processing"
            })

        # Hierarchical recommendations
        if complexity > 9.0 or num_requirements > 5:
            recommendations.append({
                "strategy": "hierarchical",
                "confidence": "medium",
                "reason": "High complexity benefits from hierarchical coordination"
            })

        # Hybrid recommendations
        if complexity > 8.0 and num_requirements >= 4:
            recommendations.append({
                "strategy": "hybrid",
                "confidence": "medium",
                "reason": "Mix of parallel and sequential processing may be optimal"
            })

        if not recommendations:
            recommendations.append({
                "strategy": "sequential_sub_agents",
                "confidence": "low",
                "reason": "Default fallback strategy"
            })

        return recommendations

    def simulate_routing(self, task: str, override_config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Simulate routing decision without recording to history.

        Args:
            task: Task description
            override_config: Temporary config overrides

        Returns:
            Routing decision and analysis
        """
        # Temporarily save and override config
        original_config = self.config.copy()

        if override_config:
            self.config.update(override_config)

        # Perform routing without recording
        original_history_len = len(self.history)
        decision = self.route(task)

        # Remove the recorded decision
        self.history = self.history[:original_history_len]

        # Restore original config
        self.config = original_config

        return {
            "decision": decision.to_dict(),
            "simulation": True,
            "config_used": override_config or {}
        }

    def __repr__(self) -> str:
        """String representation."""
        return (f"SubAgentRouter(threshold={self.config['complexity_threshold']}, "
                f"decisions={len(self.history)})")
