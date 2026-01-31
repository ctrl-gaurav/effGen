"""
Intelligent routing system for sub-agent decisions in effGen.

The router analyzes tasks and makes intelligent decisions about:
- Whether to use sub-agents at all
- How many sub-agents are needed
- What specializations are required
- Optimal execution strategy (parallel/sequential/hierarchical/hybrid)
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .complexity_analyzer import ComplexityAnalyzer, ComplexityScore
from .decomposition_engine import DecompositionEngine, TaskStructure
from .task import SubTask


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
    specializations: List[str] = field(default_factory=list)
    decomposition: List[SubTask] = field(default_factory=list)
    confidence_score: float = 0.0
    complexity_score: Optional[ComplexityScore] = None
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
            "end-to-end"
        ]
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 llm_client: Optional[Any] = None):
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

    def should_use_sub_agents(self, task: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
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
        reasoning = (
            f"Complexity score: {complexity_score.overall:.2f}. "
            f"{'Sub-agents recommended' if use_sub_agents else 'Single agent sufficient'} "
            f"based on complexity threshold of {self.config['complexity_threshold']}."
        )

        return RoutingDecision(
            use_sub_agents=use_sub_agents,
            strategy=strategy,
            complexity_score=complexity_score,
            reasoning=reasoning,
            confidence_score=0.8 if use_sub_agents else 0.9
        )

    def route(self, task: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Main routing decision function.

        Args:
            task: Task description
            context: Optional context for routing

        Returns:
            RoutingDecision with strategy and decomposition
        """
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
            print(f"Decomposition failed: {e}. Falling back to single agent.")
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

        Factors considered:
        1. Complexity score vs threshold
        2. Presence of keyword triggers
        3. Task length and structure
        4. Number of requirements

        Args:
            complexity_score: Calculated complexity score
            task: Task description

        Returns:
            True if sub-agents should be used
        """
        # Complexity threshold check
        threshold = self.config["complexity_threshold"]
        if complexity_score.overall < threshold:
            # Check if triggers present that override threshold
            if not self._check_keyword_triggers(task):
                return False

        # Keyword trigger check
        if self._check_keyword_triggers(task):
            return True

        # Multiple requirements check
        num_requirements = complexity_score.breakdown.get("num_requirements", 1)
        if num_requirements >= self.config["min_subtasks"]:
            return True

        # Task length check (very long tasks benefit from decomposition)
        word_count = complexity_score.breakdown.get("word_count", 0)
        if word_count > 100:
            return True

        # High complexity check (override threshold with margin)
        if complexity_score.overall >= threshold * 0.9:
            return True

        return False

    def _check_keyword_triggers(self, task: str) -> bool:
        """Check if task contains keyword triggers."""
        task_lower = task.lower()
        triggers = self.config["keyword_triggers"]
        return any(trigger in task_lower for trigger in triggers)

    def _get_matched_triggers(self, task: str) -> List[str]:
        """Get list of matched keyword triggers."""
        task_lower = task.lower()
        triggers = self.config["keyword_triggers"]
        return [trigger for trigger in triggers if trigger in task_lower]

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

    def _identify_specializations(self, decomposition: List[SubTask]) -> List[str]:
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

        return sorted(list(specializations))

    def _calculate_confidence(self,
                             structure: TaskStructure,
                             complexity: ComplexityScore,
                             decomposition: List[SubTask]) -> float:
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

    def get_decision_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
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

    def get_statistics(self) -> Dict[str, Any]:
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

    def analyze_routing_patterns(self) -> Dict[str, Any]:
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

    def _identify_trends(self) -> Dict[str, Any]:
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

    def _calculate_trend_direction(self, values: List[float]) -> str:
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

    def _calculate_accuracy_metrics(self) -> Dict[str, Any]:
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

    def _generate_optimization_suggestions(self) -> List[str]:
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

    def export_decision_log(self, filepath: str, format: str = "json"):
        """
        Export routing decision history to file.

        Args:
            filepath: Path to export file
            format: Export format (json, csv)
        """
        import json
        import csv

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

    def clear_history(self, keep_recent: int = 0):
        """
        Clear routing history.

        Args:
            keep_recent: Number of recent entries to keep (0 = clear all)
        """
        if keep_recent > 0:
            self.history = self.history[-keep_recent:]
        else:
            self.history = []

    def get_strategy_recommendations(self, task_characteristics: Dict[str, Any]) -> List[str]:
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

    def simulate_routing(self, task: str, override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
