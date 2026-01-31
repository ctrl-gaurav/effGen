"""
Complexity analysis for task routing in effGen.

Analyzes task complexity to determine if sub-agent decomposition is warranted.
Uses multiple weighted factors to score complexity on a 0-10 scale.
"""

import re
from typing import Dict, Optional, Any, List
from dataclasses import dataclass


@dataclass
class ComplexityScore:
    """
    Detailed complexity score breakdown.

    Attributes:
        overall: Overall weighted complexity score (0-10)
        task_length: Score based on task length
        num_requirements: Score based on number of requirements
        domain_breadth: Score based on domain breadth
        tool_requirements: Score based on tools needed
        reasoning_depth: Score based on reasoning complexity
        breakdown: Detailed score breakdown
    """
    overall: float
    task_length: float
    num_requirements: float
    domain_breadth: float
    tool_requirements: float
    reasoning_depth: float
    breakdown: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": round(self.overall, 2),
            "task_length": round(self.task_length, 2),
            "num_requirements": round(self.num_requirements, 2),
            "domain_breadth": round(self.domain_breadth, 2),
            "tool_requirements": round(self.tool_requirements, 2),
            "reasoning_depth": round(self.reasoning_depth, 2),
            "breakdown": self.breakdown
        }


class ComplexityAnalyzer:
    """
    Analyze task complexity to inform routing decisions.

    Uses multiple factors to score complexity on 0-10 scale:
    - Task length (word count)
    - Number of distinct requirements
    - Domain breadth (technical, research, business, etc.)
    - Tool requirements (estimated number of tools needed)
    - Reasoning depth (simple to very complex)
    """

    DEFAULT_WEIGHTS = {
        "task_length": 0.15,
        "num_requirements": 0.25,
        "domain_breadth": 0.20,
        "tool_requirements": 0.20,
        "reasoning_depth": 0.20
    }

    # Domain keywords for identifying different knowledge domains
    DOMAINS = {
        "technical": ["code", "programming", "software", "algorithm", "debug", "implement", "script", "api"],
        "research": ["study", "research", "investigate", "analyze", "survey", "review", "literature"],
        "business": ["market", "sales", "revenue", "business", "strategy", "roi", "profit"],
        "creative": ["design", "create", "write", "compose", "generate", "draft", "brainstorm"],
        "data": ["data", "statistics", "analytics", "metrics", "dataset", "visualization", "analysis"],
        "scientific": ["experiment", "hypothesis", "theory", "scientific", "methodology", "findings"],
        "legal": ["legal", "law", "regulation", "compliance", "contract", "policy"],
        "financial": ["financial", "accounting", "budget", "investment", "forecast", "valuation"]
    }

    # Tool indicators for different tool types
    TOOL_INDICATORS = {
        "web_search": ["search", "find online", "look up", "google", "web"],
        "code_executor": ["run code", "execute", "test", "python", "script"],
        "calculator": ["calculate", "compute", "math", "equation", "formula"],
        "file_ops": ["file", "document", "read", "write", "save", "load"],
        "api": ["api", "request", "fetch data", "endpoint", "rest"],
        "database": ["database", "sql", "query", "table", "data"],
        "image": ["image", "picture", "photo", "visualization", "chart", "graph"],
        "video": ["video", "movie", "stream", "multimedia"]
    }

    # Reasoning complexity indicators
    REASONING_INDICATORS = {
        "simple": ["list", "what is", "define", "show", "display"],
        "moderate": ["explain", "describe", "how", "summarize", "outline"],
        "complex": ["analyze", "evaluate", "compare", "assess", "critique"],
        "very_complex": ["synthesize", "design", "create strategy", "optimize", "architect", "comprehensive"]
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize complexity analyzer.

        Args:
            config: Optional configuration with custom weights
        """
        self.config = config or {}
        self.weights = self.config.get("weights", self.DEFAULT_WEIGHTS)

        # Ensure weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            # Normalize weights
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}

    def analyze(self, task: str, context: Optional[Dict[str, Any]] = None) -> ComplexityScore:
        """
        Calculate overall complexity score.

        Args:
            task: Task description
            context: Optional context for analysis

        Returns:
            ComplexityScore with detailed breakdown
        """
        task_lower = task.lower()

        # Calculate individual scores
        task_length_score = self._score_task_length(task)
        requirements_score = self._score_requirements(task)
        domain_breadth_score = self._score_domain_breadth(task_lower)
        tool_requirements_score = self._score_tool_requirements(task_lower)
        reasoning_depth_score = self._score_reasoning_depth(task_lower)

        # Calculate weighted overall score
        scores = {
            "task_length": task_length_score,
            "num_requirements": requirements_score,
            "domain_breadth": domain_breadth_score,
            "tool_requirements": tool_requirements_score,
            "reasoning_depth": reasoning_depth_score
        }

        overall = sum(scores[k] * self.weights[k] for k in scores)
        overall = min(10.0, overall)  # Cap at 10.0

        # Create detailed breakdown
        breakdown = {
            "word_count": len(task.split()),
            "num_questions": task.count("?"),
            "num_requirements": self._count_requirements(task),
            "domains_identified": self._identify_domains(task_lower),
            "tools_needed": self._identify_tools(task_lower),
            "reasoning_level": self._identify_reasoning_level(task_lower)
        }

        return ComplexityScore(
            overall=overall,
            task_length=task_length_score,
            num_requirements=requirements_score,
            domain_breadth=domain_breadth_score,
            tool_requirements=tool_requirements_score,
            reasoning_depth=reasoning_depth_score,
            breakdown=breakdown
        )

    def _score_task_length(self, task: str) -> float:
        """
        Score based on task length (0-10).

        Args:
            task: Task description

        Returns:
            Length score
        """
        word_count = len(task.split())

        if word_count < 20:
            return 2.0
        elif word_count < 50:
            return 4.0
        elif word_count < 100:
            return 6.0
        elif word_count < 200:
            return 8.0
        else:
            return 10.0

    def _score_requirements(self, task: str) -> float:
        """
        Score based on number of distinct requirements (0-10).

        Args:
            task: Task description

        Returns:
            Requirements score
        """
        num_requirements = self._count_requirements(task)
        # Scale: 1 requirement = 2.0, 5+ requirements = 10.0
        return min(10.0, num_requirements * 2.0)

    def _count_requirements(self, task: str) -> int:
        """Count distinct requirements in task."""
        # Count questions
        num_questions = task.count("?")

        # Count "and" clauses (heuristic for multiple requirements)
        num_and_clauses = task.lower().count(" and ")

        # Count numbered items
        num_numbered = len(re.findall(r'\d+[\.)]\s', task))

        # Count bullet points
        num_bullets = len(re.findall(r'[-*]\s', task))

        # Count semicolons (often separate requirements)
        num_semicolons = task.count(";")

        total = num_questions + num_and_clauses + num_numbered + num_bullets + num_semicolons
        return max(1, total)  # At least 1 requirement

    def _score_domain_breadth(self, task_lower: str) -> float:
        """
        Score based on breadth of knowledge domains (0-10).

        Args:
            task_lower: Lowercase task description

        Returns:
            Domain breadth score
        """
        domains_present = sum(
            1 for domain, keywords in self.DOMAINS.items()
            if any(kw in task_lower for kw in keywords)
        )

        # Scale: 1 domain = 2.5, 4+ domains = 10.0
        return min(10.0, domains_present * 2.5)

    def _identify_domains(self, task_lower: str) -> list:
        """Identify which domains are present in task."""
        domains = []
        for domain, keywords in self.DOMAINS.items():
            if any(kw in task_lower for kw in keywords):
                domains.append(domain)
        return domains

    def _score_tool_requirements(self, task_lower: str) -> float:
        """
        Score based on number of tools likely needed (0-10).

        Args:
            task_lower: Lowercase task description

        Returns:
            Tool requirements score
        """
        tools_needed = sum(
            1 for tool, indicators in self.TOOL_INDICATORS.items()
            if any(ind in task_lower for ind in indicators)
        )

        # Scale: 1 tool = 2.5, 4+ tools = 10.0
        return min(10.0, tools_needed * 2.5)

    def _identify_tools(self, task_lower: str) -> list:
        """Identify which tools are likely needed."""
        tools = []
        for tool, indicators in self.TOOL_INDICATORS.items():
            if any(ind in task_lower for ind in indicators):
                tools.append(tool)
        return tools

    def _score_reasoning_depth(self, task_lower: str) -> float:
        """
        Score based on reasoning complexity (0-10).

        Args:
            task_lower: Lowercase task description

        Returns:
            Reasoning depth score
        """
        # Check from most complex to least complex
        if any(ind in task_lower for ind in self.REASONING_INDICATORS["very_complex"]):
            return 9.0
        elif any(ind in task_lower for ind in self.REASONING_INDICATORS["complex"]):
            return 7.0
        elif any(ind in task_lower for ind in self.REASONING_INDICATORS["moderate"]):
            return 5.0
        elif any(ind in task_lower for ind in self.REASONING_INDICATORS["simple"]):
            return 3.0
        else:
            # Default moderate complexity
            return 5.0

    def _identify_reasoning_level(self, task_lower: str) -> str:
        """Identify reasoning complexity level."""
        if any(ind in task_lower for ind in self.REASONING_INDICATORS["very_complex"]):
            return "very_complex"
        elif any(ind in task_lower for ind in self.REASONING_INDICATORS["complex"]):
            return "complex"
        elif any(ind in task_lower for ind in self.REASONING_INDICATORS["moderate"]):
            return "moderate"
        else:
            return "simple"

    def analyze_complexity(self, task: str, context: Optional[Dict[str, Any]] = None) -> ComplexityScore:
        """
        Alias for analyze() method for backwards compatibility.

        Calculate overall complexity score.

        Args:
            task: Task description
            context: Optional context for analysis

        Returns:
            ComplexityScore with detailed breakdown
        """
        return self.analyze(task, context)

    def should_use_sub_agents(self, complexity_score: ComplexityScore,
                             threshold: float = 7.0) -> bool:
        """
        Determine if complexity warrants sub-agent decomposition.

        Args:
            complexity_score: Calculated complexity score
            threshold: Complexity threshold (default: 7.0)

        Returns:
            True if sub-agents should be used
        """
        return complexity_score.overall >= threshold

    def get_insights(self, complexity_score: ComplexityScore) -> Dict[str, Any]:
        """
        Get detailed insights about complexity score.

        Args:
            complexity_score: Calculated complexity score

        Returns:
            Dictionary with insights and recommendations
        """
        insights = {
            "overall_assessment": self._assess_overall_complexity(complexity_score.overall),
            "dominant_factors": self._identify_dominant_factors(complexity_score),
            "recommendations": self._generate_recommendations(complexity_score),
            "expected_resources": self._estimate_resources(complexity_score),
            "risk_factors": self._identify_risk_factors(complexity_score)
        }
        return insights

    def _assess_overall_complexity(self, score: float) -> str:
        """Assess overall complexity level."""
        if score < 3.0:
            return "Low - Simple task suitable for single-agent execution"
        elif score < 5.0:
            return "Low-Medium - May benefit from basic task breakdown"
        elif score < 7.0:
            return "Medium - Consider sub-agent decomposition for efficiency"
        elif score < 8.5:
            return "High - Strongly recommended to use sub-agents"
        else:
            return "Very High - Complex task requiring hierarchical sub-agent approach"

    def _identify_dominant_factors(self, score: ComplexityScore) -> List[str]:
        """Identify which factors contribute most to complexity."""
        factors = {
            "task_length": score.task_length,
            "num_requirements": score.num_requirements,
            "domain_breadth": score.domain_breadth,
            "tool_requirements": score.tool_requirements,
            "reasoning_depth": score.reasoning_depth
        }

        # Sort by score descending
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)

        # Get top 3 factors
        dominant = []
        for factor, value in sorted_factors[:3]:
            if value > 6.0:
                dominant.append(f"{factor.replace('_', ' ').title()} (score: {value:.1f})")

        return dominant if dominant else ["All factors contributing moderately"]

    def _generate_recommendations(self, score: ComplexityScore) -> List[str]:
        """Generate actionable recommendations based on complexity."""
        recommendations = []

        # Task length recommendations
        if score.task_length > 7.0:
            recommendations.append(
                "Consider breaking down the task description into smaller, "
                "more focused subtasks for better processing"
            )

        # Requirements recommendations
        if score.num_requirements > 7.0:
            recommendations.append(
                f"Task has {score.breakdown['num_requirements']} requirements - "
                "use parallel sub-agents to handle multiple requirements simultaneously"
            )

        # Domain breadth recommendations
        if score.domain_breadth > 7.0:
            domains = score.breakdown.get('domains_identified', [])
            recommendations.append(
                f"Task spans {len(domains)} domains ({', '.join(domains)}) - "
                "assign specialized sub-agents for each domain"
            )

        # Tool requirements recommendations
        if score.tool_requirements > 7.0:
            tools = score.breakdown.get('tools_needed', [])
            recommendations.append(
                f"Requires {len(tools)} different tool types - "
                "ensure all necessary tools are available and properly configured"
            )

        # Reasoning depth recommendations
        if score.reasoning_depth > 7.0:
            level = score.breakdown.get('reasoning_level', 'complex')
            recommendations.append(
                f"Task requires {level} reasoning - "
                "use iterative refinement or hierarchical decomposition"
            )

        # Overall complexity recommendations
        if score.overall > 8.5:
            recommendations.append(
                "Very high complexity detected - consider hierarchical decomposition "
                "with manager agent coordinating specialized workers"
            )
        elif score.overall > 7.0:
            recommendations.append(
                "High complexity - use sub-agent decomposition with appropriate "
                "execution strategy (parallel for independent tasks, sequential for dependent)"
            )

        if not recommendations:
            recommendations.append(
                "Complexity is manageable - single agent execution should be sufficient"
            )

        return recommendations

    def _estimate_resources(self, score: ComplexityScore) -> Dict[str, Any]:
        """Estimate resource requirements based on complexity."""
        # Base estimates
        base_time = 10  # seconds
        base_tokens = 500
        base_agents = 1

        # Scale by complexity
        complexity_multiplier = score.overall / 5.0

        estimated_time = base_time * complexity_multiplier
        estimated_tokens = int(base_tokens * complexity_multiplier * 1.5)

        # Determine optimal agent count
        if score.overall < 5.0:
            optimal_agents = 1
        elif score.overall < 7.0:
            optimal_agents = 2
        elif score.overall < 8.5:
            optimal_agents = min(3, score.breakdown.get('num_requirements', 2))
        else:
            optimal_agents = min(5, max(3, score.breakdown.get('num_requirements', 3)))

        return {
            "estimated_execution_time_seconds": round(estimated_time, 1),
            "estimated_token_usage": estimated_tokens,
            "optimal_agent_count": optimal_agents,
            "confidence": "medium" if score.overall > 4.0 else "high"
        }

    def _identify_risk_factors(self, score: ComplexityScore) -> List[Dict[str, str]]:
        """Identify potential risk factors in task execution."""
        risks = []

        # Task length risk
        if score.task_length > 8.0:
            risks.append({
                "factor": "Task Length",
                "severity": "medium",
                "description": "Very long task description may lead to context overflow or loss of focus",
                "mitigation": "Break into smaller chunks or use summarization"
            })

        # Multiple domains risk
        if score.domain_breadth > 8.0:
            risks.append({
                "factor": "Domain Breadth",
                "severity": "high",
                "description": "Task spans many domains, increasing chance of incomplete or shallow coverage",
                "mitigation": "Use specialized sub-agents for each domain"
            })

        # High tool requirements risk
        if score.tool_requirements > 8.0:
            risks.append({
                "factor": "Tool Requirements",
                "severity": "medium",
                "description": "Many tools required, increasing failure points and execution complexity",
                "mitigation": "Implement robust error handling and tool fallbacks"
            })

        # Complex reasoning risk
        if score.reasoning_depth > 8.0:
            risks.append({
                "factor": "Reasoning Depth",
                "severity": "high",
                "description": "Very complex reasoning required, may exceed model capabilities",
                "mitigation": "Use chain-of-thought prompting and iterative refinement"
            })

        # Overall complexity risk
        if score.overall > 9.0:
            risks.append({
                "factor": "Overall Complexity",
                "severity": "critical",
                "description": "Extremely complex task with high failure risk",
                "mitigation": "Consider simplifying task or using hierarchical multi-agent approach"
            })

        return risks

    def compare_tasks(self, task1: str, task2: str) -> Dict[str, Any]:
        """
        Compare complexity of two tasks.

        Args:
            task1: First task description
            task2: Second task description

        Returns:
            Comparison results
        """
        score1 = self.analyze(task1)
        score2 = self.analyze(task2)

        return {
            "task1_complexity": score1.overall,
            "task2_complexity": score2.overall,
            "difference": abs(score1.overall - score2.overall),
            "more_complex": "task1" if score1.overall > score2.overall else "task2",
            "task1_breakdown": score1.to_dict(),
            "task2_breakdown": score2.to_dict(),
            "comparison_summary": self._generate_comparison_summary(score1, score2)
        }

    def _generate_comparison_summary(self, score1: ComplexityScore, score2: ComplexityScore) -> str:
        """Generate human-readable comparison summary."""
        diff = abs(score1.overall - score2.overall)

        if diff < 1.0:
            return "Tasks have similar complexity levels"
        elif diff < 3.0:
            return f"Task {'1' if score1.overall > score2.overall else '2'} is moderately more complex"
        else:
            more_complex = "1" if score1.overall > score2.overall else "2"
            return f"Task {more_complex} is significantly more complex ({diff:.1f} point difference)"

    def calibrate_weights(self, training_data: List[Dict[str, Any]]):
        """
        Calibrate weights based on historical performance data.

        Args:
            training_data: List of dicts with 'task', 'actual_complexity', 'execution_metrics'
        """
        # Placeholder for weight calibration using training data
        # In production, would use optimization algorithm to find best weights
        print(f"Calibrating weights with {len(training_data)} samples...")

        # Simple heuristic: adjust weights based on correlation with actual complexity
        # This is a simplified version - real implementation would use proper optimization

        if len(training_data) < 10:
            print("Warning: Insufficient training data for reliable calibration (need at least 10 samples)")
            return

        # Calculate correlations (simplified)
        # In production: use proper statistical methods
        print("Weight calibration complete (feature in development)")

    def batch_analyze(self, tasks: List[str]) -> List[ComplexityScore]:
        """
        Analyze multiple tasks efficiently.

        Args:
            tasks: List of task descriptions

        Returns:
            List of ComplexityScore objects
        """
        return [self.analyze(task) for task in tasks]

    def get_complexity_distribution(self, scores: List[ComplexityScore]) -> Dict[str, Any]:
        """
        Analyze distribution of complexity scores.

        Args:
            scores: List of complexity scores

        Returns:
            Distribution statistics
        """
        if not scores:
            return {"error": "No scores provided"}

        overall_scores = [s.overall for s in scores]

        return {
            "count": len(scores),
            "mean": sum(overall_scores) / len(overall_scores),
            "min": min(overall_scores),
            "max": max(overall_scores),
            "median": sorted(overall_scores)[len(overall_scores) // 2],
            "distribution": {
                "low (< 5.0)": sum(1 for s in overall_scores if s < 5.0),
                "medium (5.0-7.0)": sum(1 for s in overall_scores if 5.0 <= s < 7.0),
                "high (7.0-9.0)": sum(1 for s in overall_scores if 7.0 <= s < 9.0),
                "very_high (>= 9.0)": sum(1 for s in overall_scores if s >= 9.0)
            }
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"ComplexityAnalyzer(weights={self.weights})"
