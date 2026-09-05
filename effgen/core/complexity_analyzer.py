"""
Complexity analysis for task routing in effGen.

Analyzes task complexity to determine if sub-agent decomposition is warranted.
Uses multiple weighted factors to score complexity on a 0-10 scale.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ._i18n import count_and_clauses, fold, fold_keywords

logger = logging.getLogger(__name__)



@lru_cache(maxsize=512)
def _indicator_pattern(indicator: str) -> re.Pattern[str]:
    """Return a whole-word matcher for *indicator*, compiled once per value.

    Args:
        indicator: The literal phrase to look for.

    Returns:
        A compiled pattern matching *indicator* only on word boundaries.
    """
    # \b is only meaningful next to a word character. An indicator that starts
    # or ends with punctuation ("c++") would otherwise never match.
    left = r"\b" if indicator[:1].isalnum() or indicator[:1] == "_" else ""
    right = r"\b" if indicator[-1:].isalnum() or indicator[-1:] == "_" else ""
    return re.compile(f"{left}{re.escape(indicator)}{right}")


def _mentions(task_lower: str, indicators: "Iterable[str]") -> bool:
    """Whether *task_lower* contains any of *indicators* as whole words.

    Substring matching read "api" out of "capital" and "graph" out of
    "paragraph", so ordinary prose was scored as needing tools it did not.

    Args:
        task_lower: The lowercased task text.
        indicators: Phrases that suggest a tool is needed.

    Returns:
        True when at least one indicator appears as a whole word.
    """
    return any(_indicator_pattern(ind).search(task_lower) for ind in indicators)


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
    breakdown: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
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

    # Domain keywords for identifying different knowledge domains.
    #
    # These are matched as WHOLE WORDS (see _mentions), so a Spanish verb needs
    # both the form a description uses and the form an instruction uses —
    # "analizar" and "analiza" — because neither contains the other and a stem
    # covering both would not be a word. Keywords are compared against text
    # folded to unaccented lowercase (effgen.core._i18n), which is applied to
    # the lists too, so they stay spelled correctly here.
    # Bilingual (EN/ES) by design: matching against a merged keyword set
    # avoids needing a language-detection step, which is unreliable on the
    # short queries typical of a complexity router (see docs/i18n-notes.md).
    DOMAINS = fold_keywords({
        "technical": ["code", "programming", "software", "algorithm", "debug", "implement", "script", "api",
                      "código", "programación", "algoritmo", "depurar", "depura",
                      "implementar", "implementa"],
        "research": ["study", "research", "investigate", "analyze", "survey", "review", "literature",
                     "estudio", "investigación", "investigar", "investiga",
                     "analizar", "analiza", "encuesta", "revisión", "literatura"],
        "business": ["market", "sales", "revenue", "business", "strategy", "roi", "profit",
                     "mercado", "ventas", "ingresos", "negocio", "estrategia", "beneficio"],
        "creative": ["design", "create", "write", "compose", "generate", "draft", "brainstorm",
                     "diseño", "diseñar", "diseña", "crear", "crea", "escribir", "escribe",
                      "componer", "compone", "generar", "genera", "borrador"],
        "data": ["data", "statistics", "analytics", "metrics", "dataset", "visualization", "analysis",
                 "datos", "estadísticas", "analítica", "métricas", "conjunto de datos", "visualización", "análisis"],
        "scientific": ["experiment", "hypothesis", "theory", "scientific", "methodology", "findings",
                       "experimento", "hipótesis", "teoría", "científico", "metodología", "hallazgos"],
        "legal": ["legal", "law", "regulation", "compliance", "contract", "policy",
                  "legal", "ley", "regulación", "cumplimiento", "contrato", "política"],
        "financial": ["financial", "accounting", "budget", "investment", "forecast", "valuation",
                      "financiero", "contabilidad", "presupuesto", "inversión", "pronóstico", "valuación"]
    })

    # Tool indicators for different tool types (bilingual EN/ES, see DOMAINS note above)
    TOOL_INDICATORS = fold_keywords({
        "web_search": ["search", "find online", "look up", "google", "web",
                       "buscar", "busca", "búsqueda", "consultar en línea", "web"],
        "code_executor": ["run code", "execute", "test", "python", "script",
                          "ejecutar código", "ejecutar", "ejecuta", "probar", "prueba", "script"],
        "calculator": ["calculate", "compute", "math", "equation", "formula",
                       "calcular", "calcula", "computar", "computa", "matemáticas",
                       "ecuación", "fórmula"],
        "file_ops": ["file", "document", "read", "write", "save", "load",
                     "archivo", "documento", "leer", "lee", "escribir", "escribe",
                     "guardar", "guarda", "cargar", "carga"],
        "api": ["api", "request", "fetch data", "endpoint", "rest",
                "solicitud", "obtener datos", "obtén datos", "endpoint"],
        "database": ["database", "sql", "query", "table", "data",
                     "base de datos", "consulta", "tabla", "datos"],
        "image": ["image", "picture", "photo", "visualization", "chart", "graph",
                  "imagen", "foto", "visualización", "gráfico", "gráfica"],
        "video": ["video", "movie", "stream", "multimedia",
                  "vídeo", "película", "transmisión"]
    })

    # Reasoning complexity indicators (bilingual EN/ES, see DOMAINS note above)
    REASONING_INDICATORS = fold_keywords({
        "simple": ["list", "what is", "define", "show", "display",
                   "lista", "qué es", "define", "muestra", "despliega"],
        "moderate": ["explain", "describe", "how", "summarize", "outline",
                     "explica", "describe", "cómo", "resume", "esquematiza"],
        "complex": ["analyze", "evaluate", "compare", "assess", "critique",
                    "analizar", "analiza", "evaluar", "evalúa", "comparar", "compara", "critica"],
        "very_complex": ["synthesize", "design", "create strategy", "optimize", "architect", "comprehensive",
                          "sintetizar", "sintetiza", "diseñar", "diseña", "crea una estrategia",
                          "optimizar", "optimiza", "arquitectura", "integral"]
    })

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize complexity analyzer.

        Args:
            config: Optional configuration with custom weights
        """
        self.config = config or {}
        self.weights: dict[str, float] = self.config.get("weights", self.DEFAULT_WEIGHTS)

        # Ensure weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            # Normalize weights
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}

    def analyze(self, task: str, context: dict[str, Any] | None = None) -> ComplexityScore:
        """
        Calculate overall complexity score.

        Args:
            task: Task description
            context: Optional context for analysis

        Returns:
            ComplexityScore with detailed breakdown
        """
        task_lower = fold(task)

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

        # Count "and" clauses (heuristic for multiple requirements), scaled
        # by task length. A short task with several "and"s describes several
        # sequential asks ("fetch the data and convert it and email me the
        # result"); a long passage — e.g. one instruction plus a pasted
        # reference document such as a paper abstract — accumulates "and"s in
        # ordinary prose without describing separate requirements. Density,
        # not the raw count, is the signal, so the contribution tapers off
        # for longer text instead of scaling unbounded with length.
        words = task.split()
        # " y " is Spanish for "and" and carries the same signal, but unlike
        # " and " it is ambiguous: an English task naming an axis or a
        # variable ("the y axis and the x axis") contains it without
        # describing a second requirement, and counting it there inflated the
        # requirement count of English text. It is counted only where a
        # Spanish function word says the sentence is Spanish, which no
        # English task carries by accident.
        raw_and_clauses = count_and_clauses(task)
        num_and_clauses = round(raw_and_clauses * min(1.0, 30 / max(len(words), 1)))

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
            1 for indicators in self.TOOL_INDICATORS.values()
            if _mentions(task_lower, indicators)
        )

        # Scale: 1 tool = 2.5, 4+ tools = 10.0
        return min(10.0, tools_needed * 2.5)

    def _identify_tools(self, task_lower: str) -> list:
        """Identify which tools are likely needed."""
        tools = []
        for tool, indicators in self.TOOL_INDICATORS.items():
            if _mentions(task_lower, indicators):
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

    def analyze_complexity(self, task: str, context: dict[str, Any] | None = None) -> ComplexityScore:
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

    def get_insights(self, complexity_score: ComplexityScore) -> dict[str, Any]:
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

    def _identify_dominant_factors(self, score: ComplexityScore) -> list[str]:
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

    def _generate_recommendations(self, score: ComplexityScore) -> list[str]:
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

    def _estimate_resources(self, score: ComplexityScore) -> dict[str, Any]:
        """Estimate resource requirements based on complexity."""
        # Base estimates
        base_time = 10  # seconds
        base_tokens = 500

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

    def _identify_risk_factors(self, score: ComplexityScore) -> list[dict[str, str]]:
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

    def compare_tasks(self, task1: str, task2: str) -> dict[str, Any]:
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

    def calibrate_weights(self, training_data: list[dict[str, Any]]) -> dict[str, float]:
        """
        Calibrate the analyzer's feature weights from labelled examples.

        Each example is a dict with a ``task`` string and an
        ``actual_complexity`` score in ``[0, 10]``. Weights for features that
        correlate positively with the observed complexity are nudged up and
        re-normalised; features that don't correlate are left alone. Returns the
        updated weight mapping.

        Raises ``ValueError`` if fewer than 10 labelled samples are supplied
        (too few for a stable estimate) — a clear, actionable error rather
        than a silent no-op.
        """
        if len(training_data) < 10:
            raise ValueError(
                f"calibrate_weights needs at least 10 labelled samples for a "
                f"stable estimate; got {len(training_data)}."
            )

        logger.debug("Calibrating complexity weights from %d samples", len(training_data))

        # Pearson-style correlation between each feature's score and the
        # observed complexity, computed over the provided samples.
        feature_scores: dict[str, list[float]] = {k: [] for k in self.weights}
        actuals: list[float] = []
        for sample in training_data:
            score = self.analyze(str(sample.get("task", "")))
            actuals.append(float(sample.get("actual_complexity", score.overall)))
            for feat in self.weights:
                feature_scores[feat].append(float(getattr(score, feat, 0.0)))

        def _corr(xs: list[float], ys: list[float]) -> float:
            n = len(xs)
            if n < 2:
                return 0.0
            mx, my = sum(xs) / n, sum(ys) / n
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            vx = sum((x - mx) ** 2 for x in xs)
            vy = sum((y - my) ** 2 for y in ys)
            denom = (vx * vy) ** 0.5
            return cov / denom if denom else 0.0

        # Nudge weights toward features that track the actual complexity.
        for feat in self.weights:
            corr = _corr(feature_scores[feat], actuals)
            if corr > 0:
                self.weights[feat] *= (1.0 + 0.25 * corr)

        # Re-normalise so the weights still sum to 1.
        total = sum(self.weights.values())
        if total > 0:
            for feat in self.weights:
                self.weights[feat] /= total

        logger.debug("Complexity weights recalibrated: %s", self.weights)
        return self.weights

    def batch_analyze(self, tasks: list[str]) -> list[ComplexityScore]:
        """
        Analyze multiple tasks efficiently.

        Args:
            tasks: List of task descriptions

        Returns:
            List of ComplexityScore objects
        """
        return [self.analyze(task) for task in tasks]

    def get_complexity_distribution(self, scores: list[ComplexityScore]) -> dict[str, Any]:
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
