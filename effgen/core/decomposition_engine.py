"""
Task decomposition engine for effGen.

Uses LLM to intelligently decompose complex tasks into subtasks based on
task structure and chosen strategy.
"""

import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .task import SubTask, TaskPriority, TaskStatus


@dataclass
class TaskStructure:
    """
    Analysis of task structure.

    Attributes:
        has_multiple_questions: Task contains multiple questions
        has_data_gathering: Requires gathering data
        has_analysis: Requires analysis
        has_synthesis: Requires synthesis
        has_dependencies: Has sequential dependencies
        parallelizable: Can be executed in parallel
        metadata: Additional structural metadata
    """
    has_multiple_questions: bool = False
    has_data_gathering: bool = False
    has_analysis: bool = False
    has_synthesis: bool = False
    has_dependencies: bool = False
    parallelizable: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_multiple_questions": self.has_multiple_questions,
            "has_data_gathering": self.has_data_gathering,
            "has_analysis": self.has_analysis,
            "has_synthesis": self.has_synthesis,
            "has_dependencies": self.has_dependencies,
            "parallelizable": self.parallelizable,
            "metadata": self.metadata
        }


class DecompositionEngine:
    """
    Break down complex tasks into subtasks.

    Uses LLM to intelligently decompose based on:
    - Task structure
    - Chosen strategy (parallel, sequential, hierarchical, hybrid)
    - Available resources
    """

    # Strategy-specific prompts
    PARALLEL_PROMPT_TEMPLATE = """You are a task decomposition expert. Break down this complex task into independent subtasks that can be executed in parallel.

Task: {task}

Requirements:
1. Identify 2-5 independent subtasks
2. Each subtask should be self-contained
3. Subtasks should not depend on each other
4. Cover all aspects of the original task
5. Be specific about what each subtask should accomplish

Format your response as JSON:
{{
    "subtasks": [
        {{
            "id": "st_1",
            "description": "Detailed description of what to do",
            "expected_output": "What this subtask should produce",
            "estimated_complexity": 5.0,
            "required_specialization": "research|coding|analysis|synthesis",
            "priority": "high|medium|low"
        }}
    ],
    "reasoning": "Brief explanation of decomposition strategy"
}}

Respond with ONLY the JSON, no additional text."""

    SEQUENTIAL_PROMPT_TEMPLATE = """You are a task decomposition expert. Break down this complex task into dependent subtasks that must be executed in sequence.

Task: {task}

Requirements:
1. Identify 2-5 subtasks in logical order
2. Each subtask builds on previous results
3. Show dependencies clearly
4. Cover all aspects of the original task
5. Be specific about inputs and outputs

Format your response as JSON:
{{
    "subtasks": [
        {{
            "id": "st_1",
            "description": "Detailed description of what to do",
            "depends_on": [],
            "expected_output": "What this subtask should produce",
            "estimated_complexity": 5.0,
            "required_specialization": "research|coding|analysis|synthesis",
            "priority": "high|medium|low"
        }}
    ],
    "reasoning": "Brief explanation of decomposition strategy"
}}

Respond with ONLY the JSON, no additional text."""

    HYBRID_PROMPT_TEMPLATE = """You are a task decomposition expert. Break down this complex task into a hybrid structure with both parallel and sequential subtasks.

Task: {task}

Requirements:
1. Identify subtasks that can run in parallel
2. Identify subtasks that depend on others
3. Optimize for both speed and dependencies
4. Cover all aspects of the original task
5. Be specific about what each subtask should accomplish

Format your response as JSON:
{{
    "subtasks": [
        {{
            "id": "st_1",
            "description": "Detailed description of what to do",
            "depends_on": ["st_0"],
            "expected_output": "What this subtask should produce",
            "estimated_complexity": 5.0,
            "required_specialization": "research|coding|analysis|synthesis",
            "priority": "high|medium|low"
        }}
    ],
    "reasoning": "Brief explanation of decomposition strategy"
}}

Respond with ONLY the JSON, no additional text."""

    def __init__(self, llm_client: Optional[Any] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize decomposition engine.

        Args:
            llm_client: LLM client for generating decompositions
            config: Optional configuration
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.max_subtasks = self.config.get("max_subtasks", 5)
        self.min_subtasks = self.config.get("min_subtasks", 2)

    def analyze_task_structure(self, task: str) -> TaskStructure:
        """
        Analyze the structure of a task.

        Args:
            task: Task description

        Returns:
            TaskStructure with analysis
        """
        task_lower = task.lower()
        structure = TaskStructure()

        # Check for multiple distinct requirements
        structure.has_multiple_questions = self._detect_multiple_requirements(task)

        # Check for data gathering needs
        structure.has_data_gathering = any(word in task_lower for word in
            ["research", "find", "search", "gather", "collect", "fetch", "retrieve"])

        # Check for analysis needs
        structure.has_analysis = any(word in task_lower for word in
            ["analyze", "compare", "evaluate", "assess", "calculate", "compute"])

        # Check for synthesis needs
        structure.has_synthesis = any(word in task_lower for word in
            ["report", "summary", "combine", "integrate", "synthesize", "compile"])

        # Check for dependencies
        structure.has_dependencies = any(word in task_lower for word in
            ["first", "then", "after", "before", "once", "following", "next", "subsequently"])

        # Determine if parallelizable
        structure.parallelizable = (
            structure.has_multiple_questions and
            not structure.has_dependencies
        )

        # Store additional metadata
        structure.metadata = {
            "word_count": len(task.split()),
            "num_questions": task.count("?"),
            "has_numbered_list": bool(re.search(r'\d+[\.)]\s', task))
        }

        return structure

    def _detect_multiple_requirements(self, task: str) -> bool:
        """Detect if task has multiple requirements."""
        # Multiple questions
        if task.count("?") > 1:
            return True

        # Multiple "and" clauses
        if task.lower().count(" and ") > 2:
            return True

        # Numbered or bulleted list
        if re.search(r'\d+[\.)]\s', task) or re.search(r'[-*]\s', task):
            return True

        return False

    def decompose(self,
                  task: str,
                  strategy: str,
                  structure: Optional[TaskStructure] = None,
                  context: Optional[Dict[str, Any]] = None) -> List[SubTask]:
        """
        Decompose task into subtasks.

        Args:
            task: Task description
            strategy: Decomposition strategy (parallel_sub_agents, sequential_sub_agents, hybrid)
            structure: Optional pre-analyzed task structure
            context: Optional context for decomposition

        Returns:
            List of SubTask objects
        """
        # Analyze structure if not provided
        if structure is None:
            structure = self.analyze_task_structure(task)

        # If LLM client available, use it for decomposition
        if self.llm_client:
            return self._llm_decompose(task, strategy, structure, context)
        else:
            # Fallback to rule-based decomposition
            return self._rule_based_decompose(task, strategy, structure)

    def _llm_decompose(self,
                       task: str,
                       strategy: str,
                       structure: TaskStructure,
                       context: Optional[Dict[str, Any]]) -> List[SubTask]:
        """
        Use LLM to decompose task.

        Args:
            task: Task description
            strategy: Decomposition strategy
            structure: Task structure
            context: Optional context

        Returns:
            List of SubTask objects
        """
        # Select appropriate prompt template
        if strategy == "parallel_sub_agents":
            prompt = self.PARALLEL_PROMPT_TEMPLATE.format(task=task)
        elif strategy == "sequential_sub_agents":
            prompt = self.SEQUENTIAL_PROMPT_TEMPLATE.format(task=task)
        elif strategy in ["hybrid", "hierarchical"]:
            prompt = self.HYBRID_PROMPT_TEMPLATE.format(task=task)
        else:
            # Default to parallel
            prompt = self.PARALLEL_PROMPT_TEMPLATE.format(task=task)

        try:
            # Generate decomposition using LLM
            response = self.llm_client.generate(prompt, max_tokens=1000, temperature=0.3)

            # Parse JSON response
            subtasks = self._parse_decomposition(response)

            # Validate and return
            return self._validate_decomposition(subtasks, task, structure)

        except Exception as e:
            # If LLM decomposition fails, fall back to rule-based
            print(f"LLM decomposition failed: {e}. Using rule-based fallback.")
            return self._rule_based_decompose(task, strategy, structure)

    def _parse_decomposition(self, response: str) -> List[SubTask]:
        """
        Parse LLM response into SubTask objects.

        Args:
            response: LLM response text

        Returns:
            List of SubTask objects
        """
        # Extract JSON from response (handle cases where LLM adds extra text)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in LLM response")

        json_str = json_match.group(0)
        data = json.loads(json_str)

        subtasks = []
        for i, subtask_data in enumerate(data.get("subtasks", [])):
            # Parse priority
            priority_str = subtask_data.get("priority", "medium").lower()
            priority_map = {
                "low": TaskPriority.LOW,
                "medium": TaskPriority.MEDIUM,
                "high": TaskPriority.HIGH,
                "critical": TaskPriority.CRITICAL
            }
            priority = priority_map.get(priority_str, TaskPriority.MEDIUM)

            # Create SubTask
            subtask = SubTask(
                id=subtask_data.get("id", f"st_{i+1}"),
                description=subtask_data["description"],
                expected_output=subtask_data["expected_output"],
                estimated_complexity=subtask_data.get("estimated_complexity", 5.0),
                required_specialization=subtask_data.get("required_specialization"),
                depends_on=subtask_data.get("depends_on", []),
                metadata={
                    "reasoning": data.get("reasoning", ""),
                    "priority": priority.name  # Store priority in metadata instead
                }
            )
            subtasks.append(subtask)

        return subtasks

    def _rule_based_decompose(self,
                              task: str,
                              strategy: str,
                              structure: TaskStructure) -> List[SubTask]:
        """
        Fallback rule-based decomposition when LLM is unavailable.

        Args:
            task: Task description
            strategy: Decomposition strategy
            structure: Task structure

        Returns:
            List of SubTask objects
        """
        subtasks = []

        # Simple heuristic: split by "and", numbered items, or questions
        task_parts = []

        # Try to split by numbered items
        numbered_parts = re.split(r'\d+[\.)]\s+', task)
        if len(numbered_parts) > 2:
            task_parts = [p.strip() for p in numbered_parts[1:] if p.strip()]
        else:
            # Try to split by "and"
            and_parts = task.split(" and ")
            if len(and_parts) > 1:
                task_parts = [p.strip() for p in and_parts if p.strip()]
            else:
                # Single task - create default subtasks
                if structure.has_data_gathering and structure.has_synthesis:
                    task_parts = [
                        "Gather and research relevant information",
                        "Analyze and synthesize findings"
                    ]
                else:
                    # Can't decompose meaningfully - return single subtask
                    task_parts = [task]

        # Create SubTask objects
        for i, part in enumerate(task_parts[:self.max_subtasks]):
            # Determine specialization based on keywords
            specialization = self._infer_specialization(part)

            # Determine dependencies for sequential strategy
            depends_on = []
            if strategy == "sequential_sub_agents" and i > 0:
                depends_on = [f"st_{i}"]

            subtask = SubTask(
                id=f"st_{i+1}",
                description=part,
                expected_output=f"Results for: {part[:50]}...",
                estimated_complexity=5.0,
                required_specialization=specialization,
                depends_on=depends_on,
                metadata={"priority": TaskPriority.MEDIUM.name}
            )
            subtasks.append(subtask)

        return subtasks

    def _infer_specialization(self, task_part: str) -> str:
        """Infer required specialization from task description."""
        task_lower = task_part.lower()

        if any(word in task_lower for word in ["search", "find", "research", "gather"]):
            return "research"
        elif any(word in task_lower for word in ["code", "program", "implement", "script"]):
            return "coding"
        elif any(word in task_lower for word in ["analyze", "calculate", "compute", "evaluate"]):
            return "analysis"
        elif any(word in task_lower for word in ["summarize", "combine", "integrate", "synthesize"]):
            return "synthesis"
        else:
            return "general"

    def _validate_decomposition(self,
                                subtasks: List[SubTask],
                                original_task: str,
                                structure: TaskStructure) -> List[SubTask]:
        """
        Validate decomposition and adjust if needed.

        Args:
            subtasks: Proposed subtasks
            original_task: Original task description
            structure: Task structure

        Returns:
            Validated subtasks
        """
        # Ensure we have at least min_subtasks
        if len(subtasks) < self.min_subtasks:
            # If too few subtasks and original task is complex, use rule-based fallback
            if structure.has_multiple_questions or structure.has_data_gathering:
                return self._rule_based_decompose(original_task, "parallel_sub_agents", structure)

        # Ensure we don't exceed max_subtasks
        if len(subtasks) > self.max_subtasks:
            subtasks = subtasks[:self.max_subtasks]

        # Validate dependencies (no circular dependencies)
        subtask_ids = {st.id for st in subtasks}
        for subtask in subtasks:
            # Remove dependencies on non-existent subtasks
            subtask.depends_on = [dep for dep in subtask.depends_on if dep in subtask_ids]

        # Ensure IDs are unique
        seen_ids = set()
        for i, subtask in enumerate(subtasks):
            if subtask.id in seen_ids:
                subtask.id = f"st_{i+1}_unique"
            seen_ids.add(subtask.id)

        return subtasks

    def optimize_decomposition(self, subtasks: List[SubTask]) -> List[SubTask]:
        """
        Optimize subtask decomposition for better execution.

        Args:
            subtasks: List of subtasks

        Returns:
            Optimized list of subtasks
        """
        # Remove duplicates
        unique_subtasks = self._remove_duplicates(subtasks)

        # Merge similar subtasks
        merged_subtasks = self._merge_similar_subtasks(unique_subtasks)

        # Balance complexity
        balanced_subtasks = self._balance_complexity(merged_subtasks)

        # Optimize dependencies
        optimized_subtasks = self._optimize_dependencies(balanced_subtasks)

        return optimized_subtasks

    def _remove_duplicates(self, subtasks: List[SubTask]) -> List[SubTask]:
        """Remove duplicate subtasks based on description similarity."""
        unique = []
        seen_descriptions = set()

        for subtask in subtasks:
            # Simple check - could use semantic similarity in production
            desc_normalized = subtask.description.lower().strip()
            if desc_normalized not in seen_descriptions:
                unique.append(subtask)
                seen_descriptions.add(desc_normalized)

        return unique

    def _merge_similar_subtasks(self, subtasks: List[SubTask]) -> List[SubTask]:
        """Merge subtasks that are very similar."""
        if len(subtasks) <= 1:
            return subtasks

        # Simple implementation - could use embeddings for semantic similarity
        # For now, just return as-is
        return subtasks

    def _balance_complexity(self, subtasks: List[SubTask]) -> List[SubTask]:
        """Balance complexity across subtasks."""
        if not subtasks:
            return subtasks

        # Calculate average complexity
        avg_complexity = sum(st.estimated_complexity for st in subtasks) / len(subtasks)

        # If there's high variance, could split complex tasks or merge simple ones
        # For now, just flag high-complexity tasks
        for subtask in subtasks:
            if subtask.estimated_complexity > avg_complexity * 1.5:
                subtask.metadata["high_complexity"] = True

        return subtasks

    def _optimize_dependencies(self, subtasks: List[SubTask]) -> List[SubTask]:
        """Optimize dependency graph."""
        if not subtasks:
            return subtasks

        # Remove circular dependencies
        subtasks = self._remove_circular_dependencies(subtasks)

        # Topological sort for optimal execution order
        subtasks = self._topological_sort(subtasks)

        return subtasks

    def _remove_circular_dependencies(self, subtasks: List[SubTask]) -> List[SubTask]:
        """Remove circular dependencies from subtask graph."""
        subtask_map = {st.id: st for st in subtasks}

        def has_cycle(start_id: str, visited: set, rec_stack: set) -> bool:
            """Check if there's a cycle starting from start_id."""
            visited.add(start_id)
            rec_stack.add(start_id)

            if start_id in subtask_map:
                for dep_id in subtask_map[start_id].depends_on:
                    if dep_id not in visited:
                        if has_cycle(dep_id, visited, rec_stack):
                            return True
                    elif dep_id in rec_stack:
                        return True

            rec_stack.remove(start_id)
            return False

        # Find and remove cycles
        for subtask in subtasks:
            visited = set()
            rec_stack = set()

            if has_cycle(subtask.id, visited, rec_stack):
                # Remove problematic dependencies
                subtask.depends_on = []
                subtask.metadata["circular_dependency_removed"] = True

        return subtasks

    def _topological_sort(self, subtasks: List[SubTask]) -> List[SubTask]:
        """Sort subtasks topologically based on dependencies."""
        if not subtasks:
            return subtasks

        # Build adjacency list
        subtask_map = {st.id: st for st in subtasks}
        in_degree = {st.id: len(st.depends_on) for st in subtasks}

        # Find nodes with no dependencies
        queue = [st for st in subtasks if in_degree[st.id] == 0]
        sorted_subtasks = []

        while queue:
            current = queue.pop(0)
            sorted_subtasks.append(current)

            # Update in-degrees
            for subtask in subtasks:
                if current.id in subtask.depends_on:
                    in_degree[subtask.id] -= 1
                    if in_degree[subtask.id] == 0:
                        queue.append(subtask)

        # If not all tasks are sorted, there might be cycles
        if len(sorted_subtasks) < len(subtasks):
            # Add remaining tasks
            for subtask in subtasks:
                if subtask not in sorted_subtasks:
                    sorted_subtasks.append(subtask)

        return sorted_subtasks

    def get_decomposition_metrics(self, subtasks: List[SubTask]) -> Dict[str, Any]:
        """
        Get metrics about decomposition quality.

        Args:
            subtasks: List of subtasks

        Returns:
            Metrics dictionary
        """
        if not subtasks:
            return {"error": "No subtasks provided"}

        # Dependency analysis
        max_depth = self._calculate_dependency_depth(subtasks)
        avg_dependencies = sum(len(st.depends_on) for st in subtasks) / len(subtasks)

        # Complexity analysis
        complexities = [st.estimated_complexity for st in subtasks]
        avg_complexity = sum(complexities) / len(complexities)
        complexity_variance = sum((c - avg_complexity) ** 2 for c in complexities) / len(complexities)

        # Specialization analysis
        specializations = {}
        for st in subtasks:
            spec = st.required_specialization or "general"
            specializations[spec] = specializations.get(spec, 0) + 1

        return {
            "num_subtasks": len(subtasks),
            "max_dependency_depth": max_depth,
            "avg_dependencies_per_task": round(avg_dependencies, 2),
            "avg_complexity": round(avg_complexity, 2),
            "complexity_variance": round(complexity_variance, 2),
            "specializations_distribution": specializations,
            "parallelization_potential": len([st for st in subtasks if not st.depends_on]) / len(subtasks)
        }

    def _calculate_dependency_depth(self, subtasks: List[SubTask]) -> int:
        """Calculate maximum depth of dependency chain."""
        subtask_map = {st.id: st for st in subtasks}

        def get_depth(subtask_id: str, memo: Dict[str, int]) -> int:
            if subtask_id in memo:
                return memo[subtask_id]

            if subtask_id not in subtask_map:
                return 0

            subtask = subtask_map[subtask_id]
            if not subtask.depends_on:
                memo[subtask_id] = 1
                return 1

            max_dep_depth = max(get_depth(dep_id, memo) for dep_id in subtask.depends_on)
            memo[subtask_id] = max_dep_depth + 1
            return max_dep_depth + 1

        memo = {}
        return max(get_depth(st.id, memo) for st in subtasks) if subtasks else 0

    def visualize_decomposition(self, subtasks: List[SubTask]) -> str:
        """
        Create a text-based visualization of task decomposition.

        Args:
            subtasks: List of subtasks

        Returns:
            ASCII visualization
        """
        lines = ["Task Decomposition:"]
        lines.append("=" * 60)

        for i, subtask in enumerate(subtasks, 1):
            # Basic info
            lines.append(f"\n{i}. [{subtask.id}] {subtask.description[:50]}...")
            lines.append(f"   Complexity: {subtask.estimated_complexity:.1f}/10")

            if subtask.required_specialization:
                lines.append(f"   Specialization: {subtask.required_specialization}")

            if subtask.depends_on:
                lines.append(f"   Depends on: {', '.join(subtask.depends_on)}")

            lines.append(f"   Expected: {subtask.expected_output[:40]}...")

        lines.append("\n" + "=" * 60)

        # Add metrics
        metrics = self.get_decomposition_metrics(subtasks)
        lines.append(f"Total subtasks: {metrics['num_subtasks']}")
        lines.append(f"Average complexity: {metrics['avg_complexity']}")
        lines.append(f"Max dependency depth: {metrics['max_dependency_depth']}")
        lines.append(f"Parallelization potential: {metrics['parallelization_potential']:.1%}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return (f"DecompositionEngine(max_subtasks={self.max_subtasks}, "
                f"has_llm={self.llm_client is not None})")
