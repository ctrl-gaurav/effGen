"""
Prompt Optimizer for Small Language Models

Implements SLM-specific optimization techniques including prompt compression,
instruction clarity enhancement, context management, and format optimization.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelSize(Enum):
    """Model size categories"""
    TINY = "tiny"  # < 1B parameters
    SMALL = "small"  # 1-3B parameters
    MEDIUM = "medium"  # 3-7B parameters
    LARGE = "large"  # 7B+ parameters


@dataclass
class OptimizationConfig:
    """Configuration for prompt optimization"""

    model_size: ModelSize = ModelSize.SMALL
    max_prompt_tokens: int = 1024
    max_context_tokens: int = 2048
    few_shot_examples: int = 2
    chain_max_depth: int = 3
    use_structured_output: bool = True
    compress_aggressively: bool = False
    use_bullet_points: bool = True
    simplify_language: bool = True
    remove_redundancy: bool = True
    target_compression_ratio: float = 0.7  # Target 30% reduction

    @classmethod
    def for_model_size(cls, model_size: ModelSize) -> 'OptimizationConfig':
        """Get optimization config for model size"""
        configs = {
            ModelSize.TINY: cls(
                model_size=ModelSize.TINY,
                max_prompt_tokens=512,
                max_context_tokens=1024,
                few_shot_examples=1,
                chain_max_depth=2,
                use_structured_output=True,
                compress_aggressively=True,
                target_compression_ratio=0.6
            ),
            ModelSize.SMALL: cls(
                model_size=ModelSize.SMALL,
                max_prompt_tokens=1024,
                max_context_tokens=2048,
                few_shot_examples=2,
                chain_max_depth=3,
                use_structured_output=True,
                compress_aggressively=False,
                target_compression_ratio=0.7
            ),
            ModelSize.MEDIUM: cls(
                model_size=ModelSize.MEDIUM,
                max_prompt_tokens=2048,
                max_context_tokens=4096,
                few_shot_examples=3,
                chain_max_depth=5,
                use_structured_output=False,
                compress_aggressively=False,
                target_compression_ratio=0.8
            ),
            ModelSize.LARGE: cls(
                model_size=ModelSize.LARGE,
                max_prompt_tokens=4096,
                max_context_tokens=8192,
                few_shot_examples=5,
                chain_max_depth=10,
                use_structured_output=False,
                compress_aggressively=False,
                target_compression_ratio=0.9
            )
        }
        return configs.get(model_size, cls())


@dataclass
class OptimizationResult:
    """Result of prompt optimization"""

    original_prompt: str
    optimized_prompt: str
    original_tokens: int
    optimized_tokens: int
    compression_ratio: float
    techniques_applied: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_savings(self) -> int:
        """Get token savings"""
        return self.original_tokens - self.optimized_tokens

    def get_compression_percentage(self) -> float:
        """Get compression percentage"""
        return (1.0 - self.compression_ratio) * 100


class PromptOptimizer:
    """
    Optimizes prompts for Small Language Models with techniques:
    - Prompt compression (remove redundancy, use shorter synonyms)
    - Instruction clarity (explicit, structured format)
    - Context management (fit within token limits)
    - Format optimization (bullet points, structured outputs)
    """

    # Synonym replacements for compression
    COMPRESSION_REPLACEMENTS = {
        # Verbose phrases to concise
        r'\bplease\b': '',
        r'\bkindly\b': '',
        r'\bi would like you to\b': '',
        r'\bcan you\b': '',
        r'\bcould you\b': '',
        r'\bwould you\b': '',
        r'\byou should\b': '',
        r'\byou need to\b': '',
        r'\bit is important to\b': '',
        r'\bmake sure to\b': '',
        r'\bin order to\b': 'to',
        r'\bat this point in time\b': 'now',
        r'\bdue to the fact that\b': 'because',
        r'\bfor the purpose of\b': 'for',
        r'\bin the event that\b': 'if',
        r'\bprior to\b': 'before',
        r'\bsubsequent to\b': 'after',
        r'\bin spite of the fact that\b': 'although',
        r'\bas a matter of fact\b': 'actually',
        r'\bat the present time\b': 'now',
        r'\bin the near future\b': 'soon',
        r'\bon a regular basis\b': 'regularly',
        r'\bwith regard to\b': 'about',
        r'\bwith reference to\b': 'about',
        r'\bin relation to\b': 'about',

        # Verbose words to concise
        r'\butilize\b': 'use',
        r'\bimplement\b': 'do',
        r'\bfacilitate\b': 'help',
        r'\bdemonstrate\b': 'show',
        r'\bconsequently\b': 'so',
        r'\btherefore\b': 'so',
        r'\bnevertheless\b': 'but',
        r'\bfurthermore\b': 'also',
        r'\badditionally\b': 'also',
        r'\bsubsequently\b': 'then',
    }

    # Patterns to identify and remove
    REDUNDANT_PATTERNS = [
        r'\s+',  # Multiple spaces
        r'\n\s*\n\s*\n+',  # Multiple blank lines
        r'(?:please\s+)?(?:make sure|ensure|be sure)\s+(?:to\s+)?',  # Redundant politeness
    ]

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimizer

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        logger.info(f"PromptOptimizer initialized for {self.config.model_size.value} models")

    def optimize(
        self,
        prompt: str,
        context: Optional[str] = None,
        preserve_format: bool = False
    ) -> OptimizationResult:
        """
        Optimize a prompt for SLMs

        Args:
            prompt: Original prompt
            context: Optional context to include
            preserve_format: Whether to preserve original formatting

        Returns:
            OptimizationResult with optimized prompt
        """
        original_tokens = self.estimate_token_count(prompt)
        techniques_applied = []

        optimized = prompt

        # 1. Compress prompt
        if self.config.compress_aggressively or original_tokens > self.config.max_prompt_tokens:
            optimized = self.compress_prompt(optimized)
            techniques_applied.append("compression")

        # 2. Simplify language
        if self.config.simplify_language:
            optimized = self.simplify_language(optimized)
            techniques_applied.append("simplification")

        # 3. Remove redundancy
        if self.config.remove_redundancy:
            optimized = self.remove_redundancy(optimized)
            techniques_applied.append("redundancy_removal")

        # 4. Format optimization
        if self.config.use_bullet_points and not preserve_format:
            optimized = self.format_as_bullet_points(optimized)
            techniques_applied.append("bullet_formatting")

        # 5. Structure for clarity
        if self.config.use_structured_output and not preserve_format:
            optimized = self.structure_for_clarity(optimized)
            techniques_applied.append("structured_format")

        # 6. Ensure within context limits
        optimized_tokens = self.estimate_token_count(optimized)
        if optimized_tokens > self.config.max_prompt_tokens:
            optimized = self.truncate_to_limit(optimized, self.config.max_prompt_tokens)
            techniques_applied.append("truncation")
            optimized_tokens = self.estimate_token_count(optimized)

        # Calculate compression ratio
        compression_ratio = optimized_tokens / original_tokens if original_tokens > 0 else 1.0

        result = OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            compression_ratio=compression_ratio,
            techniques_applied=techniques_applied,
            metadata={
                'config': self.config.model_size.value,
                'target_tokens': self.config.max_prompt_tokens
            }
        )

        logger.info(
            f"Optimized prompt: {original_tokens} -> {optimized_tokens} tokens "
            f"({result.get_compression_percentage():.1f}% reduction)"
        )

        return result

    def compress_prompt(self, prompt: str) -> str:
        """
        Compress prompt by removing redundancy and using shorter expressions

        Args:
            prompt: Prompt to compress

        Returns:
            Compressed prompt
        """
        compressed = prompt

        # Apply compression replacements
        for pattern, replacement in self.COMPRESSION_REPLACEMENTS.items():
            compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)

        # Clean up extra whitespace
        compressed = re.sub(r'\s+', ' ', compressed)
        compressed = re.sub(r'\n\s*\n\s*\n+', '\n\n', compressed)

        return compressed.strip()

    def simplify_language(self, prompt: str) -> str:
        """
        Simplify language for better SLM comprehension

        Args:
            prompt: Prompt to simplify

        Returns:
            Simplified prompt
        """
        simplified = prompt

        # Break long sentences
        sentences = re.split(r'([.!?]\s+)', simplified)
        processed_sentences = []

        for sentence in sentences:
            # Skip punctuation separators
            if re.match(r'^[.!?]\s+$', sentence):
                processed_sentences.append(sentence)
                continue

            # Break sentences longer than ~20 words
            words = sentence.split()
            if len(words) > 20:
                # Try to split at conjunctions
                for i, word in enumerate(words):
                    if word.lower() in ['and', 'but', 'or', 'so', 'because']:
                        if i > 5:  # Only split if there's enough before
                            first_part = ' '.join(words[:i])
                            second_part = ' '.join(words[i:])
                            processed_sentences.append(first_part + '.')
                            processed_sentences.append(' ' + second_part)
                            break
                else:
                    processed_sentences.append(sentence)
            else:
                processed_sentences.append(sentence)

        simplified = ''.join(processed_sentences)

        return simplified.strip()

    def remove_redundancy(self, prompt: str) -> str:
        """
        Remove redundant information and repetition

        Args:
            prompt: Prompt to clean

        Returns:
            Cleaned prompt
        """
        cleaned = prompt

        # Remove redundant patterns
        for pattern in self.REDUNDANT_PATTERNS:
            if pattern == r'\s+':
                cleaned = re.sub(pattern, ' ', cleaned)
            elif pattern == r'\n\s*\n\s*\n+':
                cleaned = re.sub(pattern, '\n\n', cleaned)
            else:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Remove duplicate consecutive sentences
        sentences = re.split(r'([.!?])\s+', cleaned)
        unique_sentences = []
        prev_sentence = None

        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i]

            # Check if different from previous
            sentence_normalized = re.sub(r'\s+', ' ', sentence.lower()).strip()
            if sentence_normalized != prev_sentence:
                unique_sentences.append(sentence)
                prev_sentence = sentence_normalized

        cleaned = ' '.join(unique_sentences)

        return cleaned.strip()

    def format_as_bullet_points(self, prompt: str) -> str:
        """
        Convert instructions to bullet point format

        Args:
            prompt: Prompt to format

        Returns:
            Bullet-formatted prompt
        """
        # Check if already formatted
        if re.search(r'^[\s]*[-*•]\s', prompt, re.MULTILINE):
            return prompt

        # Split into sentences
        sentences = re.split(r'[.!?]\s+', prompt)

        # Filter out very short sentences (< 3 words)
        sentences = [s.strip() for s in sentences if len(s.split()) >= 3]

        if not sentences:
            return prompt

        # Format as bullet points
        bullet_points = []
        for sentence in sentences:
            if sentence:
                # Ensure starts with capital letter
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                bullet_points.append(f"- {sentence}")

        return '\n'.join(bullet_points)

    def structure_for_clarity(self, prompt: str) -> str:
        """
        Structure prompt for maximum clarity

        Args:
            prompt: Prompt to structure

        Returns:
            Structured prompt
        """
        # Check if prompt has clear sections
        has_sections = bool(re.search(r'^(Task|Goal|Instructions|Context|Output):', prompt, re.MULTILINE | re.IGNORECASE))

        if has_sections:
            return prompt  # Already structured

        # Try to identify sections
        lines = prompt.split('\n')
        structured_parts = []

        # Look for instruction keywords
        instruction_keywords = ['you should', 'you must', 'please', 'make sure', 'ensure', 'generate', 'create', 'write']

        task_lines = []
        instruction_lines = []
        other_lines = []

        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in instruction_keywords):
                instruction_lines.append(line.strip())
            elif len(line.split()) > 3:  # Substantial line
                task_lines.append(line.strip())
            elif line.strip():
                other_lines.append(line.strip())

        # Build structured output
        if task_lines:
            structured_parts.append("Task:")
            structured_parts.extend(f"- {line}" if not line.startswith('-') else line for line in task_lines)
            structured_parts.append("")

        if instruction_lines:
            structured_parts.append("Instructions:")
            structured_parts.extend(f"- {line}" if not line.startswith('-') else line for line in instruction_lines)
            structured_parts.append("")

        if other_lines:
            structured_parts.extend(other_lines)

        return '\n'.join(structured_parts) if structured_parts else prompt

    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text

        Uses rough heuristic: ~4 characters per token for English

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Simple heuristic: average of character count / 4 and word count * 1.3
        char_estimate = len(text) / 4
        word_estimate = len(text.split()) * 1.3

        return int((char_estimate + word_estimate) / 2)

    def truncate_to_limit(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens

        Returns:
            Truncated text
        """
        current_tokens = self.estimate_token_count(text)

        if current_tokens <= max_tokens:
            return text

        # Calculate target character length
        target_ratio = max_tokens / current_tokens
        target_chars = int(len(text) * target_ratio * 0.9)  # 0.9 for safety margin

        # Try to truncate at sentence boundary
        truncated = text[:target_chars]

        # Find last sentence boundary
        last_period = max(
            truncated.rfind('.'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )

        if last_period > target_chars * 0.7:  # At least 70% of target
            truncated = truncated[:last_period + 1]
        else:
            # Truncate at last space
            last_space = truncated.rfind(' ')
            if last_space > 0:
                truncated = truncated[:last_space]

        return truncated.strip()

    def ensure_within_context(self, prompt: str, context: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Ensure prompt and context fit within model's context window

        Args:
            prompt: Main prompt
            context: Optional context

        Returns:
            Tuple of (adjusted_prompt, adjusted_context)
        """
        prompt_tokens = self.estimate_token_count(prompt)
        context_tokens = self.estimate_token_count(context) if context else 0
        total_tokens = prompt_tokens + context_tokens

        max_context = self.config.max_context_tokens

        if total_tokens <= max_context:
            return prompt, context

        # Prioritize prompt over context
        if prompt_tokens > self.config.max_prompt_tokens:
            prompt = self.truncate_to_limit(prompt, self.config.max_prompt_tokens)
            prompt_tokens = self.config.max_prompt_tokens

        # Adjust context to fit
        if context:
            remaining_tokens = max_context - prompt_tokens
            if context_tokens > remaining_tokens:
                context = self.truncate_to_limit(context, remaining_tokens)

        return prompt, context

    def optimize_for_task(self, prompt: str, task_type: str = "general") -> OptimizationResult:
        """
        Optimize prompt for specific task type

        Args:
            prompt: Prompt to optimize
            task_type: Type of task (general, coding, reasoning, analysis)

        Returns:
            OptimizationResult
        """
        # Task-specific optimizations
        if task_type == "coding":
            # For coding tasks, preserve code structure
            return self.optimize(prompt, preserve_format=True)

        elif task_type == "reasoning":
            # For reasoning, add step-by-step structure
            if "step by step" not in prompt.lower():
                prompt += "\n\nThink step by step."
            return self.optimize(prompt)

        elif task_type == "analysis":
            # For analysis, encourage structured output
            if "format" not in prompt.lower():
                prompt += "\n\nProvide analysis in structured format."
            return self.optimize(prompt)

        else:
            # General optimization
            return self.optimize(prompt)

    def select_few_shot_examples(
        self,
        examples: List[Dict[str, str]],
        task: str,
        max_examples: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Select optimal few-shot examples for task

        Args:
            examples: Available examples
            task: Current task description
            max_examples: Maximum examples to select

        Returns:
            Selected examples
        """
        if not examples:
            return []

        max_examples = max_examples or self.config.few_shot_examples

        # Simple selection: take first N examples
        # In production, could use similarity-based selection
        selected = examples[:max_examples]

        # Check token budget
        total_example_tokens = sum(
            self.estimate_token_count(str(ex)) for ex in selected
        )

        # If examples too large, reduce count
        while selected and total_example_tokens > self.config.max_prompt_tokens * 0.3:
            selected = selected[:-1]
            total_example_tokens = sum(
                self.estimate_token_count(str(ex)) for ex in selected
            )

        logger.debug(f"Selected {len(selected)} examples (~{total_example_tokens} tokens)")

        return selected

    def optimize_chain_depth(self, chain_length: int) -> int:
        """
        Optimize chain depth for model capabilities

        Args:
            chain_length: Desired chain length

        Returns:
            Optimized chain length
        """
        if chain_length <= self.config.chain_max_depth:
            return chain_length

        logger.info(
            f"Chain length {chain_length} exceeds max {self.config.chain_max_depth}, "
            f"decomposing into multiple chains"
        )

        return self.config.chain_max_depth


def create_optimizer_for_model(model_name: str) -> PromptOptimizer:
    """
    Create optimizer configured for specific model

    Args:
        model_name: Name of the model

    Returns:
        Configured PromptOptimizer
    """
    # Detect model size from name
    model_name_lower = model_name.lower()

    if any(size in model_name_lower for size in ['0.5b', '1b', '1.5b', 'tiny']):
        model_size = ModelSize.TINY
    elif any(size in model_name_lower for size in ['2b', '3b', 'small']):
        model_size = ModelSize.SMALL
    elif any(size in model_name_lower for size in ['7b', 'medium']):
        model_size = ModelSize.MEDIUM
    else:
        model_size = ModelSize.LARGE

    config = OptimizationConfig.for_model_size(model_size)
    optimizer = PromptOptimizer(config)

    logger.info(f"Created optimizer for {model_name} (size: {model_size.value})")

    return optimizer
