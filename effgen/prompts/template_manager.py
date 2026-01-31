"""
Template Manager for Prompt Engineering

Provides comprehensive template management with Jinja2 support, dynamic example selection,
template optimization for SLMs, and versioning capabilities.
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import re
from jinja2 import Environment, FileSystemLoader, Template, meta
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Represents a prompt template with metadata"""

    name: str
    template: str
    description: Optional[str] = None
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def render(self, **kwargs) -> str:
        """Render template with provided variables"""
        template = Template(self.template)
        return template.render(**kwargs)

    def get_required_variables(self) -> List[str]:
        """Extract required variables from template"""
        env = Environment()
        parsed = env.parse(self.template)
        return list(meta.find_undeclared_variables(parsed))

    def validate_inputs(self, **kwargs) -> bool:
        """Validate that all required variables are provided"""
        required = set(self.get_required_variables())
        provided = set(kwargs.keys())
        missing = required - provided

        if missing:
            logger.warning(f"Missing required variables: {missing}")
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'name': self.name,
            'template': self.template,
            'description': self.description,
            'version': self.version,
            'tags': self.tags,
            'variables': self.variables,
            'examples': self.examples,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create from dictionary representation"""
        if 'created_at' in data and data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and data['updated_at']:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class FewShotExample:
    """Represents a few-shot example"""

    input: str
    output: str
    context: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format(self, style: str = "default") -> str:
        """Format example for inclusion in prompt"""
        if style == "default":
            result = f"Input: {self.input}\nOutput: {self.output}"
            if self.context:
                result = f"Context: {self.context}\n{result}"
            return result

        elif style == "chat":
            result = f"User: {self.input}\nAssistant: {self.output}"
            if self.context:
                result = f"Context: {self.context}\n{result}"
            return result

        elif style == "xml":
            result = f"<example>\n  <input>{self.input}</input>\n  <output>{self.output}</output>\n"
            if self.context:
                result = f"<example>\n  <context>{self.context}</context>\n  <input>{self.input}</input>\n  <output>{self.output}</output>\n"
            return result + "</example>"

        elif style == "json":
            return json.dumps({
                "input": self.input,
                "output": self.output,
                "context": self.context
            }, indent=2)

        return f"Input: {self.input}\nOutput: {self.output}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'input': self.input,
            'output': self.output,
            'context': self.context,
            'tags': self.tags,
            'difficulty': self.difficulty,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FewShotExample':
        """Create from dictionary"""
        return cls(**data)


class TemplateManager:
    """
    Manages prompt templates with support for:
    - Loading from YAML/JSON files
    - Template versioning
    - Dynamic example selection
    - Template optimization for SLMs
    - Variable extraction and validation
    """

    def __init__(self, template_dir: Optional[str] = None):
        """
        Initialize template manager

        Args:
            template_dir: Directory containing template files
        """
        self.templates: Dict[str, PromptTemplate] = {}
        self.examples: Dict[str, List[FewShotExample]] = {}
        self.template_versions: Dict[str, List[PromptTemplate]] = {}

        # Set default template directory
        if template_dir is None:
            template_dir = os.path.join(
                os.path.dirname(__file__),
                'templates'
            )

        self.template_dir = Path(template_dir)
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )

        logger.info(f"TemplateManager initialized with directory: {self.template_dir}")

    def load_from_yaml(self, filepath: Union[str, Path]) -> None:
        """
        Load templates from YAML file

        Args:
            filepath: Path to YAML file
        """
        filepath = Path(filepath)

        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)

            # Load templates
            if 'templates' in data:
                for template_data in data['templates']:
                    template = PromptTemplate.from_dict(template_data)
                    self.add_template(template)

            # Load examples
            if 'examples' in data:
                for example_group, examples_list in data['examples'].items():
                    self.examples[example_group] = [
                        FewShotExample.from_dict(ex) for ex in examples_list
                    ]

            logger.info(f"Loaded templates from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load templates from {filepath}: {e}")
            raise

    def load_from_directory(self, directory: Optional[Union[str, Path]] = None) -> None:
        """
        Load all YAML templates from directory

        Args:
            directory: Directory to load from (defaults to template_dir)
        """
        if directory is None:
            directory = self.template_dir
        else:
            directory = Path(directory)

        if not directory.exists():
            logger.warning(f"Template directory does not exist: {directory}")
            return

        # Load all YAML files
        for filepath in directory.glob("*.yaml"):
            try:
                self.load_from_yaml(filepath)
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")

        for filepath in directory.glob("*.yml"):
            try:
                self.load_from_yaml(filepath)
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")

        logger.info(f"Loaded {len(self.templates)} templates from {directory}")

    def add_template(self, template: PromptTemplate) -> None:
        """
        Add or update a template

        Args:
            template: Template to add
        """
        # Store version history
        if template.name in self.templates:
            if template.name not in self.template_versions:
                self.template_versions[template.name] = []
            self.template_versions[template.name].append(self.templates[template.name])

        # Update timestamp
        template.updated_at = datetime.now()

        # Store template
        self.templates[template.name] = template
        logger.debug(f"Added template: {template.name} (v{template.version})")

    def get_template(self, name: str, version: Optional[str] = None) -> Optional[PromptTemplate]:
        """
        Get template by name and optionally version

        Args:
            name: Template name
            version: Specific version (None for latest)

        Returns:
            Template or None if not found
        """
        if version is None:
            return self.templates.get(name)

        # Search version history
        if name in self.template_versions:
            for template in self.template_versions[name]:
                if template.version == version:
                    return template

        # Check current version
        current = self.templates.get(name)
        if current and current.version == version:
            return current

        return None

    def list_templates(self, tag: Optional[str] = None) -> List[str]:
        """
        List available template names

        Args:
            tag: Filter by tag

        Returns:
            List of template names
        """
        if tag is None:
            return list(self.templates.keys())

        return [
            name for name, template in self.templates.items()
            if tag in template.tags
        ]

    def render_template(
        self,
        name: str,
        variables: Optional[Dict[str, Any]] = None,
        include_examples: bool = False,
        num_examples: int = 3,
        example_style: str = "default",
        example_filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Render a template with variables and optional examples

        Args:
            name: Template name
            variables: Variables to render template with
            include_examples: Whether to include few-shot examples
            num_examples: Number of examples to include
            example_style: Style for formatting examples
            example_filter: Filter criteria for example selection
            **kwargs: Additional variables

        Returns:
            Rendered prompt string
        """
        template = self.get_template(name)
        if template is None:
            raise ValueError(f"Template not found: {name}")

        # Merge variables
        all_vars = {**(variables or {}), **kwargs}

        # Validate inputs
        if not template.validate_inputs(**all_vars):
            missing = set(template.get_required_variables()) - set(all_vars.keys())
            raise ValueError(f"Missing required variables: {missing}")

        # Build prompt parts
        prompt_parts = []

        # Add few-shot examples if requested
        if include_examples:
            examples = self.select_examples(
                name,
                num_examples=num_examples,
                filter_criteria=example_filter
            )

            if examples:
                prompt_parts.append("Here are some examples:\n")
                for i, example in enumerate(examples, 1):
                    prompt_parts.append(f"\nExample {i}:")
                    prompt_parts.append(example.format(style=example_style))
                prompt_parts.append("\n\nNow, your turn:\n")

        # Render main template
        rendered = template.render(**all_vars)
        prompt_parts.append(rendered)

        return "\n".join(prompt_parts)

    def select_examples(
        self,
        template_name: str,
        num_examples: int = 3,
        filter_criteria: Optional[Dict[str, Any]] = None,
        diversity_mode: str = "random"
    ) -> List[FewShotExample]:
        """
        Select few-shot examples for a template

        Args:
            template_name: Template name
            num_examples: Number of examples to select
            filter_criteria: Filter by tags, difficulty, etc.
            diversity_mode: Selection strategy (random, diverse, sequential)

        Returns:
            List of selected examples
        """
        # Get examples for this template
        examples = self.examples.get(template_name, [])

        if not examples:
            # Try to get examples from template's example list
            template = self.get_template(template_name)
            if template and template.examples:
                examples = [FewShotExample.from_dict(ex) for ex in template.examples]
            else:
                return []

        # Apply filters
        if filter_criteria:
            filtered = []
            for example in examples:
                match = True

                # Filter by tags
                if 'tags' in filter_criteria:
                    required_tags = set(filter_criteria['tags'])
                    example_tags = set(example.tags)
                    if not required_tags.issubset(example_tags):
                        match = False

                # Filter by difficulty
                if 'difficulty' in filter_criteria:
                    if example.difficulty != filter_criteria['difficulty']:
                        match = False

                # Filter by custom metadata
                if 'metadata' in filter_criteria:
                    for key, value in filter_criteria['metadata'].items():
                        if example.metadata.get(key) != value:
                            match = False

                if match:
                    filtered.append(example)

            examples = filtered

        # Select examples based on strategy
        if not examples:
            return []

        if diversity_mode == "random":
            import random
            return random.sample(examples, min(num_examples, len(examples)))

        elif diversity_mode == "diverse":
            # Select diverse examples (by tags)
            selected = []
            used_tags = set()

            for example in examples:
                example_tags = set(example.tags)
                if not example_tags.intersection(used_tags) or len(selected) < num_examples:
                    selected.append(example)
                    used_tags.update(example_tags)

                    if len(selected) >= num_examples:
                        break

            return selected

        elif diversity_mode == "sequential":
            # Return first N examples
            return examples[:num_examples]

        else:
            return examples[:num_examples]

    def optimize_template_for_slm(
        self,
        template: PromptTemplate,
        max_tokens: int = 512,
        simplify: bool = True,
        use_bullet_points: bool = True
    ) -> PromptTemplate:
        """
        Optimize template for Small Language Models

        Args:
            template: Template to optimize
            max_tokens: Maximum token budget
            simplify: Whether to simplify language
            use_bullet_points: Convert to bullet point format

        Returns:
            Optimized template
        """
        optimized_text = template.template

        # Simplification strategies
        if simplify:
            # Remove redundant phrases
            redundant_patterns = [
                (r'\bplease\b', ''),
                (r'\bkindly\b', ''),
                (r'\bi would like you to\b', ''),
                (r'\byou should\b', ''),
                (r'\bin order to\b', 'to'),
                (r'\bat this point in time\b', 'now'),
                (r'\bdue to the fact that\b', 'because'),
            ]

            for pattern, replacement in redundant_patterns:
                optimized_text = re.sub(pattern, replacement, optimized_text, flags=re.IGNORECASE)

        # Convert to bullet points if requested
        if use_bullet_points:
            # Find sentences that could be bullet points
            sentences = re.split(r'[.!?]\s+', optimized_text)
            if len(sentences) > 2:
                # Convert instructions to bullet points
                bullet_parts = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and not sentence.startswith('-') and not sentence.startswith('*'):
                        bullet_parts.append(f"- {sentence}")
                    elif sentence:
                        bullet_parts.append(sentence)

                if bullet_parts:
                    optimized_text = '\n'.join(bullet_parts)

        # Clean up whitespace
        optimized_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', optimized_text)
        optimized_text = optimized_text.strip()

        # Create optimized template
        optimized = PromptTemplate(
            name=f"{template.name}_optimized",
            template=optimized_text,
            description=f"Optimized version of {template.name}",
            version=template.version,
            tags=template.tags + ['optimized', 'slm'],
            variables=template.variables,
            examples=template.examples,
            metadata={**template.metadata, 'optimized_for': 'slm', 'original': template.name}
        )

        return optimized

    def create_chain_template(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        description: Optional[str] = None
    ) -> PromptTemplate:
        """
        Create a template for prompt chaining

        Args:
            name: Chain template name
            steps: List of step definitions
            description: Chain description

        Returns:
            Chain template
        """
        chain_template = {
            'name': name,
            'description': description,
            'steps': steps
        }

        template_text = yaml.dump(chain_template, default_flow_style=False)

        template = PromptTemplate(
            name=name,
            template=template_text,
            description=description,
            tags=['chain', 'multi-step'],
            metadata={'type': 'chain', 'num_steps': len(steps)}
        )

        self.add_template(template)
        return template

    def save_template(
        self,
        template: PromptTemplate,
        filepath: Union[str, Path]
    ) -> None:
        """
        Save template to YAML file

        Args:
            template: Template to save
            filepath: Output file path
        """
        filepath = Path(filepath)

        data = {
            'templates': [template.to_dict()]
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved template {template.name} to {filepath}")

    def export_all_templates(
        self,
        filepath: Union[str, Path]
    ) -> None:
        """
        Export all templates to single YAML file

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)

        data = {
            'templates': [t.to_dict() for t in self.templates.values()],
            'examples': {
                name: [ex.to_dict() for ex in examples]
                for name, examples in self.examples.items()
            }
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Exported {len(self.templates)} templates to {filepath}")

    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded templates"""
        return {
            'total_templates': len(self.templates),
            'total_versions': sum(len(v) for v in self.template_versions.values()),
            'templates_with_examples': len(self.examples),
            'total_examples': sum(len(ex) for ex in self.examples.values()),
            'tags': list(set(tag for t in self.templates.values() for tag in t.tags))
        }


def create_default_template_manager() -> TemplateManager:
    """Create template manager with default templates loaded"""
    manager = TemplateManager()

    # Try to load templates from default directory
    try:
        manager.load_from_directory()
    except Exception as e:
        logger.warning(f"Could not load default templates: {e}")

    return manager
