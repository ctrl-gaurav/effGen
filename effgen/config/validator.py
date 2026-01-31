"""
Configuration validator using JSON schemas.

Provides validation for:
- Model configurations
- Tool configurations
- Prompt configurations
- API key configurations
- Agent configurations

Features:
- JSON Schema validation
- Cross-reference validation
- Semantic validation
- Security checks
- Custom validation rules
"""

import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

try:
    import jsonschema
    from jsonschema import Draft7Validator, validators
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    Draft7Validator = None
    validators = None

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Configuration validation error."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        """
        Initialize validation error.

        Args:
            message: Error message
            errors: List of specific validation errors
        """
        super().__init__(message)
        self.errors = errors or []

    def __str__(self):
        """Format error message with details."""
        if not self.errors:
            return super().__str__()

        error_list = "\n  - ".join(self.errors)
        return f"{super().__str__()}\n  - {error_list}"


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning."""
        self.warnings.append(warning)


class ConfigValidator:
    """
    Configuration validator with JSON Schema support.

    Validates:
    - Syntax (via JSON schema)
    - Semantics (cross-references, constraints)
    - Security (no plaintext secrets in non-key files)
    """

    def __init__(self, schema_dir: Optional[Path] = None):
        """
        Initialize validator.

        Args:
            schema_dir: Directory containing JSON schemas
        """
        if not JSONSCHEMA_AVAILABLE:
            logger.warning("jsonschema not installed. Validation will be limited.")

        self.schema_dir = schema_dir or Path(__file__).parent / "schemas"
        self._schemas: Dict[str, Dict] = {}
        self._load_schemas()

    def _load_schemas(self) -> None:
        """Load JSON schemas from schema directory."""
        if not self.schema_dir.exists():
            logger.warning(f"Schema directory not found: {self.schema_dir}")
            return

        import json

        for schema_file in self.schema_dir.glob("*.json"):
            try:
                with open(schema_file, "r") as f:
                    schema = json.load(f)
                    schema_name = schema_file.stem
                    self._schemas[schema_name] = schema
                    logger.debug(f"Loaded schema: {schema_name}")
            except Exception as e:
                logger.error(f"Failed to load schema {schema_file}: {e}")

    def validate_models(self, models_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate models configuration.

        Args:
            models_config: Models configuration dictionary

        Returns:
            Validation result
        """
        result = ValidationResult(valid=True)

        # JSON Schema validation
        if JSONSCHEMA_AVAILABLE and "model_config" in self._schemas:
            schema_errors = self._validate_with_schema(
                models_config,
                self._schemas["model_config"]
            )
            for error in schema_errors:
                result.add_error(f"Schema validation: {error}")

        # Semantic validation
        if "models" in models_config:
            for model_name, model_config in models_config["models"].items():
                # Validate model type
                model_type = model_config.get("type")
                if not model_type:
                    result.add_error(f"Model '{model_name}' missing 'type' field")
                    continue

                # Validate based on type
                if model_type == "huggingface":
                    self._validate_huggingface_model(model_name, model_config, result)
                elif model_type == "openai":
                    self._validate_openai_model(model_name, model_config, result)
                elif model_type == "anthropic":
                    self._validate_anthropic_model(model_name, model_config, result)
                elif model_type == "google":
                    self._validate_google_model(model_name, model_config, result)
                else:
                    result.add_warning(f"Unknown model type for '{model_name}': {model_type}")

        # Validate default model exists
        if "default_model" in models_config:
            default = models_config["default_model"]
            if "models" in models_config and default not in models_config["models"]:
                result.add_error(f"Default model '{default}' not found in models")

        # Validate fallback chain
        if "fallback_chain" in models_config:
            for model_name in models_config["fallback_chain"]:
                if "models" in models_config and model_name not in models_config["models"]:
                    result.add_error(f"Fallback model '{model_name}' not found in models")

        return result

    def _validate_huggingface_model(
        self,
        model_name: str,
        config: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate HuggingFace model configuration."""
        # Required fields
        if "model_id" not in config:
            result.add_error(f"Model '{model_name}' missing 'model_id' field")

        # Validate inference engine
        inference_engine = config.get("inference_engine", "transformers")
        valid_engines = ["transformers", "vllm", "tgi", "ctransformers"]
        if inference_engine not in valid_engines:
            result.add_error(
                f"Model '{model_name}' has invalid inference_engine. "
                f"Must be one of: {valid_engines}"
            )

        # Validate quantization
        quantization = config.get("quantization")
        if quantization:
            valid_quant = ["none", "4bit", "8bit", "awq", "gptq"]
            if quantization not in valid_quant:
                result.add_warning(
                    f"Model '{model_name}' has unknown quantization: {quantization}"
                )

        # Validate GPU devices
        gpu_devices = config.get("gpu_devices", [])
        if gpu_devices and not isinstance(gpu_devices, list):
            result.add_error(f"Model '{model_name}' gpu_devices must be a list")

        # Check tensor parallel consistency
        tensor_parallel = config.get("tensor_parallel_size")
        if tensor_parallel:
            if not gpu_devices:
                result.add_error(
                    f"Model '{model_name}' has tensor_parallel_size but no gpu_devices"
                )
            elif len(gpu_devices) != tensor_parallel:
                result.add_warning(
                    f"Model '{model_name}' tensor_parallel_size ({tensor_parallel}) "
                    f"doesn't match gpu_devices length ({len(gpu_devices)})"
                )

    def _validate_openai_model(
        self,
        model_name: str,
        config: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate OpenAI model configuration."""
        if "model_name" not in config:
            result.add_error(f"Model '{model_name}' missing 'model_name' field")

        # Validate temperature range
        temp = config.get("temperature")
        if temp is not None and not (0 <= temp <= 2):
            result.add_warning(
                f"Model '{model_name}' temperature ({temp}) outside typical range [0, 2]"
            )

    def _validate_anthropic_model(
        self,
        model_name: str,
        config: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate Anthropic model configuration."""
        if "model_name" not in config:
            result.add_error(f"Model '{model_name}' missing 'model_name' field")

        # Validate temperature range
        temp = config.get("temperature")
        if temp is not None and not (0 <= temp <= 1):
            result.add_warning(
                f"Model '{model_name}' temperature ({temp}) outside typical range [0, 1]"
            )

    def _validate_google_model(
        self,
        model_name: str,
        config: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate Google model configuration."""
        if "model_name" not in config:
            result.add_error(f"Model '{model_name}' missing 'model_name' field")

    def validate_tools(self, tools_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate tools configuration.

        Args:
            tools_config: Tools configuration dictionary

        Returns:
            Validation result
        """
        result = ValidationResult(valid=True)

        # JSON Schema validation
        if JSONSCHEMA_AVAILABLE and "tool_config" in self._schemas:
            schema_errors = self._validate_with_schema(
                tools_config,
                self._schemas["tool_config"]
            )
            for error in schema_errors:
                result.add_error(f"Schema validation: {error}")

        # Validate built-in tools
        if "tools" in tools_config:
            for tool_name, tool_config in tools_config["tools"].items():
                if not isinstance(tool_config, dict):
                    result.add_error(f"Tool '{tool_name}' config must be a dictionary")
                    continue

                # Check enabled flag
                if "enabled" not in tool_config:
                    result.add_warning(f"Tool '{tool_name}' missing 'enabled' field")

        # Validate MCP servers
        if "mcp_servers" in tools_config:
            for server_name, server_config in tools_config["mcp_servers"].items():
                self._validate_mcp_server(server_name, server_config, result)

        # Validate tool selection
        if "tool_selection" in tools_config:
            selection = tools_config["tool_selection"]

            # Validate mode
            mode = selection.get("mode")
            if mode and mode not in ["auto", "manual", "hybrid"]:
                result.add_error(f"Invalid tool_selection mode: {mode}")

            # Validate recommendations reference existing tools
            if "recommendations" in selection and "tools" in tools_config:
                available_tools = set(tools_config["tools"].keys())
                if "mcp_servers" in tools_config:
                    available_tools.update(tools_config["mcp_servers"].keys())

                for task_type, tools in selection["recommendations"].items():
                    for tool_name in tools:
                        if tool_name not in available_tools:
                            result.add_warning(
                                f"Recommended tool '{tool_name}' for task '{task_type}' "
                                f"not found in available tools"
                            )

        return result

    def _validate_mcp_server(
        self,
        server_name: str,
        config: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate MCP server configuration."""
        # Check required fields
        if "command" not in config:
            result.add_error(f"MCP server '{server_name}' missing 'command' field")

        # Validate type
        server_type = config.get("type", "mcp")
        if server_type != "mcp":
            result.add_warning(
                f"MCP server '{server_name}' has unexpected type: {server_type}"
            )

        # Check environment variables are in correct format
        env = config.get("env", {})
        if env and not isinstance(env, dict):
            result.add_error(f"MCP server '{server_name}' env must be a dictionary")

    def validate_prompts(self, prompts_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate prompts configuration.

        Args:
            prompts_config: Prompts configuration dictionary

        Returns:
            Validation result
        """
        result = ValidationResult(valid=True)

        # Validate system prompts
        if "system_prompts" in prompts_config:
            for prompt_name, prompt_config in prompts_config["system_prompts"].items():
                if "template" not in prompt_config:
                    result.add_error(
                        f"System prompt '{prompt_name}' missing 'template' field"
                    )

        # Validate prompt chains
        if "chains" in prompts_config:
            for chain_name, chain_config in prompts_config["chains"].items():
                self._validate_chain(chain_name, chain_config, result)

        # Validate few-shot examples
        if "few_shot_examples" in prompts_config:
            for example_type, example_config in prompts_config["few_shot_examples"].items():
                if "examples" not in example_config:
                    result.add_error(
                        f"Few-shot example type '{example_type}' missing 'examples' field"
                    )

        # Validate SLM optimization settings
        if "slm_optimization" in prompts_config:
            slm_opt = prompts_config["slm_optimization"]

            # Check reasonable values
            max_examples = slm_opt.get("max_few_shot_examples")
            if max_examples and max_examples > 10:
                result.add_warning(
                    f"max_few_shot_examples ({max_examples}) is quite high. "
                    f"Consider reducing for SLMs."
                )

        return result

    def _validate_chain(
        self,
        chain_name: str,
        config: Dict[str, Any],
        result: ValidationResult
    ) -> None:
        """Validate a prompt chain configuration."""
        if "steps" not in config:
            result.add_error(f"Chain '{chain_name}' missing 'steps' field")
            return

        steps = config["steps"]
        if not isinstance(steps, list):
            result.add_error(f"Chain '{chain_name}' steps must be a list")
            return

        step_names = set()
        output_vars = set()

        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                result.add_error(f"Chain '{chain_name}' step {i} must be a dictionary")
                continue

            # Check required fields
            if "name" not in step:
                result.add_error(f"Chain '{chain_name}' step {i} missing 'name' field")
                continue

            step_name = step["name"]

            # Check for duplicate step names
            if step_name in step_names:
                result.add_error(
                    f"Chain '{chain_name}' has duplicate step name: {step_name}"
                )
            step_names.add(step_name)

            # Validate step type
            step_type = step.get("type", "prompt")
            valid_types = ["prompt", "tool", "conditional", "iterative"]
            if step_type not in valid_types:
                result.add_error(
                    f"Chain '{chain_name}' step '{step_name}' has invalid type: {step_type}"
                )

            # Track output variables
            if "output_var" in step:
                output_vars.add(step["output_var"])

            # Validate variable references in prompts
            if "prompt" in step:
                self._validate_template_vars(
                    step["prompt"],
                    output_vars,
                    f"Chain '{chain_name}' step '{step_name}'",
                    result
                )

    def _validate_template_vars(
        self,
        template: str,
        available_vars: Set[str],
        context: str,
        result: ValidationResult
    ) -> None:
        """Validate that template variables are available."""
        # Find all {variable} references
        var_pattern = re.compile(r"\{(\w+)\}")
        referenced_vars = set(var_pattern.findall(template))

        # Check if all referenced variables are available
        for var in referenced_vars:
            if var not in available_vars:
                result.add_warning(
                    f"{context} references undefined variable: {var}"
                )

    def validate_api_keys(self, api_keys_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate API keys configuration.

        Checks for:
        - Proper structure
        - Environment variable format
        - No plaintext secrets

        Args:
            api_keys_config: API keys configuration dictionary

        Returns:
            Validation result
        """
        result = ValidationResult(valid=True)

        # Check for plaintext secrets (should use env vars)
        self._check_plaintext_secrets(api_keys_config, "", result)

        return result

    def _check_plaintext_secrets(
        self,
        config: Any,
        path: str,
        result: ValidationResult
    ) -> None:
        """
        Recursively check for plaintext secrets.

        Args:
            config: Configuration to check
            path: Current path in config
            result: Validation result to update
        """
        if isinstance(config, dict):
            for key, value in config.items():
                current_path = f"{path}.{key}" if path else key

                # Check if this looks like a sensitive field
                sensitive_keys = [
                    "key", "token", "password", "secret", "credential",
                    "api_key", "access_key", "private_key"
                ]

                is_sensitive = any(
                    sensitive in key.lower() for sensitive in sensitive_keys
                )

                if is_sensitive and isinstance(value, str):
                    # Check if it's an environment variable reference
                    if not (value.startswith("${") and value.endswith("}")):
                        # Could be a plaintext secret
                        if len(value) > 10 and not value.startswith("http"):
                            result.add_error(
                                f"Possible plaintext secret at '{current_path}'. "
                                f"Use environment variables: ${{VAR_NAME}}"
                            )

                # Recurse
                self._check_plaintext_secrets(value, current_path, result)

        elif isinstance(config, list):
            for i, item in enumerate(config):
                current_path = f"{path}[{i}]"
                self._check_plaintext_secrets(item, current_path, result)

    def validate_agent(self, agent_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate agent configuration.

        Args:
            agent_config: Agent configuration dictionary

        Returns:
            Validation result
        """
        result = ValidationResult(valid=True)

        # JSON Schema validation
        if JSONSCHEMA_AVAILABLE and "agent_config" in self._schemas:
            schema_errors = self._validate_with_schema(
                agent_config,
                self._schemas["agent_config"]
            )
            for error in schema_errors:
                result.add_error(f"Schema validation: {error}")

        # Validate required fields
        required_fields = ["name", "model"]
        for field in required_fields:
            if field not in agent_config:
                result.add_error(f"Agent config missing required field: {field}")

        return result

    def _validate_with_schema(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[str]:
        """
        Validate configuration against JSON schema.

        Args:
            config: Configuration to validate
            schema: JSON schema

        Returns:
            List of validation errors
        """
        if not JSONSCHEMA_AVAILABLE:
            return []

        errors = []

        try:
            validator = Draft7Validator(schema)
            for error in validator.iter_errors(config):
                # Format error message
                path = ".".join(str(p) for p in error.path) if error.path else "root"
                errors.append(f"At '{path}': {error.message}")
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            errors.append(f"Schema validation error: {e}")

        return errors

    def validate_all(
        self,
        config: Dict[str, Any],
        config_type: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a complete configuration.

        Args:
            config: Configuration dictionary
            config_type: Type of config (models, tools, prompts, api_keys, agent)

        Returns:
            Combined validation result
        """
        result = ValidationResult(valid=True)

        # Auto-detect config type if not specified
        if not config_type:
            if "models" in config:
                config_type = "models"
            elif "tools" in config or "mcp_servers" in config:
                config_type = "tools"
            elif "system_prompts" in config or "chains" in config:
                config_type = "prompts"

        # Validate based on type
        if config_type == "models":
            models_result = self.validate_models(config)
            result.errors.extend(models_result.errors)
            result.warnings.extend(models_result.warnings)
            if not models_result.valid:
                result.valid = False

        elif config_type == "tools":
            tools_result = self.validate_tools(config)
            result.errors.extend(tools_result.errors)
            result.warnings.extend(tools_result.warnings)
            if not tools_result.valid:
                result.valid = False

        elif config_type == "prompts":
            prompts_result = self.validate_prompts(config)
            result.errors.extend(prompts_result.errors)
            result.warnings.extend(prompts_result.warnings)
            if not prompts_result.valid:
                result.valid = False

        elif config_type == "api_keys":
            api_result = self.validate_api_keys(config)
            result.errors.extend(api_result.errors)
            result.warnings.extend(api_result.warnings)
            if not api_result.valid:
                result.valid = False

        elif config_type == "agent":
            agent_result = self.validate_agent(config)
            result.errors.extend(agent_result.errors)
            result.warnings.extend(agent_result.warnings)
            if not agent_result.valid:
                result.valid = False

        # Raise exception if validation failed
        if not result.valid:
            raise ValidationError(
                f"Configuration validation failed with {len(result.errors)} error(s)",
                result.errors
            )

        # Log warnings
        for warning in result.warnings:
            logger.warning(warning)

        return result
