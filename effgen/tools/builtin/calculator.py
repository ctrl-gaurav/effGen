"""
Mathematical calculator tool with expression evaluation.

This module provides a calculator tool for mathematical operations,
expression evaluation, and unit conversions.
"""

import ast
import operator
import math
from typing import Dict, Any, Union, Optional
import logging
import re

from ..base_tool import (
    BaseTool,
    ToolMetadata,
    ToolCategory,
    ParameterSpec,
    ParameterType,
)


logger = logging.getLogger(__name__)


class Calculator(BaseTool):
    """
    Mathematical calculator with safe expression evaluation.

    Features:
    - Basic arithmetic operations (+, -, *, /, **, %)
    - Mathematical functions (sin, cos, sqrt, log, etc.)
    - Constants (pi, e)
    - Expression evaluation with operator precedence
    - Safe evaluation (no code execution)
    - Unit conversions
    - Statistical operations

    Security:
    - AST-based evaluation (no eval/exec)
    - Restricted to mathematical operations
    - No variable assignment or imports
    - Timeout protection
    """

    # Supported operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Supported mathematical functions
    FUNCTIONS = {
        # Trigonometric
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        # Hyperbolic
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        # Exponential and logarithmic
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        # Power and roots
        "sqrt": math.sqrt,
        "pow": math.pow,
        # Rounding
        "ceil": math.ceil,
        "floor": math.floor,
        "round": round,
        # Absolute value
        "abs": abs,
        # Other
        "factorial": math.factorial,
        "degrees": math.degrees,
        "radians": math.radians,
        # List operations (common for benchmark tasks)
        "sum": sum,
        "min": min,
        "max": max,
        "len": len,
        "prod": math.prod,
        "product": math.prod,
        "sorted": sorted,
        "mean": lambda x: sum(x) / len(x),
        "median": lambda x: (sorted(x)[len(x)//2] if len(x) % 2 else (sorted(x)[len(x)//2-1] + sorted(x)[len(x)//2])/2),
    }

    # Mathematical constants
    CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
    }

    # Unit conversion factors (to base unit)
    UNIT_CONVERSIONS = {
        # Length (base: meter)
        "length": {
            "m": 1.0,
            "km": 1000.0,
            "cm": 0.01,
            "mm": 0.001,
            "mi": 1609.34,
            "yd": 0.9144,
            "ft": 0.3048,
            "in": 0.0254,
        },
        # Mass (base: kilogram)
        "mass": {
            "kg": 1.0,
            "g": 0.001,
            "mg": 0.000001,
            "lb": 0.453592,
            "oz": 0.0283495,
        },
        # Temperature (special handling required)
        "temperature": {
            "c": "celsius",
            "f": "fahrenheit",
            "k": "kelvin",
        },
        # Time (base: second)
        "time": {
            "s": 1.0,
            "min": 60.0,
            "h": 3600.0,
            "day": 86400.0,
            "week": 604800.0,
        },
    }

    def __init__(self):
        """Initialize the calculator."""
        super().__init__(
            metadata=ToolMetadata(
                name="calculator",
                description="Perform mathematical calculations, evaluate expressions, and convert units",
                category=ToolCategory.COMPUTATION,
                parameters=[
                    ParameterSpec(
                        name="expression",
                        type=ParameterType.STRING,
                        description="Mathematical expression to evaluate",
                        required=True,
                        min_length=1,
                    ),
                    ParameterSpec(
                        name="operation",
                        type=ParameterType.STRING,
                        description="Type of operation",
                        required=False,
                        default="calculate",
                        enum=["calculate", "convert_units", "statistics"],
                    ),
                    ParameterSpec(
                        name="from_unit",
                        type=ParameterType.STRING,
                        description="Source unit for conversion",
                        required=False,
                    ),
                    ParameterSpec(
                        name="to_unit",
                        type=ParameterType.STRING,
                        description="Target unit for conversion",
                        required=False,
                    ),
                    ParameterSpec(
                        name="precision",
                        type=ParameterType.INTEGER,
                        description="Number of decimal places for result",
                        required=False,
                        min_value=0,
                        max_value=15,
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "result": {"type": "number"},
                        "formatted": {"type": "string"},
                        "expression": {"type": "string"},
                    },
                },
                timeout_seconds=5,
                tags=["math", "calculator", "computation", "conversion"],
                examples=[
                    {
                        "expression": "2 + 2 * 3",
                        "output": {"result": 8, "formatted": "8"},
                    },
                    {
                        "expression": "sqrt(16) + pow(2, 3)",
                        "output": {"result": 12.0, "formatted": "12.0"},
                    },
                    {
                        "expression": "sin(pi / 2)",
                        "output": {"result": 1.0, "formatted": "1.0"},
                    },
                ],
            )
        )

    async def _execute(
        self,
        expression: str,
        operation: str = "calculate",
        from_unit: Optional[str] = None,
        to_unit: Optional[str] = None,
        precision: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute calculator operation.

        Args:
            expression: Mathematical expression or value
            operation: Type of operation
            from_unit: Source unit for conversion
            to_unit: Target unit for conversion
            precision: Decimal places for result

        Returns:
            Dict with result, formatted string, and expression
        """
        try:
            if operation == "calculate":
                result = self._evaluate_expression(expression)
            elif operation == "convert_units":
                if not from_unit or not to_unit:
                    raise ValueError("Both from_unit and to_unit required for conversion")
                value = self._evaluate_expression(expression)
                result = self._convert_units(value, from_unit, to_unit)
            elif operation == "statistics":
                result = self._calculate_statistics(expression)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            # Format result
            if precision is not None and isinstance(result, (int, float)):
                formatted = f"{result:.{precision}f}"
            else:
                formatted = str(result)

            return {
                "result": result,
                "formatted": formatted,
                "expression": expression,
            }

        except Exception as e:
            # Log at debug level for expected calculation errors
            logger.debug(f"Calculator error: {e}")
            raise ValueError(f"Calculation failed: {str(e)}")

    def _evaluate_expression(self, expression: str) -> Union[int, float]:
        """
        Safely evaluate a mathematical expression using AST.

        Args:
            expression: Mathematical expression string

        Returns:
            Numerical result
        """
        # Sanitize input - remove common natural language patterns
        expr = expression.strip()

        # Remove markdown code fences (```python, ```, backticks)
        expr = re.sub(r'^```(?:python|math|markdown)?\s*', '', expr, flags=re.MULTILINE)
        expr = re.sub(r'\s*```$', '', expr, flags=re.MULTILINE)
        expr = re.sub(r'^`+|`+$', '', expr)  # Remove backticks at start/end
        expr = expr.strip()

        # Remove phrases like "What is", "Calculate", "perform", etc.
        expr = re.sub(r'^(what is|calculate|compute|find|evaluate|perform)\s+', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\?+$', '', expr)  # Remove trailing question marks

        # Handle "perform operation(args)" patterns
        expr = re.sub(r'perform\s+(addition|subtraction|multiplication|division)\s+', '', expr, flags=re.IGNORECASE)

        # Handle function-style syntax: add(x, y) -> x + y
        expr = re.sub(r'add\s*\(?\s*(\d+\.?\d*)\s*,?\s*(\d+\.?\d*)\s*\)?', r'\1 + \2', expr, flags=re.IGNORECASE)
        expr = re.sub(r'subtract\s*\(?\s*(\d+\.?\d*)\s*,?\s*(\d+\.?\d*)\s*\)?', r'\1 - \2', expr, flags=re.IGNORECASE)
        expr = re.sub(r'multiply\s*\(?\s*(\d+\.?\d*)\s*,?\s*(\d+\.?\d*)\s*\)?', r'\1 * \2', expr, flags=re.IGNORECASE)
        expr = re.sub(r'divide\s*\(?\s*(\d+\.?\d*)\s*,?\s*(\d+\.?\d*)\s*\)?', r'\1 / \2', expr, flags=re.IGNORECASE)

        # Handle "add x y" without parentheses
        expr = re.sub(r'\b(add|sum)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\b', r'\2 + \3', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\b(subtract|minus)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\b', r'\2 - \3', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\b(multiply|times)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\b', r'\2 * \3', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\b(divide)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\b', r'\2 / \3', expr, flags=re.IGNORECASE)

        # Replace ^ with ** for exponentiation (handle BitXor issue)
        expr = re.sub(r'(\d+\.?\d*)\s*\^\s*(\d+\.?\d*)', r'\1 ** \2', expr)

        # Replace "to the power of" with **
        expr = re.sub(r'\s+to\s+the\s+power\s+of\s+', '**', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\s+raised\s+to\s+', '**', expr, flags=re.IGNORECASE)

        # Replace "square root of" with sqrt()
        expr = re.sub(r'square\s+root\s+of\s+(\d+)', r'sqrt(\1)', expr, flags=re.IGNORECASE)

        # Handle "X squared" -> X**2
        expr = re.sub(r'(\d+)\s+squared', r'\1**2', expr, flags=re.IGNORECASE)

        # Replace constants
        for const_name, const_value in self.CONSTANTS.items():
            expr = re.sub(
                r'\b' + const_name + r'\b',
                str(const_value),
                expr,
                flags=re.IGNORECASE,
            )

        # Parse expression
        try:
            node = ast.parse(expr, mode='eval').body
        except SyntaxError as e:
            # Try to provide helpful error message
            raise ValueError(f"Invalid expression syntax. Input was: '{expression}'. Please use mathematical operators like +, -, *, /, **, sqrt(), etc.")

        # Evaluate AST
        return self._eval_node(node)

    def _eval_node(self, node: ast.AST) -> Union[int, float, list, tuple]:
        """
        Recursively evaluate an AST node.

        Args:
            node: AST node to evaluate

        Returns:
            Numerical result or list/tuple for aggregate functions
        """
        if isinstance(node, ast.Num):  # Number
            return node.n

        elif isinstance(node, ast.Constant):  # Python 3.8+ constants
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.BinOp):  # Binary operation
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.OPERATORS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):  # Unary operation
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return self.OPERATORS[op_type](operand)

        elif isinstance(node, ast.Call):  # Function call
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are supported")

            func_name = node.func.id.lower()
            if func_name not in self.FUNCTIONS:
                raise ValueError(f"Unknown function: {func_name}")

            # Evaluate arguments
            args = [self._eval_node(arg) for arg in node.args]

            # Call function
            try:
                return self.FUNCTIONS[func_name](*args)
            except Exception as e:
                raise ValueError(f"Function {func_name} error: {e}")

        elif isinstance(node, ast.List):  # List literal [1, 2, 3]
            return [self._eval_node(elem) for elem in node.elts]

        elif isinstance(node, ast.Tuple):  # Tuple literal (1, 2, 3)
            return tuple(self._eval_node(elem) for elem in node.elts)

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    def _convert_units(
        self, value: Union[int, float], from_unit: str, to_unit: str
    ) -> float:
        """
        Convert value between units.

        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value
        """
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        # Find unit category
        category = None
        for cat, units in self.UNIT_CONVERSIONS.items():
            if from_unit in units and to_unit in units:
                category = cat
                break

        if not category:
            raise ValueError(f"Cannot convert from '{from_unit}' to '{to_unit}'")

        # Special handling for temperature
        if category == "temperature":
            return self._convert_temperature(value, from_unit, to_unit)

        # Standard conversion: convert to base unit, then to target unit
        base_value = value * self.UNIT_CONVERSIONS[category][from_unit]
        result = base_value / self.UNIT_CONVERSIONS[category][to_unit]

        return result

    def _convert_temperature(
        self, value: float, from_unit: str, to_unit: str
    ) -> float:
        """Convert temperature between Celsius, Fahrenheit, and Kelvin."""
        # Convert to Celsius first
        if from_unit == "c":
            celsius = value
        elif from_unit == "f":
            celsius = (value - 32) * 5 / 9
        elif from_unit == "k":
            celsius = value - 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")

        # Convert from Celsius to target
        if to_unit == "c":
            return celsius
        elif to_unit == "f":
            return celsius * 9 / 5 + 32
        elif to_unit == "k":
            return celsius + 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")

    def _calculate_statistics(self, expression: str) -> Dict[str, float]:
        """
        Calculate statistics for a list of numbers.

        Args:
            expression: Comma-separated list of numbers

        Returns:
            Dict with statistical measures
        """
        # Parse numbers
        try:
            numbers = [float(x.strip()) for x in expression.split(",")]
        except ValueError as e:
            raise ValueError(f"Invalid number format: {e}")

        if not numbers:
            raise ValueError("No numbers provided")

        # Calculate statistics
        n = len(numbers)
        mean = sum(numbers) / n
        sorted_nums = sorted(numbers)

        # Median
        if n % 2 == 0:
            median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
        else:
            median = sorted_nums[n // 2]

        # Variance and standard deviation
        variance = sum((x - mean) ** 2 for x in numbers) / n
        std_dev = math.sqrt(variance)

        return {
            "count": n,
            "sum": sum(numbers),
            "mean": mean,
            "median": median,
            "min": min(numbers),
            "max": max(numbers),
            "range": max(numbers) - min(numbers),
            "variance": variance,
            "std_dev": std_dev,
        }
