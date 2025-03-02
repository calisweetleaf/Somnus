import sys
import re
import math
import operator
from typing import Union, Dict, Callable
import logging
from decimal import Decimal, InvalidOperation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CalculatorSystem")

class CalculatorError(Exception):
    """Custom exception for calculation errors."""
    pass

class SecureCalculator:
    """Secure calculator implementation with sanitized evaluation."""
    
    def __init__(self):
        """Initialize calculator with allowed operations."""
        self.operations: Dict[str, Callable] = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '**': operator.pow,
            '%': operator.mod,
            '//': operator.floordiv
        }
        
        self.functions: Dict[str, Callable] = {
            'abs': abs,
            'round': round,
            'max': max,
            'min': min,
            'pow': pow,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'log10': math.log10,
            'exp': math.exp
        }
    
    def sanitize_input(self, expr: str) -> str:
        """Sanitize input expression."""
        # Remove whitespace and validate characters
        expr = expr.replace(' ', '')
        if not re.match(r'^[\d\+\-\*\/\(\)\.\,\s\w]+$', expr):
            raise CalculatorError("Invalid characters in expression")
        return expr
    
    def validate_expression(self, expr: str) -> bool:
        """Validate expression structure."""
        # Check parentheses matching
        if expr.count('(') != expr.count(')'):
            raise CalculatorError("Unmatched parentheses")
        
        # Check for common dangerous patterns
        dangerous_patterns = ['import', 'eval', 'exec', 'open', '__']
        if any(pattern in expr.lower() for pattern in dangerous_patterns):
            raise CalculatorError("Invalid operation attempted")
        
        return True
    
    def evaluate(self, expr: str) -> Union[int, float, Decimal]:
        """Safely evaluate mathematical expression."""
        try:
            # Critical: More stringent input validation
            expr = self.sanitize_input(expr)
            self.validate_expression(expr)
            
            # Critical: Only allow specific mathematical operations
            if not re.match(r'^[\d\+\-\*\/\(\)\.\s\w]+$', expr):
                raise CalculatorError("Invalid expression format")
            
            # Use safe eval with restricted globals and locals
            result = Decimal(str(eval(
                expr,
                {"__builtins__": None},  # Critical: More restrictive
                {**self.operations, **self.functions}
            )))
            
            # Format result based on type
            if result == result.to_integral():
                return int(result)
            return float(result)
            
        except (ArithmeticError, InvalidOperation) as e:
            raise CalculatorError(f"Math error: {str(e)}")
        except Exception as e:
            raise CalculatorError(f"Calculation error: {str(e)}")

def main():
    """Main calculator function."""
    if len(sys.argv) < 2:
        logger.error("No expression provided")
        print("Usage: python calculate.py \"<mathematical expression>\"")
        print("Example: python calculate.py \"(2 + 3) * 4\"")
        sys.exit(1)
    
    calculator = SecureCalculator()
    expr = sys.argv[1]
    
    try:
        logger.info(f"Processing expression: {expr}")
        result = calculator.evaluate(expr)
        print(f"Result: {result}")
        logger.info(f"Calculation successful: {expr} = {result}")
    
    except CalculatorError as e:
        logger.error(f"Calculation failed: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()