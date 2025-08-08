"""
Tool registry for managing built-in and custom tools.
"""
import importlib
import inspect
from typing import Dict, Any, List, Callable, Optional
from langchain.tools import BaseTool
from langchain_core.tools import tool


class ToolRegistry:
    """Registry for managing available tools for agents."""
    
    def __init__(self):
        self._built_in_tools: Dict[str, Callable] = {}
        self._custom_tools: Dict[str, BaseTool] = {}
        self._initialize_built_in_tools()
    
    def _initialize_built_in_tools(self):
        """Initialize built-in tools."""
        
        @tool
        def web_search(query: str) -> str:
            """Search the web for information using Serper API (Google search)."""
            import requests
            import json
            import os
            from datetime import datetime
            
            # Input validation
            if not query or not isinstance(query, str):
                return "Error: Query must be a non-empty string"
            
            query = query.strip()
            if not query:
                return "Error: Query cannot be empty or only whitespace"
            
            if len(query) > 500:
                return "Error: Query too long (maximum 500 characters)"
            
            try:
                api_key = os.getenv('SERPER_API_KEY')
                if not api_key:
                    return "Web search unavailable: Please set SERPER_API_KEY environment variable. Get your free API key at https://serper.dev (2,500 free searches)."
                
                url = "https://google.serper.dev/search"
                payload = json.dumps({"q": query, "num": 8})
                headers = {
                    'X-API-KEY': api_key,
                    'Content-Type': 'application/json'
                }
                
                response = requests.post(url, headers=headers, data=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                results = []
                
                # Add answer box if available
                if data.get('answerBox'):
                    answer_box = data['answerBox']
                    if answer_box.get('answer'):
                        results.append(f"Answer: {answer_box['answer']}")
                    elif answer_box.get('snippet'):
                        results.append(f"Featured Snippet: {answer_box['snippet']}")
                
                # Add knowledge graph if available  
                if data.get('knowledgeGraph'):
                    kg = data['knowledgeGraph']
                    if kg.get('description'):
                        results.append(f"About: {kg['description']}")
                
                # Add organic search results
                if data.get('organic'):
                    if results:
                        results.append("\nWeb Results:")
                    search_results = []
                    for item in data['organic'][:5]:
                        title = item.get('title', 'No title')
                        snippet = item.get('snippet', 'No description')
                        search_results.append(f"• {title}: {snippet}")
                    results.extend(search_results)
                
                if results:
                    current_date = datetime.now().strftime("%B %d, %Y")
                    return f"Search results for '{query}' (searched on {current_date} via Serper API):\n\n" + "\n".join(results)
                else:
                    return f"No search results found for '{query}'. Please try a different search query."
                    
            except requests.exceptions.RequestException as e:
                return f"Web search error: Network issue - {str(e)}. Please check your internet connection."
            except json.JSONDecodeError:
                return f"Web search error: Invalid response from Serper API. Please try again."
            except Exception as e:
                return f"Web search error: {str(e)}. Please check your SERPER_API_KEY and try again."
        
        @tool
        def calculator(expression: str) -> str:
            """Calculate mathematical expressions safely."""
            import ast
            import operator
            import math
            
            # Input validation
            if not expression or not isinstance(expression, str):
                return "Error: Expression must be a non-empty string"
            
            expression = expression.strip()
            if not expression:
                return "Error: Expression cannot be empty or only whitespace"
            
            if len(expression) > 1000:
                return "Error: Expression too long (maximum 1000 characters)"
            
            # Safe operators and functions
            safe_operators = {
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
            
            safe_functions = {
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'sum': sum,
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'log10': math.log10,
                'exp': math.exp,
                'pi': math.pi,
                'e': math.e,
            }
            
            def safe_eval(node):
                if isinstance(node, ast.Expression):
                    return safe_eval(node.body)
                elif isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
                    return node.n
                elif isinstance(node, ast.BinOp):
                    left = safe_eval(node.left)
                    right = safe_eval(node.right)
                    op = safe_operators.get(type(node.op))
                    if op:
                        return op(left, right)
                    else:
                        raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
                elif isinstance(node, ast.UnaryOp):
                    operand = safe_eval(node.operand)
                    op = safe_operators.get(type(node.op))
                    if op:
                        return op(operand)
                    else:
                        raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in safe_functions:
                        func = safe_functions[node.func.id]
                        args = [safe_eval(arg) for arg in node.args]
                        return func(*args)
                    else:
                        raise ValueError(f"Unsupported function call")
                elif isinstance(node, ast.Name):
                    if node.id in safe_functions:
                        return safe_functions[node.id]
                    else:
                        raise ValueError(f"Unsupported variable: {node.id}")
                else:
                    raise ValueError(f"Unsupported expression type: {type(node).__name__}")
            
            try:
                # Parse the expression
                tree = ast.parse(expression, mode='eval')
                result = safe_eval(tree)
                return str(result)
            except (SyntaxError, ValueError) as e:
                return f"Invalid expression: {str(e)}"
            except ZeroDivisionError:
                return "Error: Division by zero"
            except Exception as e:
                return f"Calculation error: {str(e)}"
        
        @tool
        def file_reader(file_path: str) -> str:
            """Read contents of a file."""
            import os
            
            # Input validation
            if not file_path or not isinstance(file_path, str):
                return "Error: File path must be a non-empty string"
            
            file_path = file_path.strip()
            if not file_path:
                return "Error: File path cannot be empty or only whitespace"
            
            try:
                # Security: Check if file path is safe
                if '..' in file_path or file_path.startswith('/'):
                    return "Security error: Absolute paths and directory traversal not allowed"
                
                # Check if file exists and is readable
                if not os.path.exists(file_path):
                    return f"Error: File '{file_path}' does not exist"
                
                if not os.path.isfile(file_path):
                    return f"Error: '{file_path}' is not a file"
                
                # Check file size (limit to 1MB)
                file_size = os.path.getsize(file_path)
                if file_size > 1024 * 1024:  # 1MB limit
                    return f"Error: File too large ({file_size} bytes). Maximum size is 1MB"
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    return content
                    
            except UnicodeDecodeError:
                return f"Error: File '{file_path}' contains non-text content or unsupported encoding"
            except PermissionError:
                return f"Error: Permission denied to read file '{file_path}'"
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        @tool
        def file_writer(file_path: str, content: str) -> str:
            """Write content to a file."""
            import os
            
            # Input validation
            if not file_path or not isinstance(file_path, str):
                return "Error: File path must be a non-empty string"
            
            if not isinstance(content, str):
                return "Error: Content must be a string"
            
            file_path = file_path.strip()
            if not file_path:
                return "Error: File path cannot be empty or only whitespace"
            
            try:
                # Security: Check if file path is safe
                if '..' in file_path or file_path.startswith('/'):
                    return "Security error: Absolute paths and directory traversal not allowed"
                
                # Check content size (limit to 1MB)
                if len(content.encode('utf-8')) > 1024 * 1024:  # 1MB limit
                    return "Error: Content too large. Maximum size is 1MB"
                
                # Create directory if it doesn't exist
                directory = os.path.dirname(file_path)
                if directory and not os.path.exists(directory):
                    try:
                        os.makedirs(directory, exist_ok=True)
                    except Exception as e:
                        return f"Error creating directory '{directory}': {str(e)}"
                
                # Check if we can write to the location
                if os.path.exists(file_path) and not os.access(file_path, os.W_OK):
                    return f"Error: No write permission for file '{file_path}'"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return f"Successfully wrote {len(content)} characters to '{file_path}'"
                
            except PermissionError:
                return f"Error: Permission denied to write file '{file_path}'"
            except Exception as e:
                return f"Error writing file: {str(e)}"
        
        @tool
        def code_executor(code: str, language: str = "python") -> str:
            """Execute code in specified language with safety restrictions."""
            # Input validation
            if not code or not isinstance(code, str):
                return "Error: Code must be a non-empty string"
            
            code = code.strip()
            if not code:
                return "Error: Code cannot be empty or only whitespace"
            
            if len(code) > 10000:
                return "Error: Code too long (maximum 10,000 characters)"
            
            if not isinstance(language, str):
                return "Error: Language must be a string"
            
            if language.lower() != "python":
                return f"Error: Language '{language}' not supported. Only Python is currently supported."
            
            import ast
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            
            # Security checks - block dangerous operations
            dangerous_keywords = {
                'import os', 'import sys', 'import subprocess', 'import shutil',
                'open(', '__import__', 'eval(', 'exec(', 'compile(',
                'getattr(', 'setattr(', 'delattr(', 'globals(', 'locals(',
                'dir(', 'vars(', '__builtins__', '__globals__', 'breakpoint('
            }
            
            code_lower = code.lower()
            for keyword in dangerous_keywords:
                if keyword in code_lower:
                    return f"Security error: Code contains restricted operation: '{keyword}'"
            
            # Define allowed modules
            allowed_modules = {
                'math', 'random', 'datetime', 'json', 're', 'string', 'collections',
                'itertools', 'functools', 'statistics', 'decimal', 'fractions'
            }
            
            # Additional AST-based validation
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    # Block import statements for disallowed modules
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        module_names = []
                        if isinstance(node, ast.Import):
                            module_names = [alias.name for alias in node.names]
                        else:  # ast.ImportFrom
                            module_names = [node.module] if node.module else ['*']
                        
                        # Check if all imports are allowed
                        for module_name in module_names:
                            if module_name and module_name not in allowed_modules:
                                return f"Security error: Import of module '{module_name}' is not allowed. Allowed modules: {', '.join(sorted(allowed_modules))}"
                    
                    # Block function definitions that could be problematic
                    if isinstance(node, ast.FunctionDef):
                        if node.name.startswith('_'):
                            return "Security error: Private function definitions are not allowed"
                            
            except SyntaxError as e:
                return f"Syntax error: {str(e)}"
            
            try:
                # Create restricted execution environment
                safe_builtins = {
                    'abs': abs, 'all': all, 'any': any, 'bool': bool, 'dict': dict,
                    'enumerate': enumerate, 'filter': filter, 'float': float,
                    'int': int, 'len': len, 'list': list, 'map': map, 'max': max,
                    'min': min, 'print': print, 'range': range, 'reversed': reversed,
                    'round': round, 'set': set, 'sorted': sorted, 'str': str,
                    'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip,
                    'chr': chr, 'ord': ord, 'hex': hex, 'bin': bin, 'oct': oct
                }
                
                # Add __import__ function that only allows safe modules
                def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
                    if name in allowed_modules:
                        return __import__(name, globals, locals, fromlist, level)
                    else:
                        raise ImportError(f"Import of module '{name}' is not allowed")
                
                safe_builtins['__import__'] = safe_import
                
                safe_globals = {
                    '__builtins__': safe_builtins,
                }
                
                # Capture stdout and stderr
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, safe_globals)
                
                output = stdout_capture.getvalue()
                errors = stderr_capture.getvalue()
                
                result = []
                if output:
                    result.append(f"Output:\n{output.rstrip()}")
                if errors:
                    # Filter out common warnings that are not critical
                    error_lines = errors.strip().split('\n')
                    filtered_errors = []
                    for line in error_lines:
                        if 'SyntaxWarning: invalid escape sequence' not in line:
                            filtered_errors.append(line)
                    
                    if filtered_errors:
                        result.append(f"Errors:\n" + '\n'.join(filtered_errors))
                
                if not result:
                    result.append("Code executed successfully (no output)")
                
                return "\n".join(result)
                
            except Exception as e:
                return f"Execution error: {str(e)}"
        
        # Register built-in tools
        self._built_in_tools = {
            "web_search": web_search,
            "calculator": calculator,
            "file_reader": file_reader,
            "file_writer": file_writer,
            "code_executor": code_executor
        }
    
    def get_built_in_tools(self) -> List[str]:
        """Get list of available built-in tools."""
        return list(self._built_in_tools.keys())
    
    def get_tool(self, tool_name: str) -> Optional[Callable]:
        """Get a tool by name."""
        if tool_name in self._built_in_tools:
            return self._built_in_tools[tool_name]
        elif tool_name in self._custom_tools:
            return self._custom_tools[tool_name]
        return None
    
    def get_tools_by_names(self, tool_names: List[str]) -> List[Callable]:
        """Get multiple tools by their names."""
        tools = []
        for name in tool_names:
            tool_func = self.get_tool(name)
            if tool_func:
                tools.append(tool_func)
        return tools
    
    def register_custom_tool(self, name: str, module_path: str, class_name: str, 
                           description: str, parameters: Dict[str, Any] = None):
        """Register a custom tool from a module."""
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get the tool class
            tool_class = getattr(module, class_name)
            
            # Create tool instance
            if parameters:
                tool_instance = tool_class(**parameters)
            else:
                tool_instance = tool_class()
            
            # Validate that it's a proper tool
            if not isinstance(tool_instance, BaseTool):
                if hasattr(tool_instance, '__call__'):
                    # Convert function to tool if it's callable
                    tool_instance = tool(name=name, description=description)(tool_instance)
                else:
                    raise ValueError(f"Custom tool {name} must be a BaseTool or callable")
            
            self._custom_tools[name] = tool_instance
            return True
            
        except Exception as e:
            print(f"Error registering custom tool {name}: {str(e)}")
            return False
    
    def register_function_as_tool(self, name: str, func: Callable, description: str):
        """Register a function as a custom tool."""
        try:
            # Use the tool decorator without name parameter
            tool_instance = tool(func)
            # Manually set the name and description
            tool_instance.name = name
            tool_instance.description = description
            self._custom_tools[name] = tool_instance
            return True
        except Exception as e:
            print(f"Error registering function as tool {name}: {str(e)}")
            return False
    
    def list_all_tools(self) -> Dict[str, str]:
        """List all available tools with their descriptions."""
        all_tools = {}
        
        # Add built-in tools
        for name, tool_func in self._built_in_tools.items():
            if hasattr(tool_func, 'description'):
                all_tools[name] = tool_func.description
            else:
                all_tools[name] = f"Built-in tool: {name}"
        
        # Add custom tools
        for name, tool_instance in self._custom_tools.items():
            if hasattr(tool_instance, 'description'):
                all_tools[name] = tool_instance.description
            else:
                all_tools[name] = f"Custom tool: {name}"
        
        return all_tools
    
    def remove_custom_tool(self, name: str) -> bool:
        """Remove a custom tool."""
        if name in self._custom_tools:
            del self._custom_tools[name]
            return True
        return False
    
    def validate_tools(self, tool_names: List[str]) -> List[str]:
        """Validate that all specified tools exist. Returns list of missing tools."""
        missing_tools = []
        for name in tool_names:
            if name not in self._built_in_tools and name not in self._custom_tools:
                missing_tools.append(name)
        return missing_tools