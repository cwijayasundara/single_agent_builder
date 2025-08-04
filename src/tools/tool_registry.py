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
            """Search the web for information."""
            return f"Web search results for: {query}"
        
        @tool
        def calculator(expression: str) -> str:
            """Calculate mathematical expressions."""
            try:
                result = eval(expression)
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        @tool
        def file_reader(file_path: str) -> str:
            """Read contents of a file."""
            try:
                with open(file_path, 'r') as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        @tool
        def file_writer(file_path: str, content: str) -> str:
            """Write content to a file."""
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
                return f"Successfully wrote to {file_path}"
            except Exception as e:
                return f"Error writing file: {str(e)}"
        
        @tool
        def code_executor(code: str, language: str = "python") -> str:
            """Execute code in specified language."""
            if language.lower() == "python":
                try:
                    exec_globals = {}
                    exec(code, exec_globals)
                    return "Code executed successfully"
                except Exception as e:
                    return f"Error executing code: {str(e)}"
            else:
                return f"Language {language} not supported"
        
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