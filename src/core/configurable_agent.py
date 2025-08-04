"""
Main configurable agent class that ties everything together.
"""
import os
import time
import uuid
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

from .config_loader import ConfigLoader, AgentConfiguration
from ..tools.tool_registry import ToolRegistry
from ..memory.memory_manager import MemoryManager
from ..evaluation.evaluation_manager import EvaluationManager
from ..custom_logging import get_logger, CorrelationContext

# Load environment variables from .env file
load_dotenv()

# Get logger for this module
logger = get_logger(__name__, "agent")


class ConfigurableAgent:
    """Main configurable agent class."""
    
    def __init__(self, config_file: str):
        self.agent_id = str(uuid.uuid4())[:8]
        self.config_file = config_file
        self.config_loader = ConfigLoader()
        self.config = None
        self.tool_registry = ToolRegistry()
        self.memory_manager = None
        self.evaluation_manager = None
        self.llm = None
        self.graph = None
        
        # Start initialization with correlation context
        with CorrelationContext(self.agent_id):
            logger.log_agent_start(
                agent_name=getattr(self.config, 'agent', {}).get('name', 'unnamed') if self.config else 'unnamed',
                agent_type="single"
            )
            
            try:
                self.config = self.config_loader.load_config(config_file)
                logger.info(
                    "Configuration loaded successfully",
                    extra={
                        "config_file": config_file,
                        "agent_name": self.config.agent.name,
                        "llm_provider": self.config.llm.provider,
                        "tools_count": len(self.config.tools.built_in) + len(self.config.tools.custom)
                    }
                )
                self._initialize_components()
            except Exception as e:
                logger.error(
                    "Agent initialization failed",
                    extra={
                        "config_file": config_file,
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    exc_info=True
                )
                raise
    
    def _initialize_components(self):
        """Initialize all agent components."""
        logger.debug("Starting component initialization", extra={"agent_id": self.agent_id})
        
        self._setup_llm()
        self._setup_tools()
        self._setup_memory()
        self._setup_evaluation()
        self._setup_graph()
        
        # Log successful initialization
        logger.log_agent_ready(
            agent_name=self.config.agent.name,
            agent_type="single",
            config={
                "llm_provider": self.config.llm.provider,
                "llm_model": self.config.llm.model,
                "tools_enabled": len(self.config.tools.built_in) + len(self.config.tools.custom),
                "memory_enabled": bool(self.memory_manager),
                "evaluation_enabled": bool(self.evaluation_manager)
            }
        )
        
        logger.info(
            "Agent initialization completed successfully",
            extra={
                "agent_id": self.agent_id,
                "agent_name": self.config.agent.name,
                "components_initialized": {
                    "llm": bool(self.llm),
                    "tools": len(self.config.tools.built_in) + len(self.config.tools.custom),
                    "memory": bool(self.memory_manager),
                    "evaluation": bool(self.evaluation_manager),
                    "graph": bool(self.graph)
                }
            }
        )
    
    def _setup_llm(self):
        """Initialize the LLM based on configuration using init_chat_model."""
        llm_config = self.config.llm
        
        logger.debug(
            "Initializing LLM",
            extra={
                "provider": llm_config.provider,
                "model": llm_config.model,
                "temperature": llm_config.temperature,
                "max_tokens": llm_config.max_tokens
            }
        )
        
        # Get API key from environment
        api_key = os.getenv(llm_config.api_key_env)
        if not api_key:
            logger.error(
                "API key not found",
                extra={
                    "api_key_env": llm_config.api_key_env,
                    "provider": llm_config.provider
                }
            )
            raise ValueError(f"API key not found in environment variable: {llm_config.api_key_env}")
        
        # Map provider names to init_chat_model format
        provider_mapping = {
            "openai": "openai",
            "anthropic": "anthropic", 
            "gemini": "google_vertexai",  # Google Gemini models
            "groq": "groq"
        }
        
        provider = provider_mapping.get(llm_config.provider.lower())
        if not provider:
            logger.error(
                "Unsupported LLM provider",
                extra={
                    "provider": llm_config.provider,
                    "supported_providers": list(provider_mapping.keys())
                }
            )
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")
        
        # Prepare model name with provider prefix
        model_name = f"{provider}:{llm_config.model}"
        
        # Prepare additional kwargs
        llm_kwargs = {
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
            "api_key": api_key
        }
        
        # Add base_url if specified
        if llm_config.base_url:
            llm_kwargs["base_url"] = llm_config.base_url
        
        # Initialize LLM using init_chat_model
        try:
            self.llm = init_chat_model(
                model=model_name,
                **llm_kwargs
            )
            logger.info(
                "LLM initialized successfully",
                extra={
                    "provider": llm_config.provider,
                    "model": llm_config.model,
                    "model_name": model_name,
                    "has_api_key": bool(api_key)
                }
            )
            logger.log_llm_call(provider=llm_config.provider, model=llm_config.model)
        except Exception as e:
            logger.error(
                "Failed to initialize LLM",
                extra={
                    "provider": llm_config.provider,
                    "model": llm_config.model,
                    "model_name": model_name,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise ValueError(f"Failed to initialize LLM with model {model_name}: {str(e)}")
    
    def _setup_tools(self):
        """Setup tools from configuration."""
        tools_config = self.config.tools
        
        # Register custom tools
        for custom_tool in tools_config.custom:
            self.tool_registry.register_custom_tool(
                name=custom_tool.name,
                module_path=custom_tool.module_path,
                class_name=custom_tool.class_name,
                description=custom_tool.description,
                parameters=custom_tool.parameters
            )
        
        # Validate all required tools exist
        all_tool_names = tools_config.built_in + [t.name for t in tools_config.custom]
        missing_tools = self.tool_registry.validate_tools(all_tool_names)
        if missing_tools:
            raise ValueError(f"Missing tools: {missing_tools}")
    
    def _setup_memory(self):
        """Setup memory management if enabled."""
        if self.config.memory.enabled:
            self.memory_manager = MemoryManager(self.config.memory)
    
    def _setup_evaluation(self):
        """Setup evaluation management if enabled."""
        if self.config.evaluation.enabled:
            logger.debug("Initializing evaluation manager", extra={"evaluators_count": len(self.config.evaluation.evaluators)})
            try:
                self.evaluation_manager = EvaluationManager(self.config.evaluation)
                logger.info(
                    "Evaluation manager initialized successfully",
                    extra={
                        "evaluators_count": len(self.config.evaluation.evaluators),
                        "auto_evaluate": self.config.evaluation.auto_evaluate,
                        "evaluation_frequency": self.config.evaluation.evaluation_frequency
                    }
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize evaluation manager",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "suggestions": [
                            "Set the LANGSMITH_API_KEY environment variable",
                            "Disable LangSmith integration in evaluation config", 
                            "Disable evaluation entirely"
                        ]
                    },
                    exc_info=True
                )
                self.evaluation_manager = None
        else:
            logger.debug("Evaluation disabled in configuration")
    
    def _setup_graph(self):
        """Setup the LangGraph workflow using prebuilt ReAct agent."""
        # Get all configured tools
        all_tool_names = self.config.tools.built_in + [t.name for t in self.config.tools.custom]
        tools = self.tool_registry.get_tools_by_names(all_tool_names)
        
        # Create system message from configuration
        system_prompt = self.config_loader.get_prompt_template(
            "system_prompt", 
            query="", 
            memory_context="",
            programming_language="", 
            project_context="",
            customer_info="",
            knowledge_base=""
        )
        
        # For Groq models, we need to handle system prompts differently
        if self.config.llm.provider.lower() == "groq":
            # Create the ReAct agent without tools first for Groq
            self.graph = create_react_agent(
                model=self.llm,
                tools=[]  # Start without tools for Groq
            )
        else:
            # Create the ReAct agent with tools for other providers
            self.graph = create_react_agent(
                model=self.llm,
                tools=tools
            )
        
        # Store system prompt and tools for use in run method
        self.system_prompt = system_prompt
        self.available_tools = tools
    
    def run(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Run the agent with given input."""
        if not self.graph:
            logger.error("Graph not initialized", extra={"agent_id": self.agent_id})
            raise ValueError("Graph not initialized")
        
        execution_id = str(uuid.uuid4())[:8]
        
        with CorrelationContext(execution_id):
            logger.log_agent_execution(
                query=input_text,
                agent_name=self.config.agent.name,
                execution_id=execution_id
            )
            
            start_time = time.time()
            
            # Add memory context if available
            enhanced_input = input_text
            memory_context_used = False
            if self.memory_manager:
                logger.debug("Retrieving memory context", extra={"execution_id": execution_id})
                memory_context = self.memory_manager.get_relevant_context(input_text)
                if memory_context:
                    enhanced_input = f"{input_text}\n\nRelevant context: {memory_context}"
                    memory_context_used = True
                    logger.debug(
                        "Memory context added to input",
                        extra={
                            "execution_id": execution_id,
                            "context_length": len(memory_context)
                        }
                    )
        
            try:
                # Prepare messages with system prompt
                messages = []
                if hasattr(self, 'system_prompt') and self.system_prompt:
                    messages.append(SystemMessage(content=self.system_prompt))
                messages.append(HumanMessage(content=enhanced_input))
                
                logger.debug(
                    "Prepared messages for LLM",
                    extra={
                        "execution_id": execution_id,
                        "messages_count": len(messages),
                        "has_system_prompt": bool(hasattr(self, 'system_prompt') and self.system_prompt),
                        "enhanced_input_length": len(enhanced_input),
                        "memory_context_used": memory_context_used
                    }
                )
                
                # Handle Groq differently - use direct LLM call instead of ReAct
                if self.config.llm.provider.lower() == "groq":
                    logger.debug("Using direct LLM call for Groq provider", extra={"execution_id": execution_id})
                    # For Groq, use direct LLM call without tools to avoid function calling issues
                    response = self.llm.invoke(messages)
                    result = {"messages": [response]}
                else:
                    logger.debug("Using ReAct agent execution", extra={"execution_id": execution_id})
                    # Run the ReAct agent for other providers
                    result = self.graph.invoke({"messages": messages})
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                logger.debug(
                    "LLM execution completed",
                    extra={
                        "execution_id": execution_id,
                        "execution_time": execution_time,
                        "provider": self.config.llm.provider
                    }
                )
                
                # Extract response from ReAct agent result
                messages = result.get("messages", [])
                last_message = messages[-1] if messages else None
                
                if last_message:
                    if hasattr(last_message, 'content'):
                        response_text = last_message.content
                    else:
                        response_text = last_message.get("content", "No response")
                else:
                    response_text = "No response"
                
                # Extract message contents safely
                message_contents = []
                for msg in messages:
                    if hasattr(msg, 'content'):
                        message_contents.append(msg.content)
                    elif isinstance(msg, dict):
                        message_contents.append(msg.get("content", ""))
                    else:
                        message_contents.append(str(msg))
                
                # Store interaction in memory if available
                if self.memory_manager and messages:
                    user_msg = messages[0] if messages else HumanMessage(content=input_text)
                    ai_response = last_message if last_message else None
                    if ai_response:
                        self.memory_manager.store_interaction([user_msg], ai_response)
                
                response = {
                    "response": response_text,
                    "messages": message_contents,
                    "tool_results": self._get_tool_results_for_provider(messages),
                    "iteration_count": self._get_iteration_count_for_provider(messages),
                    "response_time": execution_time,
                    "metadata": kwargs
                }
                
                # Log successful response
                logger.log_agent_response(
                    response=response_text,
                    agent_name=self.config.agent.name,
                    execution_id=execution_id,
                    execution_time=execution_time
                )
                
                logger.info(
                    "Agent execution completed successfully",
                    extra={
                        "execution_id": execution_id,
                        "agent_name": self.config.agent.name,
                        "response_length": len(response_text),
                        "execution_time": execution_time,
                        "tool_calls": len(response["tool_results"]),
                        "iterations": response["iteration_count"],
                        "memory_used": memory_context_used
                    }
                )
                
                # Run evaluation if enabled and configured for per-run evaluation
                if (self.evaluation_manager and 
                    self.config.evaluation.auto_evaluate and 
                    self.config.evaluation.evaluation_frequency == "per_run"):
                    
                    try:
                        input_data = {"query": input_text, "input": input_text}
                        output_data = response.copy()
                        
                        evaluation_results = self.evaluation_manager.evaluate_single(
                            input_data=input_data,
                            output_data=output_data
                        )
                        
                        response["evaluation"] = evaluation_results
                        
                    except Exception as eval_error:
                        response["evaluation"] = {
                            "error": f"Evaluation failed: {str(eval_error)}"
                        }
                
                return response
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                
                logger.error(
                    "Agent execution failed",
                    extra={
                        "execution_id": execution_id,
                        "agent_name": self.config.agent.name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "execution_time": execution_time,
                        "input_length": len(input_text)
                    },
                    exc_info=True
                )
                
                return {
                    "error": str(e),
                    "response": f"Error running agent: {str(e)}",
                    "messages": [],
                    "tool_results": {},
                    "iteration_count": 0,
                    "response_time": execution_time,
                    "metadata": {}
                }
    
    def _extract_tool_results(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Extract tool results from message history."""
        tool_results = {}
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('name', 'unknown_tool')
                    tool_results[tool_name] = tool_call.get('args', {})
        return tool_results
    
    def _get_tool_results_for_provider(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Get tool results based on provider."""
        if self.config.llm.provider.lower() == "groq":
            return {}
        return self._extract_tool_results(messages)
    
    def _get_iteration_count_for_provider(self, messages: List[BaseMessage]) -> int:
        """Get iteration count based on provider."""
        if self.config.llm.provider.lower() == "groq":
            return 0
        return len([m for m in messages if hasattr(m, 'tool_calls') and m.tool_calls])
    
    async def arun(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """Async version of run."""
        if not self.graph:
            raise ValueError("Graph not initialized")
        
        # Add memory context if available
        enhanced_input = input_text
        if self.memory_manager:
            memory_context = self.memory_manager.get_relevant_context(input_text)
            if memory_context:
                enhanced_input = f"{input_text}\n\nRelevant context: {memory_context}"
        
        try:
            # Prepare messages with system prompt
            messages = []
            if hasattr(self, 'system_prompt') and self.system_prompt:
                messages.append(SystemMessage(content=self.system_prompt))
            messages.append(HumanMessage(content=enhanced_input))
            
            # Handle Groq differently - use direct LLM call instead of ReAct
            if self.config.llm.provider.lower() == "groq":
                # For Groq, use direct LLM call without tools to avoid function calling issues
                response = await self.llm.ainvoke(messages)
                result = {"messages": [response]}
            else:
                # Run the ReAct agent asynchronously for other providers
                result = await self.graph.ainvoke({"messages": messages})
            
            # Extract response from ReAct agent result
            messages = result.get("messages", [])
            last_message = messages[-1] if messages else None
            
            if last_message:
                if hasattr(last_message, 'content'):
                    response_text = last_message.content
                else:
                    response_text = last_message.get("content", "No response")
            else:
                response_text = "No response"
            
            # Extract message contents safely
            message_contents = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    message_contents.append(msg.content)
                elif isinstance(msg, dict):
                    message_contents.append(msg.get("content", ""))
                else:
                    message_contents.append(str(msg))
            
            # Store interaction in memory if available
            if self.memory_manager and messages:
                user_msg = messages[0] if messages else HumanMessage(content=input_text)
                ai_response = last_message if last_message else None
                if ai_response:
                    self.memory_manager.store_interaction([user_msg], ai_response)
            
            response = {
                "response": response_text,
                "messages": message_contents,
                "tool_results": self._get_tool_results_for_provider(messages),
                "iteration_count": self._get_iteration_count_for_provider(messages),
                "metadata": kwargs
            }
            
            return response
            
        except Exception as e:
            return {
                "error": str(e),
                "response": f"Error running agent: {str(e)}",
                "messages": [],
                "tool_results": {},
                "iteration_count": 0,
                "metadata": {}
            }
    
    def get_prompt_template(self, prompt_type: str, **variables) -> str:
        """Get a formatted prompt template."""
        return self.config_loader.get_prompt_template(prompt_type, **variables)
    
    def get_available_tools(self) -> Dict[str, str]:
        """Get list of available tools."""
        return self.tool_registry.list_all_tools()
    
    def get_config(self) -> AgentConfiguration:
        """Get the loaded configuration."""
        return self.config
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if self.memory_manager:
            return self.memory_manager.get_stats()
        return {"memory_enabled": False}
    
    def update_prompts(self, prompt_updates: Dict[str, str]):
        """Update prompt templates dynamically."""
        for prompt_type, new_template in prompt_updates.items():
            if hasattr(self.config.prompts, prompt_type):
                getattr(self.config.prompts, prompt_type).template = new_template
    
    def reload_config(self, config_file: str = None):
        """Reload configuration and reinitialize components."""
        if config_file:
            self.config = self.config_loader.load_config(config_file)
        self._initialize_components()
    
    def export_conversation(self, format_type: str = "json") -> str:
        """Export conversation history."""
        if not self.memory_manager:
            return "Memory not enabled"
        
        return self.memory_manager.export_history(format_type)
    
    def clear_memory(self):
        """Clear agent memory."""
        if self.memory_manager:
            self.memory_manager.clear_memory()
    
    def evaluate_single(self, input_text: str, expected_output: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Evaluate a single input against the agent."""
        if not self.evaluation_manager:
            return {"error": "Evaluation not enabled"}
        
        # Run the agent
        result = self.run(input_text, **kwargs)
        
        # Prepare evaluation data
        input_data = {"query": input_text, "input": input_text}
        output_data = result.copy()
        
        # Run evaluation
        return self.evaluation_manager.evaluate_single(
            input_data=input_data,
            output_data=output_data,
            reference_output=expected_output
        )
    
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Get evaluation metrics summary."""
        if not self.evaluation_manager:
            return {"error": "Evaluation not enabled"}
        return self.evaluation_manager.get_metrics_summary()
    
    def create_evaluation_dataset(self, dataset_name: str, examples: List[Dict[str, Any]], description: str = "") -> str:
        """Create an evaluation dataset."""
        if not self.evaluation_manager:
            raise ValueError("Evaluation not enabled")
        
        from ..core.config_loader import DatasetConfig
        dataset_config = DatasetConfig(
            name=dataset_name,
            description=description,
            examples=examples
        )
        
        return self.evaluation_manager.create_dataset(dataset_config)
    
    def run_dataset_evaluation(self, dataset_name: str, evaluator_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run evaluation against a dataset."""
        if not self.evaluation_manager:
            raise ValueError("Evaluation not enabled")
        
        def target_function(inputs):
            query = inputs.get("query") or inputs.get("input", "")
            result = self.run(query)
            return {"answer": result["response"], "metadata": result.get("metadata", {})}
        
        return self.evaluation_manager.evaluate_dataset(
            target_function=target_function,
            dataset_name=dataset_name,
            evaluator_names=evaluator_names
        )