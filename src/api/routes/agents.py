"""
Agent management API endpoints.
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import asyncio

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, Path, Body
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from ..models.requests import AgentCreateRequest, AgentUpdateRequest, AgentRunRequest
from ..models.responses import (
    AgentResponse, AgentListResponse, AgentRunResponse, ErrorResponse,
    AgentInfo, AgentConfig, AgentRunResult, StatusEnum, AgentStatus
)
from ..utils.exceptions import (
    AgentNotFoundException, AgentConfigurationError, AgentExecutionError
)
from ...core.configurable_agent import ConfigurableAgent
from ...core.config_loader import AgentConfiguration

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

# In-memory storage for demo (replace with proper database)
agents_store: Dict[str, Dict[str, Any]] = {}
agent_instances: Dict[str, ConfigurableAgent] = {}


def get_agent_or_404(agent_id: str) -> Dict[str, Any]:
    """Get agent or raise 404."""
    if agent_id not in agents_store:
        raise AgentNotFoundException(agent_id)
    return agents_store[agent_id]


def validate_agent_prerequisites(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate agent prerequisites before creation."""
    import os
    
    validation_errors = {}
    warnings = []
    
    # Check API key availability
    llm_config = config_dict.get("llm", {})
    provider = llm_config.get("provider", "").lower()
    use_vertex = bool(llm_config.get("project") or llm_config.get("location"))
    
    # Map provider to expected environment variable
    api_key_mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", 
        "gemini": "GOOGLE_API_KEY",
        "google": "GOOGLE_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
        "google_vertexai": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY"
    }
    
    required_api_key = api_key_mapping.get(provider)
    # For Google providers: if using Vertex AI (project/location), do not require GOOGLE_API_KEY
    if provider in ["google", "gemini", "google_vertexai", "google_genai"]:
        if not use_vertex and not os.getenv("GOOGLE_API_KEY"):
            validation_errors["api_key"] = (
                "Missing GOOGLE_API_KEY environment variable for Google GenAI usage. "
                "Provide project/location to use Vertex AI without an API key."
            )
    elif required_api_key and not os.getenv(required_api_key):
        validation_errors["api_key"] = f"Missing {required_api_key} environment variable for provider '{provider}'"
    
    # Check tools configuration
    tools_config = config_dict.get("tools", {})
    built_in_tools = tools_config.get("built_in", [])
    
    if "web_search" in built_in_tools and not os.getenv("SERPER_API_KEY"):
        warnings.append("SERPER_API_KEY not found - web search tool may not work properly")
    
    # Check required config sections
    required_sections = ["agent", "llm", "prompts", "tools"]
    for section in required_sections:
        if section not in config_dict:
            validation_errors[section] = f"Missing required configuration section: {section}"
    
    # Check Google provider specific requirements
    if provider in ["google", "gemini", "google_vertexai", "google_genai"]:
        if use_vertex:
            try:
                import langchain_google_vertexai  # noqa: F401
            except ImportError:
                validation_errors["dependency"] = (
                    "Google Vertex AI usage requires 'langchain-google-vertexai'. "
                    "Install with: pip install langchain-google-vertexai"
                )
        else:
            try:
                import langchain_google_genai  # noqa: F401
            except ImportError:
                validation_errors["dependency"] = (
                    "Google GenAI usage requires 'langchain-google-genai'. "
                    "Install with: pip install langchain-google-genai"
                )
    
    return {
        "errors": validation_errors,
        "warnings": warnings,
        "is_valid": len(validation_errors) == 0
    }


def create_agent_instance(agent_id: str, config_dict: Dict[str, Any]) -> ConfigurableAgent:
    """Create agent instance from configuration."""
    import tempfile
    import yaml
    import os
    
    temp_config_path = None
    
    try:
        # Pre-validate configuration
        validation_result = validate_agent_prerequisites(config_dict)
        if not validation_result["is_valid"]:
            error_details = "; ".join([f"{k}: {v}" for k, v in validation_result["errors"].items()])
            raise AgentConfigurationError(f"Configuration validation failed: {error_details}")
        
        # Log warnings
        for warning in validation_result["warnings"]:
            logger.warning(f"Agent {agent_id} configuration warning: {warning}")
        
        # Convert request to AgentConfiguration for additional validation
        try:
            agent_config = AgentConfiguration(**config_dict)
        except Exception as e:
            raise AgentConfigurationError(f"Invalid configuration structure: {str(e)}")
        
        # Create temporary YAML file for ConfigurableAgent
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            temp_config_path = f.name
        
        # Create agent instance
        try:
            logger.info(f"Creating agent instance {agent_id} with config from {temp_config_path}")
            agent = ConfigurableAgent(temp_config_path)
            agent_instances[agent_id] = agent
            logger.info(f"Successfully created agent instance {agent_id}")
            return agent
            
        except Exception as e:
            logger.error(f"ConfigurableAgent initialization failed for {agent_id}: {str(e)}")
            # Provide more specific error messages based on common failure modes
            if "API key" in str(e):
                raise AgentConfigurationError(f"API key error: {str(e)}")
            elif "provider" in str(e).lower():
                raise AgentConfigurationError(f"LLM provider error: {str(e)}")
            elif "tool" in str(e).lower():
                raise AgentConfigurationError(f"Tool configuration error: {str(e)}")
            else:
                raise AgentConfigurationError(f"Agent initialization failed: {str(e)}")
            
    except AgentConfigurationError:
        # Re-raise configuration errors as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating agent instance {agent_id}: {str(e)}", exc_info=True)
        raise AgentConfigurationError(f"Unexpected error during agent creation: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_config_path and os.path.exists(temp_config_path):
            try:
                os.unlink(temp_config_path)
                logger.debug(f"Cleaned up temporary config file: {temp_config_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary file {temp_config_path}: {cleanup_error}")


@router.post(
    "/", 
    response_model=AgentResponse,
    status_code=201,
    summary="Create Agent",
    description="Create a new configurable agent with specified LLM, tools, and configuration",
    responses={
        201: {
            "description": "Agent created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "agent": {
                            "id": "550e8400-e29b-41d4-a716-446655440000",
                            "name": "Research Assistant",
                            "description": "AI agent specialized in research",
                            "version": "1.0.0",
                            "status": "active",
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": "2024-01-01T00:00:00Z",
                            "tags": ["research", "assistant"]
                        },
                        "config": {
                            "llm": {
                                "provider": "openai",
                                "model": "gpt-4",
                                "temperature": 0.7
                            },
                            "tools": ["web_search", "calculator"],
                            "debug_mode": False
                        }
                    }
                }
            }
        },
        400: {"description": "Invalid configuration"},
        500: {"description": "Internal server error"}
    }
)
async def create_agent(request: AgentCreateRequest):
    """
    Create a new configurable agent.
    
    Creates a new agent with the specified configuration including:
    - LLM provider and model settings
    - System prompts and variables
    - Available tools (built-in and custom)
    - Memory configuration
    - Evaluation settings
    - ReAct loop parameters
    
    **Supported LLM Providers:**
    - `openai`: Requires OPENAI_API_KEY environment variable
    - `anthropic`: Requires ANTHROPIC_API_KEY environment variable
    - `google`: Google Vertex AI models (requires GOOGLE_API_KEY and langchain-google-vertexai)
    - `groq`: Requires GROQ_API_KEY environment variable
    
    **Google Provider Notes:**
    - If `project`/`location` are provided in `llm`, uses Google Vertex AI via `langchain-google-vertexai` (ADC or service account auth)
    - Otherwise uses Google Generative AI via `langchain-google-genai` with `GOOGLE_API_KEY`
    - Models like `gemini-1.5-flash` and `gemini-1.5-pro` are supported. If the Vertex AI model isn't available, the system will fallback to GenAI when an API key is present.
    
    Args:
        request (AgentCreateRequest): Agent configuration request
        
    Returns:
        AgentResponse: Created agent information and configuration
        
    Raises:
        HTTPException: If agent creation fails due to invalid configuration, 
                      missing dependencies, or missing API keys
    """
    try:
        agent_id = str(uuid.uuid4())
        
        # Convert request to dict for storage
        config_dict = {
            "agent": {
                "name": request.name,
                "description": request.description or "",
                "version": request.version
            },
            "llm": {
                "provider": request.llm.provider.value,
                "model": request.llm.model,
                "temperature": request.llm.temperature,
                "max_tokens": request.llm.max_tokens,
                "api_key_env": f"{request.llm.provider.value.upper()}_API_KEY",
                # Optional routing hints for Google Vertex AI
                "project": getattr(request.llm, "project", None),
                "location": getattr(request.llm, "location", None),
            },
            "prompts": {
                "system_prompt": {
                    "template": request.prompts.system_prompt,
                    "variables": list(request.prompts.variables.keys()) if isinstance(request.prompts.variables, dict) else []
                },
                "user_prompt": {
                    "template": "User query: {query}",
                    "variables": ["query"]
                }
            },
            "tools": {
                "built_in": [tool if isinstance(tool, str) else tool.name for tool in request.tools],
                "custom": []
            },
            "memory": {
                "enabled": len(request.memory) > 0,
                "provider": "langmem",
                "types": {
                    "semantic": any(mem.type.value == "semantic" for mem in request.memory),
                    "episodic": any(mem.type.value == "episodic" for mem in request.memory),
                    "procedural": any(mem.type.value == "procedural" for mem in request.memory)
                },
                "storage": {"backend": "memory"},
                "settings": {"max_memory_size": 5000}
            },
            "react": {
                "max_iterations": request.react.max_iterations,
                "recursion_limit": request.react.recursion_limit
            },
            "optimization": {
                "enabled": False,
                "prompt_optimization": {"enabled": False},
                "performance_tracking": {"enabled": False}
            },
            "runtime": {
                "max_iterations": 50,
                "timeout_seconds": 300,
                "retry_attempts": 3,
                "debug_mode": request.debug_mode
            },
            "evaluation": {
                "enabled": request.evaluation.enabled,
                "langsmith": {"enabled": False},
                "evaluators": [
                    {
                        "name": eval.name,
                        "type": eval.type.value,
                        "parameters": eval.parameters,
                        "enabled": True
                    }
                    for eval in request.evaluation.evaluators
                ],
                "datasets": [],
                "metrics": request.evaluation.metrics,
                "auto_evaluate": request.evaluation.auto_evaluate,
                "evaluation_frequency": "manual",
                "batch_size": 10,
                "max_concurrency": 2
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "format": "structured",
                "console": {"enabled": True, "level": "INFO", "format": "structured"},
                "file": {"enabled": False},
                "components": {},
                "correlation": {"enabled": True},
                "performance": {},
                "privacy": {},
                "custom": {}
            }
        }
        
        # Store agent
        now = datetime.utcnow()
        agent_data = {
            "id": agent_id,
            "config": config_dict,
            "status": AgentStatus.CONFIGURING.value,
            "created_at": now,
            "updated_at": now,
            "tags": request.tags,
            "run_count": 0
        }
        
        agents_store[agent_id] = agent_data
        
        # Try to create agent instance
        try:
            create_agent_instance(agent_id, config_dict)
            agent_data["status"] = AgentStatus.ACTIVE.value
            agent_data["error_message"] = None
            
        except AgentConfigurationError as config_error:
            agent_data["status"] = AgentStatus.ERROR.value
            agent_data["error_message"] = config_error.message
            logger.error(f"Configuration error for agent {agent_id}: {config_error.message}")
            
            # Return error response immediately for configuration errors
            error_response = {
                "status": "error",
                "error_code": config_error.error_code,
                "message": config_error.message,
                "details": config_error.details,
                "agent_id": agent_id
            }
            raise HTTPException(status_code=config_error.status_code, detail=error_response)
            
        except Exception as e:
            agent_data["status"] = AgentStatus.ERROR.value
            agent_data["error_message"] = f"Unexpected error: {str(e)}"
            logger.error(f"Failed to initialize agent {agent_id}: {e}", exc_info=True)
            
            # Return error response for unexpected errors
            raise HTTPException(
                status_code=500, 
                detail={
                    "status": "error",
                    "error_code": "AGENT_INITIALIZATION_ERROR",
                    "message": f"Failed to initialize agent: {str(e)}",
                    "agent_id": agent_id
                }
            )
        
        # Create response
        agent_info = AgentInfo(
            id=agent_id,
            name=request.name,
            description=request.description,
            version=request.version,
            status=AgentStatus(agent_data["status"]),
            created_at=now,
            updated_at=now,
            tags=request.tags
        )
        
        agent_config = AgentConfig(
            llm=config_dict["llm"],
            prompts=config_dict["prompts"],
            tools=config_dict["tools"]["built_in"],
            memory=[config_dict["memory"]] if config_dict["memory"] else [],
            evaluation=config_dict["evaluation"],
            react=config_dict["react"],
            debug_mode=config_dict["runtime"]["debug_mode"]
        )
        
        return AgentResponse(
            agent=agent_info,
            config=agent_config
        )
        
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/", 
    response_model=AgentListResponse,
    summary="List Agents",
    description="Retrieve a paginated list of agents with optional filtering",
    responses={
        200: {
            "description": "List of agents retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "agents": [
                            {
                                "id": "550e8400-e29b-41d4-a716-446655440000", 
                                "name": "Research Assistant",
                                "description": "AI research agent",
                                "version": "1.0.0",
                                "status": "active",
                                "created_at": "2024-01-01T00:00:00Z",
                                "updated_at": "2024-01-01T00:00:00Z",
                                "tags": ["research"]
                            }
                        ],
                        "total": 1,
                        "page": 1,
                        "page_size": 20
                    }
                }
            }
        }
    }
)
async def list_agents(
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of agents per page"),
    status: Optional[AgentStatus] = Query(None, description="Filter by agent status"),
    tags: Optional[str] = Query(None, description="Comma-separated tags to filter by")
):
    """
    List all agents with pagination and filtering.
    
    Retrieves a paginated list of agents with optional filtering by:
    - Status (active, configuring, error, inactive)
    - Tags (comma-separated list)
    
    Args:
        page (int): Page number (1-based)
        page_size (int): Number of agents per page (1-100)
        status (AgentStatus, optional): Filter by agent status
        tags (str, optional): Comma-separated tags to filter by
        
    Returns:
        AgentListResponse: Paginated list of agents with metadata
    """
    try:
        # Filter agents
        filtered_agents = list(agents_store.values())
        
        if status:
            filtered_agents = [a for a in filtered_agents if a["status"] == status.value]
            
        if tags:
            tag_list = [t.strip() for t in tags.split(",")]
            filtered_agents = [
                a for a in filtered_agents 
                if any(tag in a.get("tags", []) for tag in tag_list)
            ]
        
        # Pagination
        total = len(filtered_agents)
        start = (page - 1) * page_size
        end = start + page_size
        page_agents = filtered_agents[start:end]
        
        # Convert to response format
        agent_infos = []
        for agent_data in page_agents:
            config = agent_data["config"]
            agent_info = AgentInfo(
                id=agent_data["id"],
                name=config["agent"]["name"],
                description=config["agent"]["description"],
                version=config["agent"]["version"],
                status=AgentStatus(agent_data["status"]),
                created_at=agent_data["created_at"],
                updated_at=agent_data["updated_at"],
                tags=agent_data.get("tags", [])
            )
            agent_infos.append(agent_info)
        
        return AgentListResponse(
            agents=agent_infos,
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{agent_id}", 
    response_model=AgentResponse,
    summary="Get Agent",
    description="Retrieve detailed information about a specific agent",
    responses={
        200: {
            "description": "Agent details retrieved successfully"
        },
        404: {
            "description": "Agent not found",
            "content": {
                "application/json": {
                    "example": {
                        "error": "AGENT_NOT_FOUND",
                        "message": "Agent 550e8400-e29b-41d4-a716-446655440000 not found"
                    }
                }
            }
        }
    }
)
async def get_agent(agent_id: str = Path(..., description="Unique identifier for the agent")):
    """
    Get detailed information about a specific agent.
    
    Retrieves complete agent information including:
    - Agent metadata (name, description, status, etc.)
    - Full configuration (LLM, prompts, tools, memory)
    - Runtime settings and debug mode
    
    Args:
        agent_id (str): Unique identifier for the agent
        
    Returns:
        AgentResponse: Complete agent information and configuration
        
    Raises:
        HTTPException: 404 if agent not found
    """
    try:
        agent_data = get_agent_or_404(agent_id)
        config = agent_data["config"]
        
        agent_info = AgentInfo(
            id=agent_id,
            name=config["agent"]["name"],
            description=config["agent"]["description"],
            version=config["agent"]["version"],
            status=AgentStatus(agent_data["status"]),
            created_at=agent_data["created_at"],
            updated_at=agent_data["updated_at"],
            tags=agent_data.get("tags", [])
        )
        
        agent_config = AgentConfig(
            llm=config["llm"],
            prompts=config["prompts"],
            tools=config["tools"]["built_in"],
            memory=[config["memory"]] if config["memory"] else [],
            evaluation=config["evaluation"],
            react=config["react"],
            debug_mode=config["runtime"]["debug_mode"]
        )
        
        return AgentResponse(
            agent=agent_info,
            config=agent_config
        )
        
    except AgentNotFoundException:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put(
    "/{agent_id}", 
    response_model=AgentResponse,
    summary="Update Agent",
    description="Update an existing agent configuration",
    responses={
        200: {"description": "Agent updated successfully"},
        404: {"description": "Agent not found"},
        400: {"description": "Invalid configuration"}
    }
)
async def update_agent(
    agent_id: str = Path(..., description="Unique identifier for the agent"), 
    request: AgentUpdateRequest = Body(..., description="Updated agent configuration")
):
    """
    Update an existing agent configuration.
    
    Updates the specified agent with new configuration. Only provided fields
    will be updated, others remain unchanged. After update, the agent instance
    is recreated to apply changes.
    
    Args:
        agent_id (str): Unique identifier for the agent
        request (AgentUpdateRequest): Updated configuration fields
        
    Returns:
        AgentResponse: Updated agent information and configuration
        
    Raises:
        HTTPException: 404 if agent not found, 400 if invalid configuration
    """
    try:
        agent_data = get_agent_or_404(agent_id)
        config = agent_data["config"]
        
        # Update configuration
        if request.name is not None:
            config["agent"]["name"] = request.name
        if request.description is not None:
            config["agent"]["description"] = request.description
        if request.version is not None:
            config["agent"]["version"] = request.version
            
        if request.llm is not None:
            config["llm"].update({
                "provider": request.llm.provider.value,
                "model": request.llm.model,
                "temperature": request.llm.temperature,
                "max_tokens": request.llm.max_tokens,
                "top_p": request.llm.top_p,
                "project": getattr(request.llm, "project", None),
                "location": getattr(request.llm, "location", None),
            })
            
        if request.prompts is not None:
            config["prompts"].update({
                "system_prompt": request.prompts.system_prompt,
                "variables": request.prompts.variables
            })
        
        # Update other fields as needed...
        
        agent_data["updated_at"] = datetime.utcnow()
        agent_data["status"] = AgentStatus.CONFIGURING.value
        
        # Recreate agent instance
        if agent_id in agent_instances:
            del agent_instances[agent_id]
            
        try:
            create_agent_instance(agent_id, config)
            agent_data["status"] = AgentStatus.ACTIVE.value
            agent_data["error_message"] = None
            
        except AgentConfigurationError as config_error:
            agent_data["status"] = AgentStatus.ERROR.value
            agent_data["error_message"] = config_error.message
            logger.error(f"Configuration error updating agent {agent_id}: {config_error.message}")
            
            # Return error response immediately for configuration errors
            raise HTTPException(
                status_code=config_error.status_code,
                detail={
                    "error": config_error.error_code,
                    "message": f"Update failed: {config_error.message}",
                    "details": config_error.details,
                    "agent_id": agent_id
                }
            )
            
        except Exception as e:
            agent_data["status"] = AgentStatus.ERROR.value
            agent_data["error_message"] = f"Update failed: {str(e)}"
            logger.error(f"Failed to update agent instance {agent_id}: {e}", exc_info=True)
            
            # Return error response for unexpected errors
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "AGENT_UPDATE_ERROR",
                    "message": f"Failed to update agent: {str(e)}",
                    "agent_id": agent_id
                }
            )
        
        # Return updated agent
        agent_info = AgentInfo(
            id=agent_id,
            name=config["agent"]["name"],
            description=config["agent"]["description"],
            version=config["agent"]["version"],
            status=AgentStatus(agent_data["status"]),
            created_at=agent_data["created_at"],
            updated_at=agent_data["updated_at"],
            tags=agent_data.get("tags", [])
        )
        
        agent_config = AgentConfig(
            llm=config["llm"],
            prompts=config["prompts"],
            tools=config["tools"]["built_in"],
            memory=[config["memory"]] if config["memory"] else [],
            evaluation=config["evaluation"],
            react=config["react"],
            debug_mode=config["runtime"]["debug_mode"]
        )
        
        return AgentResponse(
            agent=agent_info,
            config=agent_config
        )
        
    except AgentNotFoundException:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    except Exception as e:
        logger.error(f"Failed to update agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/{agent_id}",
    summary="Delete Agent",
    description="Delete an agent and clean up all associated resources", 
    responses={
        200: {
            "description": "Agent deleted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Agent 550e8400-e29b-41d4-a716-446655440000 deleted"
                    }
                }
            }
        },
        404: {"description": "Agent not found"}
    }
)
async def delete_agent(agent_id: str = Path(..., description="Unique identifier for the agent")):
    """
    Delete an agent and clean up all associated resources.
    
    Permanently removes the agent from storage and cleans up:
    - Agent configuration and metadata
    - Loaded agent instances from memory
    - Associated run history (if applicable)
    
    This operation cannot be undone.
    
    Args:
        agent_id (str): Unique identifier for the agent
        
    Returns:
        dict: Success confirmation message
        
    Raises:
        HTTPException: 404 if agent not found
    """
    try:
        get_agent_or_404(agent_id)
        
        # Remove from storage
        del agents_store[agent_id]
        if agent_id in agent_instances:
            del agent_instances[agent_id]
        
        return {"status": "success", "message": f"Agent {agent_id} deleted"}
        
    except AgentNotFoundException:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    except Exception as e:
        logger.error(f"Failed to delete agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/{agent_id}/run", 
    response_model=AgentRunResponse,
    summary="Run Agent",
    description="Execute an agent with a query and return the response",
    responses={
        200: {
            "description": "Agent executed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "run_id": "run_550e8400-e29b-41d4-a716-446655440000",
                        "agent_id": "550e8400-e29b-41d4-a716-446655440000",
                        "query": "What are the latest AI developments?",
                        "result": {
                            "response": "Based on recent research, the latest AI developments include...",
                            "execution_time": 2.5,
                            "token_usage": {
                                "prompt_tokens": 150,
                                "completion_tokens": 300,
                                "total_tokens": 450
                            },
                            "tool_calls": [
                                {
                                    "tool": "web_search",
                                    "query": "latest AI developments 2024",
                                    "result": "Search results..."
                                }
                            ]
                        },
                        "started_at": "2024-01-01T12:00:00Z",
                        "completed_at": "2024-01-01T12:00:02Z"
                    }
                }
            }
        },
        400: {"description": "Agent not active or invalid request"},
        404: {"description": "Agent not found"},
        500: {"description": "Agent execution failed"}
    }
)
async def run_agent(
    agent_id: str = Path(..., description="Unique identifier for the agent"), 
    request: AgentRunRequest = Body(..., description="Query and execution parameters")
):
    """
    Execute an agent with a query.
    
    Runs the specified agent with the provided query and returns:
    - Agent response text
    - Execution metrics (time, token usage)
    - Tool calls made during execution
    - Memory updates (if applicable)
    - Debug information (if debug mode enabled)
    
    The agent must be in 'active' status to be executed.
    
    Args:
        agent_id (str): Unique identifier for the agent
        request (AgentRunRequest): Query and execution parameters
        
    Returns:
        AgentRunResponse: Execution results with response and metadata
        
    Raises:
        HTTPException: 400 if agent not active, 404 if not found, 500 if execution fails
    """
    try:
        agent_data = get_agent_or_404(agent_id)
        
        # Check agent status and provide detailed feedback
        agent_status = agent_data["status"]
        if agent_status != AgentStatus.ACTIVE.value:
            error_message = f"Agent {agent_id} is not active (status: {agent_status})"
            
            # Provide specific guidance based on status
            if agent_status == AgentStatus.ERROR.value:
                error_message = f"Agent {agent_id} is in error state"
                agent_error = agent_data.get("error_message", "Unknown error")
                error_detail = {
                    "error": "AGENT_NOT_ACTIVE",
                    "message": error_message,
                    "agent_status": agent_status,
                    "error_details": agent_error,
                    "suggestions": [
                        "Check agent configuration and fix any issues",
                        "Ensure required API keys are available",
                        "Try recreating the agent with correct configuration"
                    ]
                }
            elif agent_status == AgentStatus.CONFIGURING.value:
                error_detail = {
                    "error": "AGENT_NOT_READY",
                    "message": f"Agent {agent_id} is still configuring",
                    "agent_status": agent_status,
                    "suggestions": [
                        "Wait for agent configuration to complete",
                        "Check agent status endpoint for updates"
                    ]
                }
            else:
                error_detail = {
                    "error": "AGENT_NOT_ACTIVE", 
                    "message": error_message,
                    "agent_status": agent_status,
                    "suggestions": [
                        "Activate the agent before running queries",
                        "Check agent configuration"
                    ]
                }
            
            raise HTTPException(status_code=400, detail=error_detail)
        
        # Get agent instance
        if agent_id not in agent_instances:
            try:
                create_agent_instance(agent_id, agent_data["config"])
            except AgentConfigurationError as e:
                # Update agent status to error if instance creation fails
                agent_data["status"] = AgentStatus.ERROR.value
                agent_data["error_message"] = e.message
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "AGENT_INSTANCE_ERROR",
                        "message": f"Failed to create agent instance: {e.message}",
                        "agent_id": agent_id
                    }
                )
            except Exception as e:
                agent_data["status"] = AgentStatus.ERROR.value
                agent_data["error_message"] = f"Failed to create instance: {str(e)}"
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "AGENT_INSTANCE_ERROR", 
                        "message": f"Failed to create agent instance: {str(e)}",
                        "agent_id": agent_id
                    }
                )
        
        agent = agent_instances[agent_id]
        
        # Run agent
        run_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        
        try:
            import time
            start_time = time.time()
            
            # Run the agent
            agent_result = agent.run(request.query)
            
            execution_time = time.time() - start_time
            completed_at = datetime.utcnow()
            
            # Extract response string from agent result
            if isinstance(agent_result, dict):
                response_text = agent_result.get("response", "No response")
                tool_calls = agent_result.get("tool_results", {})
                token_usage = agent_result.get("token_usage", {})
            else:
                response_text = str(agent_result)
                tool_calls = {}
                token_usage = {}
            
            # Create result
            result = AgentRunResult(
                response=response_text,
                execution_time=execution_time,
                token_usage=token_usage,
                tool_calls=list(tool_calls.values()) if isinstance(tool_calls, dict) else [],
                memory_updates=[], # TODO: Extract from agent
                debug_info=None if not agent_data["config"]["runtime"]["debug_mode"] else {}
            )
            
            # Update run count
            agent_data["run_count"] = agent_data.get("run_count", 0) + 1
            
            return AgentRunResponse(
                run_id=run_id,
                agent_id=agent_id,
                query=request.query,
                result=result,
                started_at=started_at,
                completed_at=completed_at
            )
            
        except Exception as e:
            logger.error(f"Agent execution failed for {agent_id}: {e}")
            raise AgentExecutionError(str(e), agent_id)
            
    except AgentNotFoundException:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    except AgentExecutionError as e:
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to run agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/{agent_id}/stream")
async def stream_agent(
    websocket: WebSocket, 
    agent_id: str = Path(..., description="Unique identifier for the agent")
):
    """
    Stream agent execution via WebSocket.
    
    Provides real-time streaming of agent execution including:
    - Execution start/completion events
    - Intermediate reasoning steps (if available)
    - Tool call results
    - Error notifications
    
    WebSocket Message Format:
    - Client sends: {"type": "run", "query": "your question"}
    - Server sends: {"type": "run_start|run_complete|run_error", "data": {...}}
    
    Args:
        websocket (WebSocket): WebSocket connection
        agent_id (str): Unique identifier for the agent
    """
    await websocket.accept()
    
    try:
        agent_data = get_agent_or_404(agent_id)
        
        if agent_data["status"] != AgentStatus.ACTIVE.value:
            await websocket.send_json({
                "type": "error",
                "data": {
                    "error": "AGENT_NOT_ACTIVE",
                    "message": f"Agent {agent_id} is not active"
                }
            })
            await websocket.close()
            return
        
        # Get agent instance
        if agent_id not in agent_instances:
            create_agent_instance(agent_id, agent_data["config"])
        
        agent = agent_instances[agent_id]
        
        while True:
            # Wait for message from client
            data = await websocket.receive_json()
            
            if data.get("type") == "run":
                query = data.get("query", "")
                run_id = str(uuid.uuid4())
                
                # Send start message
                await websocket.send_json({
                    "type": "run_start",
                    "data": {
                        "run_id": run_id,
                        "query": query,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                })
                
                try:
                    # TODO: Implement streaming agent execution
                    # For now, run synchronously and send result
                    response = agent.run(query)
                    
                    await websocket.send_json({
                        "type": "run_complete",
                        "data": {
                            "run_id": run_id,
                            "response": response,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "run_error",
                        "data": {
                            "run_id": run_id,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    })
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for agent {agent_id}")
    except Exception as e:
        logger.error(f"WebSocket error for agent {agent_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "data": {
                "error": "INTERNAL_ERROR",
                "message": str(e)
            }
        })
        await websocket.close()


@router.post(
    "/{agent_id}/retry",
    response_model=AgentResponse,
    summary="Retry Agent Initialization",
    description="Retry initializing an agent that is in error state",
    responses={
        200: {"description": "Agent initialization retried successfully"},
        400: {"description": "Agent not in error state or invalid configuration"},
        404: {"description": "Agent not found"},
        500: {"description": "Retry failed"}
    }
)
async def retry_agent_initialization(agent_id: str = Path(..., description="Unique identifier for the agent")):
    """
    Retry initializing an agent that is in error state.
    
    This endpoint allows you to retry agent initialization after fixing
    configuration issues such as missing API keys or invalid settings.
    Only agents in ERROR status can be retried.
    
    Args:
        agent_id (str): Unique identifier for the agent
        
    Returns:
        AgentResponse: Updated agent information after retry
        
    Raises:
        HTTPException: 400 if agent not in error state, 404 if not found, 500 if retry fails
    """
    try:
        agent_data = get_agent_or_404(agent_id)
        
        # Check if agent is in error state
        if agent_data["status"] != AgentStatus.ERROR.value:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "AGENT_NOT_IN_ERROR_STATE",
                    "message": f"Agent {agent_id} is not in error state (current status: {agent_data['status']})",
                    "agent_status": agent_data["status"],
                    "suggestions": ["Only agents in ERROR status can be retried"]
                }
            )
        
        # Clear previous error state
        agent_data["status"] = AgentStatus.CONFIGURING.value
        agent_data["error_message"] = None
        agent_data["updated_at"] = datetime.utcnow()
        
        # Remove existing instance if any
        if agent_id in agent_instances:
            del agent_instances[agent_id]
        
        # Try to create agent instance again
        try:
            create_agent_instance(agent_id, agent_data["config"])
            agent_data["status"] = AgentStatus.ACTIVE.value
            logger.info(f"Successfully retried agent initialization for {agent_id}")
            
        except AgentConfigurationError as config_error:
            agent_data["status"] = AgentStatus.ERROR.value
            agent_data["error_message"] = config_error.message
            logger.error(f"Retry failed for agent {agent_id}: {config_error.message}")
            
            raise HTTPException(
                status_code=400,
                detail={
                    "error": config_error.error_code,
                    "message": f"Retry failed: {config_error.message}",
                    "details": config_error.details,
                    "agent_id": agent_id
                }
            )
            
        except Exception as e:
            agent_data["status"] = AgentStatus.ERROR.value
            agent_data["error_message"] = f"Retry failed: {str(e)}"
            logger.error(f"Unexpected error during retry for agent {agent_id}: {e}", exc_info=True)
            
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "AGENT_RETRY_ERROR",
                    "message": f"Retry failed: {str(e)}",
                    "agent_id": agent_id
                }
            )
        
        # Return updated agent information
        config = agent_data["config"]
        agent_info = AgentInfo(
            id=agent_id,
            name=config["agent"]["name"],
            description=config["agent"]["description"],
            version=config["agent"]["version"],
            status=AgentStatus(agent_data["status"]),
            created_at=agent_data["created_at"],
            updated_at=agent_data["updated_at"],
            tags=agent_data.get("tags", [])
        )
        
        agent_config = AgentConfig(
            llm=config["llm"],
            prompts=config["prompts"],
            tools=config["tools"]["built_in"],
            memory=[config["memory"]] if config["memory"] else [],
            evaluation=config["evaluation"],
            react=config["react"],
            debug_mode=config["runtime"]["debug_mode"]
        )
        
        return AgentResponse(
            agent=agent_info,
            config=agent_config
        )
        
    except AgentNotFoundException:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    except Exception as e:
        logger.error(f"Failed to retry agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{agent_id}/status",
    summary="Get Agent Status", 
    description="Get current agent status and basic usage metrics",
    responses={
        200: {
            "description": "Agent status retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "data": {
                            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
                            "status": "active",
                            "run_count": 25,
                            "created_at": "2024-01-01T00:00:00Z",
                            "updated_at": "2024-01-01T12:00:00Z",
                            "instance_loaded": True
                        }
                    }
                }
            }
        },
        404: {"description": "Agent not found"}
    }
)
async def get_agent_status(agent_id: str = Path(..., description="Unique identifier for the agent")):
    """
    Get agent status and basic usage metrics.
    
    Returns current status and metrics including:
    - Current agent status (active, configuring, error, inactive)
    - Total number of runs executed
    - Creation and last update timestamps
    - Whether agent instance is currently loaded in memory
    
    Args:
        agent_id (str): Unique identifier for the agent
        
    Returns:
        dict: Agent status and metrics
        
    Raises:
        HTTPException: 404 if agent not found
    """
    try:
        agent_data = get_agent_or_404(agent_id)
        
        # Prepare status response with conditional error information
        status_data = {
            "agent_id": agent_id,
            "status": agent_data["status"],
            "run_count": agent_data.get("run_count", 0),
            "created_at": agent_data["created_at"].isoformat(),
            "updated_at": agent_data["updated_at"].isoformat(),
            "instance_loaded": agent_id in agent_instances
        }
        
        # Add error information if agent is in error state
        if agent_data["status"] == AgentStatus.ERROR.value:
            status_data["error_message"] = agent_data.get("error_message", "Unknown error")
            status_data["suggestions"] = [
                "Check agent configuration and fix any issues",
                "Ensure required API keys are available in environment",
                f"Use POST /api/agents/{agent_id}/retry to retry initialization",
                "Check server logs for detailed error information"
            ]
        
        # Add configuration summary for active agents
        if agent_data["status"] == AgentStatus.ACTIVE.value:
            config = agent_data["config"]
            status_data["configuration"] = {
                "llm_provider": config["llm"]["provider"],
                "llm_model": config["llm"]["model"],
                "tools_enabled": len(config["tools"]["built_in"]),
                "memory_enabled": config["memory"]["enabled"],
                "evaluation_enabled": config["evaluation"]["enabled"]
            }
        
        return {
            "status": "success",
            "data": status_data
        }
        
    except AgentNotFoundException:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    except Exception as e:
        logger.error(f"Failed to get agent status {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))