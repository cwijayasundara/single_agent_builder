"""
Agent management API endpoints.
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import asyncio

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
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


def create_agent_instance(agent_id: str, config_dict: Dict[str, Any]) -> ConfigurableAgent:
    """Create agent instance from configuration."""
    try:
        # Convert request to AgentConfiguration
        agent_config = AgentConfiguration(**config_dict)
        
        # Create temporary YAML file for ConfigurableAgent
        import tempfile
        import yaml
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_config_path = f.name
        
        try:
            agent = ConfigurableAgent(temp_config_path)
            agent_instances[agent_id] = agent
            return agent
        finally:
            os.unlink(temp_config_path)
            
    except Exception as e:
        logger.error(f"Failed to create agent instance: {e}")
        raise AgentConfigurationError(f"Failed to create agent: {str(e)}")


@router.post("/", response_model=AgentResponse)
async def create_agent(request: AgentCreateRequest):
    """Create a new agent."""
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
                "api_key_env": f"{request.llm.provider.value.upper()}_API_KEY"
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
        except Exception as e:
            agent_data["status"] = AgentStatus.ERROR.value
            logger.error(f"Failed to initialize agent {agent_id}: {e}")
        
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


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[AgentStatus] = None,
    tags: Optional[str] = Query(None, description="Comma-separated tags")
):
    """List all agents."""
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


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    """Get a specific agent."""
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


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str, request: AgentUpdateRequest):
    """Update an existing agent."""
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
                "top_p": request.llm.top_p
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
        except Exception as e:
            agent_data["status"] = AgentStatus.ERROR.value
            logger.error(f"Failed to update agent instance {agent_id}: {e}")
        
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


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent."""
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


@router.post("/{agent_id}/run", response_model=AgentRunResponse)
async def run_agent(agent_id: str, request: AgentRunRequest):
    """Run an agent with a query."""
    try:
        agent_data = get_agent_or_404(agent_id)
        
        if agent_data["status"] != AgentStatus.ACTIVE.value:
            raise HTTPException(
                status_code=400, 
                detail=f"Agent {agent_id} is not active (status: {agent_data['status']})"
            )
        
        # Get agent instance
        if agent_id not in agent_instances:
            create_agent_instance(agent_id, agent_data["config"])
        
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
async def stream_agent(websocket: WebSocket, agent_id: str):
    """Stream agent execution via WebSocket."""
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


@router.get("/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get agent status and basic metrics."""
    try:
        agent_data = get_agent_or_404(agent_id)
        
        return {
            "status": "success",
            "data": {
                "agent_id": agent_id,
                "status": agent_data["status"],
                "run_count": agent_data.get("run_count", 0),
                "created_at": agent_data["created_at"].isoformat(),
                "updated_at": agent_data["updated_at"].isoformat(),
                "instance_loaded": agent_id in agent_instances
            }
        }
        
    except AgentNotFoundException:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    except Exception as e:
        logger.error(f"Failed to get agent status {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))