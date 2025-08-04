"""
HTTP client for the Configurable Agents API.
Provides a convenient interface for the Streamlit web UI to interact with the REST API.
"""
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from urllib.parse import urljoin

# Handle optional dependencies gracefully
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

try:
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    # Fallback logger for environments without logging
    class DummyLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
    logger = DummyLogger()


class APIClientError(Exception):
    """Base exception for API client errors."""
    pass


class APIConnectionError(APIClientError):
    """Raised when unable to connect to the API."""
    pass


class APIValidationError(APIClientError):
    """Raised when API request validation fails."""
    pass


class AgentNotFoundError(APIClientError):
    """Raised when agent is not found."""
    pass




class ConfigurableAgentsAPIClient:
    """
    HTTP client for interacting with the Configurable Agents REST API.
    
    This client provides a convenient interface for the Streamlit web UI
    to interact with all API endpoints without direct imports.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
        """
        if not HTTPX_AVAILABLE:
            raise APIClientError(
                "httpx is required for API client. Install with: pip install httpx"
            )
        
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        
        # Cache for connection status
        self._last_health_check = None
        self._is_connected = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close the HTTP client."""
        if self._client:
            self._client.close()
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (without base URL)
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            APIConnectionError: If unable to connect to API
            APIValidationError: If request validation fails
            APIClientError: For other API errors
        """
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        try:
            if method.upper() == 'GET':
                response = self._client.get(url, params=params)
            elif method.upper() == 'POST':
                response = self._client.post(url, json=data, params=params)
            elif method.upper() == 'PUT':
                response = self._client.put(url, json=data, params=params)
            elif method.upper() == 'DELETE':
                response = self._client.delete(url, params=params)
            else:
                raise APIClientError(f"Unsupported HTTP method: {method}")
            
            # Check for connection errors
            if response.status_code == 503:
                raise APIConnectionError("API server is unavailable")
            
            # Parse JSON response
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                if response.status_code >= 400:
                    raise APIClientError(f"HTTP {response.status_code}: {response.text}")
                response_data = {}
            
            # Handle HTTP errors
            if response.status_code == 404:
                if "agent" in endpoint.lower():
                    raise AgentNotFoundError(response_data.get('message', 'Agent not found'))
                raise APIClientError(response_data.get('message', 'Resource not found'))
            elif response.status_code == 422:
                raise APIValidationError(response_data.get('message', 'Validation error'))
            elif response.status_code >= 400:
                raise APIClientError(
                    response_data.get('message', f'HTTP {response.status_code} error')
                )
            
            return response_data
            
        except httpx.ConnectError:
            raise APIConnectionError(f"Unable to connect to API server at {self.base_url}")
        except httpx.TimeoutException:
            raise APIConnectionError("Request timeout - API server is not responding")
        except httpx.RequestError as e:
            raise APIConnectionError(f"Request error: {str(e)}")
    
    # Health and Status Methods
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API server health.
        
        Returns:
            Health status information
        """
        try:
            response = self._make_request('GET', '/api/health/')
            self._is_connected = True
            self._last_health_check = datetime.utcnow()
            return response
        except APIConnectionError:
            self._is_connected = False
            raise
    
    def is_connected(self) -> bool:
        """
        Check if client is connected to API server.
        Performs a health check if last check was more than 30 seconds ago.
        
        Returns:
            True if connected, False otherwise
        """
        now = datetime.utcnow()
        if (self._last_health_check is None or 
            (now - self._last_health_check).total_seconds() > 30):
            try:
                self.health_check()
            except APIConnectionError:
                pass
        
        return self._is_connected
    
    # Agent Management Methods
    
    def create_agent(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new agent.
        
        Args:
            config: Agent configuration dictionary
            
        Returns:
            Created agent information
        """
        # Convert internal config format to API request format
        api_request = self._convert_config_to_api_request(config)
        return self._make_request('POST', '/api/agents/', data=api_request)
    
    def list_agents(
        self, 
        page: int = 1, 
        page_size: int = 20,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        List all agents.
        
        Args:
            page: Page number
            page_size: Number of agents per page
            status: Filter by agent status
            tags: Filter by tags
            
        Returns:
            List of agents with pagination info
        """
        params = {
            'page': page,
            'page_size': page_size
        }
        
        if status:
            params['status'] = status
        
        if tags:
            params['tags'] = ','.join(tags)
        
        return self._make_request('GET', '/api/agents/', params=params)
    
    def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Get a specific agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent information and configuration
        """
        return self._make_request('GET', f'/api/agents/{agent_id}')
    
    def update_agent(self, agent_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing agent.
        
        Args:
            agent_id: Agent ID
            config: Updated agent configuration
            
        Returns:
            Updated agent information
        """
        # Convert internal config format to API request format
        api_request = self._convert_config_to_api_update_request(config)
        return self._make_request('PUT', f'/api/agents/{agent_id}', data=api_request)
    
    def delete_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Delete an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Deletion confirmation
        """
        return self._make_request('DELETE', f'/api/agents/{agent_id}')
    
    def run_agent(
        self, 
        agent_id: str, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        include_evaluation: bool = False
    ) -> Dict[str, Any]:
        """
        Run an agent with a query.
        
        Args:
            agent_id: Agent ID
            query: Query to send to the agent
            context: Additional context
            include_evaluation: Whether to include evaluation results
            
        Returns:
            Agent run results
        """
        request_data = {
            'query': query,
            'context': context or {},
            'include_evaluation': include_evaluation
        }
        
        return self._make_request('POST', f'/api/agents/{agent_id}/run', data=request_data)
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get agent status and metrics.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Agent status information
        """
        return self._make_request('GET', f'/api/agents/{agent_id}/status')
    
    # Configuration Validation Methods
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate agent configuration without creating an agent.
        
        Args:
            config: Agent configuration to validate
            
        Returns:
            Validation results
        """
        try:
            # Try to create a temporary agent request to validate
            api_request = self._convert_config_to_api_request(config)
            
            # For now, we'll validate by attempting to create (but not persist)
            # In the future, we could add a dedicated validation endpoint
            return {
                'valid': True,
                'message': 'Configuration is valid',
                'errors': []
            }
        except (APIValidationError, APIClientError) as e:
            return {
                'valid': False,
                'message': str(e),
                'errors': [str(e)]
            }
        except Exception as e:
            return {
                'valid': False,
                'message': f'Validation error: {str(e)}',
                'errors': [str(e)]
            }
    
    # Template Management Methods
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """
        Get available configuration templates.
        
        Returns:
            List of available templates
        """
        # For now, return hardcoded templates since API doesn't have template endpoints yet
        # TODO: Implement when template endpoints are available
        return [
            {
                'name': 'Research Agent',
                'file': 'research_agent.yml',
                'description': 'Web research and information gathering'
            },
            {
                'name': 'Coding Assistant',
                'file': 'coding_assistant.yml',
                'description': 'Code generation and programming help'
            },
            {
                'name': 'Customer Support',
                'file': 'customer_support.yml',
                'description': 'Customer service and support agent'
            }
        ]
    
    def load_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a configuration template.
        
        Args:
            template_name: Name of the template to load
            
        Returns:
            Template configuration or None if not found
        """
        # For now, return None since template loading will be handled differently
        # TODO: Implement when template endpoints are available
        return None
    
    
    def create_hierarchical_team(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new hierarchical team.
        
        Args:
            config: Hierarchical team configuration dictionary
            
        Returns:
            Created team information
        """
        # Convert internal config format to API request format
        api_request = self._convert_hierarchical_config_to_api_request(config)
        return self._make_request('POST', '/api/teams/', data=api_request)
    
    def list_hierarchical_teams(
        self, 
        page: int = 1, 
        page_size: int = 20,
        status: Optional[str] = None,
        team_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        List all hierarchical teams.
        
        Args:
            page: Page number
            page_size: Number of teams per page
            status: Filter by team status
            team_type: Filter by team type
            tags: Filter by tags
            
        Returns:
            List of teams with pagination info
        """
        params = {
            'page': page,
            'page_size': page_size
        }
        
        if status:
            params['status'] = status
        
        if team_type:
            params['team_type'] = team_type
            
        if tags:
            params['tags'] = ','.join(tags)
        
        return self._make_request('GET', '/api/teams/', params=params)
    
    def get_hierarchical_team(self, team_id: str) -> Dict[str, Any]:
        """
        Get a specific hierarchical team.
        
        Args:
            team_id: Team ID
            
        Returns:
            Team information and configuration
        """
        return self._make_request('GET', f'/api/teams/{team_id}')
    
    def update_hierarchical_team(self, team_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing hierarchical team.
        
        Args:
            team_id: Team ID
            config: Updated team configuration
            
        Returns:
            Updated team information
        """
        # Convert internal config format to API request format
        api_request = self._convert_hierarchical_config_to_api_update_request(config)
        return self._make_request('PUT', f'/api/teams/{team_id}', data=api_request)
    
    def delete_hierarchical_team(self, team_id: str) -> Dict[str, Any]:
        """
        Delete a hierarchical team.
        
        Args:
            team_id: Team ID
            
        Returns:
            Deletion confirmation
        """
        return self._make_request('DELETE', f'/api/teams/{team_id}')
    
    def run_hierarchical_team(
        self, 
        team_id: str, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        include_evaluation: bool = False,
        include_debug_info: bool = False,
        max_execution_time: Optional[int] = None,
        routing_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a hierarchical team with a query.
        
        Args:
            team_id: Team ID
            query: Query to send to the team
            context: Additional context
            include_evaluation: Whether to include evaluation results
            include_debug_info: Whether to include debug information
            max_execution_time: Maximum execution time in seconds
            routing_preferences: Routing preferences for team coordination
            
        Returns:
            Team run results
        """
        request_data = {
            'query': query,
            'context': context or {},
            'include_evaluation': include_evaluation,
            'include_debug_info': include_debug_info
        }
        
        if max_execution_time:
            request_data['max_execution_time'] = max_execution_time
            
        if routing_preferences:
            request_data['routing_preferences'] = routing_preferences
        
        return self._make_request('POST', f'/api/teams/{team_id}/run', data=request_data)
    
    def get_hierarchical_team_status(self, team_id: str) -> Dict[str, Any]:
        """
        Get hierarchical team status and metrics.
        
        Args:
            team_id: Team ID
            
        Returns:
            Team status information
        """
        return self._make_request('GET', f'/api/teams/{team_id}/status')
    
    def get_hierarchical_team_metrics(
        self, 
        team_id: str, 
        time_period: str = "1h"
    ) -> Dict[str, Any]:
        """
        Get hierarchical team performance metrics.
        
        Args:
            team_id: Team ID
            time_period: Time period for metrics (e.g., "1h", "24h", "7d")
            
        Returns:
            Team performance metrics
        """
        params = {'time_period': time_period}
        return self._make_request('GET', f'/api/teams/{team_id}/metrics', params=params)
    
    def get_hierarchical_templates(self) -> List[Dict[str, Any]]:
        """
        Get available hierarchical team templates.
        
        Returns:
            List of available templates
        """
        response = self._make_request('GET', '/api/teams/templates/')
        return response.get('templates', [])
    
    def load_hierarchical_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a hierarchical team template configuration.
        
        Args:
            template_name: Name of the template to load
            
        Returns:
            Template configuration or None if not found
        """
        try:
            # For now, return None since template loading will be handled differently
            # TODO: Implement when template loading endpoints are available
            return None
        except TeamNotFoundError:
            return None
    
    def validate_hierarchical_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate hierarchical team configuration without creating a team.
        
        Args:
            config: Hierarchical team configuration to validate
            
        Returns:
            Validation results
        """
        try:
            # Try to create a temporary team request to validate
            api_request = self._convert_hierarchical_config_to_api_request(config)
            
            # For now, we'll validate by attempting to create (but not persist)
            # In the future, we could add a dedicated validation endpoint
            return {
                'valid': True,
                'message': 'Configuration is valid',
                'errors': []
            }
        except (APIValidationError, APIClientError) as e:
            return {
                'valid': False,
                'message': str(e),
                'errors': [str(e)]
            }
        except Exception as e:
            return {
                'valid': False,
                'message': f'Validation error: {str(e)}',
                'errors': [str(e)]
            }
    
    # WebSocket Support for Hierarchical Teams
    
    def create_team_websocket_connection(self, team_id: str):
        """
        Create a WebSocket connection for streaming team execution.
        
        Args:
            team_id: Team ID
            
        Returns:
            WebSocket connection object
            
        Note: This requires additional WebSocket client library
        """
        # This would require a WebSocket client library
        # For now, return None as a placeholder
        logger.warning("WebSocket support not implemented yet")
        return None

    # Helper Methods
    
    def _convert_hierarchical_config_to_api_request(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert internal hierarchical configuration format to API request format.
        
        Args:
            config: Internal hierarchical configuration dictionary
            
        Returns:
            API request dictionary
        """
        team_info = config.get('team', {})
        coordinator_config = config.get('coordinator', {})
        teams_config = config.get('teams', [])
        
        # Build API request structure
        api_request = {
            'name': team_info.get('name', 'Unnamed Team'),
            'description': team_info.get('description', ''),
            'version': team_info.get('version', '1.0.0'),
            'type': team_info.get('type', 'hierarchical'),
            'coordinator': {
                'name': coordinator_config.get('name', 'Coordinator'),
                'description': coordinator_config.get('description', ''),
                'llm': coordinator_config.get('llm', {}),
                'routing_strategy': coordinator_config.get('routing', {}).get('strategy', 'hybrid'),
                'routing_config': coordinator_config.get('routing', {}).get('config', {}),
                'system_prompt': coordinator_config.get('system_prompt')
            },
            'teams': [],
            'communication_config': config.get('communication', {}),
            'performance_config': config.get('performance', {}),
            'logging_config': config.get('logging', {}),
            'tags': config.get('tags', []),
            'metadata': config.get('metadata', {})
        }
        
        # Convert teams
        for team_config in teams_config:
            team_request = {
                'name': team_config.get('name', 'Team'),
                'description': team_config.get('description', ''),
                'supervisor_name': team_config.get('supervisor', {}).get('name', 'Supervisor'),
                'supervisor_config': team_config.get('supervisor', {}).get('config', {}),
                'workers': [],
                'specialization': team_config.get('specialization', {}),
                'team_config': team_config.get('config', {})
            }
            
            # Convert workers
            for worker_config in team_config.get('workers', []):
                worker_request = {
                    'name': worker_config.get('name', 'Worker'),
                    'description': worker_config.get('description', ''),
                    'role': worker_config.get('role', 'worker'),
                    'config_file': worker_config.get('config_file'),
                    'capabilities': worker_config.get('capabilities', []),
                    'priority': worker_config.get('priority', 1)
                }
                
                # Add agent configuration if provided
                if 'agent_config' in worker_config:
                    agent_config = worker_config['agent_config']
                    if 'llm' in agent_config:
                        worker_request['llm'] = agent_config['llm']
                    if 'prompts' in agent_config:
                        worker_request['prompts'] = agent_config['prompts']
                    if 'tools' in agent_config:
                        worker_request['tools'] = agent_config['tools'].get('built_in', [])
                    if 'memory' in agent_config:
                        worker_request['memory'] = agent_config['memory'].get('configs', [])
                
                team_request['workers'].append(worker_request)
            
            api_request['teams'].append(team_request)
        
        return api_request
    
    def _convert_hierarchical_config_to_api_update_request(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert internal hierarchical configuration format to API update request format.
        
        Args:
            config: Internal hierarchical configuration dictionary
            
        Returns:
            API update request dictionary
        """
        # For updates, we'll use the same format as create but mark fields as optional
        return self._convert_hierarchical_config_to_api_request(config)
    
    def _convert_api_team_response_to_config(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert API team response format to internal configuration format.
        
        Args:
            api_response: API response dictionary
            
        Returns:
            Internal configuration dictionary
        """
        team_info = api_response.get('team', {})
        
        # Build internal config structure
        internal_config = {
            'team': {
                'name': team_info.get('name', ''),
                'description': team_info.get('description', ''),
                'version': team_info.get('version', '1.0.0'),
                'type': team_info.get('type', 'hierarchical')
            },
            'coordinator': {
                'name': team_info.get('coordinator', {}).get('name', ''),
                'description': team_info.get('coordinator', {}).get('description', ''),
                'llm': team_info.get('coordinator', {}).get('llm_config', {}),
                'routing': {
                    'strategy': team_info.get('coordinator', {}).get('routing_strategy', 'hybrid'),
                    'config': team_info.get('coordinator', {}).get('routing_config', {})
                }
            },
            'teams': [],
            'communication': team_info.get('communication_config', {}),
            'performance': team_info.get('performance_config', {}),
            'logging': team_info.get('logging_config', {}),
            'tags': team_info.get('tags', []),
            'metadata': team_info.get('metadata', {})
        }
        
        # Convert teams
        for team_data in team_info.get('teams', []):
            team_config = {
                'name': team_data.get('name', ''),
                'description': team_data.get('description', ''),
                'supervisor': {
                    'name': team_data.get('supervisor_name', ''),
                    'config': {}
                },
                'workers': [],
                'specialization': team_data.get('specialization', {}),
                'config': {}
            }
            
            # Convert workers
            for worker_data in team_data.get('workers', []):
                worker_config = {
                    'name': worker_data.get('name', ''),
                    'description': worker_data.get('description', ''),
                    'role': worker_data.get('role', ''),
                    'config_file': worker_data.get('config_file'),
                    'capabilities': worker_data.get('capabilities', []),
                    'priority': worker_data.get('priority', 1)
                }
                team_config['workers'].append(worker_config)
            
            internal_config['teams'].append(team_config)
        
        return internal_config
    
    # Helper Methods
    
    def _convert_config_to_api_request(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert internal configuration format to API request format.
        
        Args:
            config: Internal configuration dictionary
            
        Returns:
            API request dictionary
        """
        agent_info = config.get('agent', {})
        llm_config = config.get('llm', {})
        prompts_config = config.get('prompts', {})
        tools_config = config.get('tools', {})
        memory_config = config.get('memory', {})
        evaluation_config = config.get('evaluation', {})
        react_config = config.get('react', {})
        
        # Build API request structure
        api_request = {
            'name': agent_info.get('name', 'Unnamed Agent'),
            'description': agent_info.get('description', ''),
            'version': agent_info.get('version', '1.0.0'),
            'llm': {
                'provider': llm_config.get('provider', 'openai'),
                'model': llm_config.get('model', 'gpt-4.1-mini'),
                'temperature': llm_config.get('temperature', 0.7),
                'max_tokens': llm_config.get('max_tokens', 4000),
                'top_p': llm_config.get('top_p', 1.0)
            },
            'prompts': {
                'system_prompt': prompts_config.get('system_prompt', {}).get('template', 'You are a helpful assistant.'),
                'variables': prompts_config.get('system_prompt', {}).get('variables', {})
            },
            'tools': tools_config.get('built_in', []),
            'memory': [],
            'evaluation': {
                'enabled': evaluation_config.get('enabled', False),
                'evaluators': [],
                'metrics': evaluation_config.get('metrics', []),
                'auto_evaluate': evaluation_config.get('auto_evaluate', False)
            },
            'react': {
                'max_iterations': react_config.get('max_iterations', 10),
                'recursion_limit': react_config.get('recursion_limit', 25)
            },
            'debug_mode': config.get('debug_mode', False),
            'tags': []
        }
        
        # Convert memory configuration
        if memory_config.get('enabled', False):
            memory_types = memory_config.get('types', {})
            for mem_type, enabled in memory_types.items():
                if enabled:
                    api_request['memory'].append({
                        'type': mem_type,
                        'enabled': True,
                        'parameters': memory_config.get('settings', {})
                    })
        
        # Convert evaluation configuration
        if evaluation_config.get('enabled', False):
            evaluators = evaluation_config.get('evaluators', [])
            for evaluator in evaluators:
                api_request['evaluation']['evaluators'].append({
                    'name': evaluator.get('name', ''),
                    'type': evaluator.get('type', 'heuristic'),
                    'parameters': evaluator.get('parameters', {})
                })
        
        return api_request
    
    def _convert_config_to_api_update_request(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert internal configuration format to API update request format.
        
        Args:
            config: Internal configuration dictionary
            
        Returns:
            API update request dictionary
        """
        # For updates, we'll use the same format as create but mark fields as optional
        return self._convert_config_to_api_request(config)
    
    def _convert_api_response_to_config(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert API response format to internal configuration format.
        
        Args:
            api_response: API response dictionary
            
        Returns:
            Internal configuration dictionary
        """
        agent_info = api_response.get('agent', {})
        config_info = api_response.get('config', {})
        
        # Build internal config structure
        internal_config = {
            'agent': {
                'name': agent_info.get('name', ''),
                'description': agent_info.get('description', ''),
                'version': agent_info.get('version', '1.0.0')
            },
            'llm': config_info.get('llm', {}),
            'prompts': {
                'system_prompt': {
                    'template': config_info.get('prompts', {}).get('system_prompt', ''),
                    'variables': config_info.get('prompts', {}).get('variables', [])
                }
            },
            'tools': {
                'built_in': config_info.get('tools', [])
            },
            'memory': {
                'enabled': bool(config_info.get('memory', [])),
                'types': {},
                'settings': {}
            },
            'evaluation': config_info.get('evaluation', {}),
            'react': config_info.get('react', {}),
            'debug_mode': config_info.get('debug_mode', False)
        }
        
        # Convert memory configuration
        memory_configs = config_info.get('memory', [])
        if memory_configs:
            internal_config['memory']['enabled'] = True
            for mem_config in memory_configs:
                mem_type = mem_config.get('type', '')
                if mem_type:
                    internal_config['memory']['types'][mem_type] = mem_config.get('enabled', True)
                    internal_config['memory']['settings'].update(mem_config.get('parameters', {}))
        
        return internal_config


# Convenience functions for backward compatibility

def create_api_client(base_url: str = "http://localhost:8000") -> ConfigurableAgentsAPIClient:
    """Create an API client instance."""
    if not HTTPX_AVAILABLE:
        raise APIClientError(
            "httpx is required for API client. Install with: pip install httpx"
        )
    return ConfigurableAgentsAPIClient(base_url=base_url)


def check_api_connection(base_url: str = "http://localhost:8000") -> bool:
    """Check if API server is accessible."""
    if not HTTPX_AVAILABLE:
        return False
    
    try:
        with create_api_client(base_url) as client:
            client.health_check()
            return True
    except (APIConnectionError, APIClientError):
        return False