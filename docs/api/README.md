# Configurable Agents API Documentation

## Overview

The Configurable Agents API is a comprehensive RESTful API for managing and deploying configurable AI agents. Built with FastAPI, it provides full lifecycle management for AI agents with support for multiple LLM providers, real-time execution, memory systems, and evaluation frameworks.

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp env.example .env
# Edit .env with your API keys

# Run the API server
python run_api.py

# Access documentation
open http://localhost:8000/docs
```

### Production Deployment

See [GCP Cloud Run Deployment Guide](../deployment/gcp-cloud-run.md) for detailed production deployment instructions.

## API Features

### 🤖 Agent Management
- **Create**: Configure agents with custom LLMs, prompts, and tools
- **Read**: Retrieve agent details and configurations
- **Update**: Modify existing agent configurations
- **Delete**: Remove agents and clean up resources
- **Status**: Monitor agent health and usage metrics

### 🔧 Multi-Provider Support
- **OpenAI**: GPT-3.5, GPT-4, and other OpenAI models
- **Anthropic**: Claude models with constitutional AI
- **Google**: Gemini and PaLM models
- **Groq**: High-speed inference with Llama and other models

### ⚡ Execution Modes
- **Synchronous**: Standard request-response pattern
- **WebSocket Streaming**: Real-time execution with live updates
- **Batch Processing**: Handle multiple queries efficiently

### 🧠 Memory Systems
- **Semantic Memory**: Store and retrieve knowledge
- **Episodic Memory**: Remember conversation history
- **Procedural Memory**: Learn and apply procedures

### 🛠️ Tool Integration
- **Built-in Tools**: Web search, calculator, file operations, code execution
- **Custom Tools**: Define and register your own tools
- **Tool Chaining**: Combine multiple tools for complex workflows

### 📊 Evaluation Framework
- **Automated Testing**: Run evaluations against test datasets
- **Performance Metrics**: Track correctness, helpfulness, response time
- **A/B Testing**: Compare different agent configurations

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information and status |
| `GET` | `/docs` | Interactive API documentation |
| `GET` | `/redoc` | Alternative API documentation |
| `GET` | `/api/health` | Health check endpoint |

### Agent Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/agents` | Create new agent |
| `GET` | `/api/agents` | List all agents (paginated) |
| `GET` | `/api/agents/{id}` | Get specific agent |
| `PUT` | `/api/agents/{id}` | Update agent configuration |
| `DELETE` | `/api/agents/{id}` | Delete agent |
| `GET` | `/api/agents/{id}/status` | Get agent status |

### Agent Execution

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/agents/{id}/run` | Execute agent with query |
| `WebSocket` | `/api/agents/{id}/stream` | Stream agent execution |

### Future Endpoints (Planned)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/evaluation/run` | Run agent evaluation |
| `GET` | `/api/templates` | List configuration templates |
| `POST` | `/api/templates` | Create configuration template |
| `GET` | `/api/analytics/usage` | Usage analytics and metrics |
| `POST` | `/api/files/upload` | Upload files for agents |

## Authentication

The API supports multiple authentication methods:

### Bearer Token (JWT)
```bash
curl -H "Authorization: Bearer <jwt-token>" \
  https://your-api-domain.com/api/agents
```

### API Key
```bash
curl -H "X-API-Key: <your-api-key>" \
  https://your-api-domain.com/api/agents
```

### Development Mode
For local development, authentication can be disabled by setting `DISABLE_AUTH=true` in your environment.

## Request/Response Examples

### Create Agent

**Request:**
```bash
curl -X POST "http://localhost:8000/api/agents" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research Assistant",
    "description": "AI agent specialized in research and information gathering",
    "version": "1.0.0",
    "llm": {
      "provider": "openai",
      "model": "gpt-4",
      "temperature": 0.7,
      "max_tokens": 2000
    },
    "prompts": {
      "system_prompt": "You are a helpful research assistant that provides accurate and well-sourced information.",
      "variables": {}
    },
    "tools": ["web_search", "calculator"],
    "memory": [
      {
        "type": "semantic",
        "enabled": true
      }
    ],
    "evaluation": {
      "enabled": true,
      "evaluators": [
        {
          "name": "correctness",
          "type": "llm_as_judge",
          "parameters": {}
        }
      ],
      "auto_evaluate": false
    },
    "react": {
      "max_iterations": 10,
      "recursion_limit": 25
    },
    "debug_mode": false,
    "tags": ["research", "assistant"]
  }'
```

**Response:**
```json
{
  "agent": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Research Assistant",
    "description": "AI agent specialized in research and information gathering",
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
      "temperature": 0.7,
      "max_tokens": 2000
    },
    "tools": ["web_search", "calculator"],
    "debug_mode": false
  }
}
```

### Run Agent

**Request:**
```bash
curl -X POST "http://localhost:8000/api/agents/550e8400-e29b-41d4-a716-446655440000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the latest developments in artificial intelligence for 2024?",
    "context": {},
    "stream": false,
    "timeout": 300,
    "include_evaluation": false
  }'
```

**Response:**
```json
{
  "run_id": "run_550e8400-e29b-41d4-a716-446655440000",
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "What are the latest developments in artificial intelligence for 2024?",
  "result": {
    "response": "Based on recent research and industry reports, here are the key AI developments in 2024...",
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
        "result": "Search results containing recent AI advances..."
      }
    ],
    "memory_updates": [],
    "debug_info": null
  },
  "started_at": "2024-01-01T12:00:00Z",
  "completed_at": "2024-01-01T12:00:02Z"
}
```

### WebSocket Streaming

Connect to WebSocket endpoint:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/agents/550e8400-e29b-41d4-a716-446655440000/stream');

// Send query
ws.send(JSON.stringify({
  type: 'run',
  query: 'Explain quantum computing in simple terms'
}));

// Receive responses
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

### Error Response Format
```json
{
  "error": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": "Additional error details (if available)"
}
```

### Common Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `201` | Created successfully |
| `400` | Bad request (invalid parameters) |
| `401` | Unauthorized |
| `404` | Resource not found |
| `422` | Validation error |
| `429` | Rate limit exceeded |
| `500` | Internal server error |

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default Limit**: 100 requests per minute per IP
- **Burst Limit**: 10 requests per second
- **Headers**: Rate limit status included in response headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609459200
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8000` |
| `HOST` | Server host | `127.0.0.1` |
| `ENVIRONMENT` | Environment mode | `development` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required |
| `GOOGLE_API_KEY` | Google API key | Required |
| `GROQ_API_KEY` | Groq API key | Required |

### CORS Configuration

Configure allowed origins for web frontend integration:

```bash
export CORS_ORIGINS="https://your-frontend.com,https://admin.your-domain.com"
```

## Monitoring and Observability

### Health Checks

The API provides comprehensive health checks for container orchestration:

```bash
curl http://localhost:8000/api/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "environment": "production",
  "dependencies": {
    "database": "healthy",
    "redis": "healthy"
  }
}
```

### Structured Logging

All requests and operations are logged in structured format for observability:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "message": "Agent executed successfully",
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "execution_time": 2.5,
  "tokens_used": 450
}
```

### Metrics

Key metrics exposed for monitoring:

- Request duration and throughput
- Agent execution times
- Token usage per provider
- Error rates and types
- Memory usage and performance

## SDK and Client Libraries

### Python Client

```python
from configurable_agents import AgentClient

client = AgentClient(
    base_url="https://your-api-domain.com",
    api_key="your-api-key"
)

# Create agent
agent = client.create_agent({
    "name": "Research Assistant",
    "llm": {"provider": "openai", "model": "gpt-4"},
    "tools": ["web_search"]
})

# Run agent
result = client.run_agent(agent.id, "What is quantum computing?")
print(result.response)
```

### JavaScript/TypeScript Client

```typescript
import { AgentClient } from '@configurable-agents/client';

const client = new AgentClient({
  baseUrl: 'https://your-api-domain.com',
  apiKey: 'your-api-key'
});

// Create and run agent
const agent = await client.createAgent({
  name: 'Research Assistant',
  llm: { provider: 'openai', model: 'gpt-4' },
  tools: ['web_search']
});

const result = await client.runAgent(agent.id, 'What is quantum computing?');
console.log(result.response);
```

## Support and Community

- **Documentation**: [https://docs.configurable-agents.com](https://docs.configurable-agents.com)
- **GitHub Issues**: [Report bugs and request features](https://github.com/your-org/configurable-agents/issues)
- **Discord Community**: [Join our Discord](https://discord.gg/configurable-agents)
- **Email Support**: support@configurable-agents.com

## License

This project is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.