# Configurable LangGraph Agents

A flexible, configuration-driven system for creating AI agents using LangGraph's ReAct pattern, with **REST API integration**, memory management via LangMem, and **web-based interfaces** for complete agent lifecycle management.

## 🚀 Features

### Core Agent Features
- **🔧 YAML Configuration**: Complete agent configuration through YAML files
- **🤖 LLM Flexibility**: Support for OpenAI, Anthropic, Google Gemini, Groq, and other providers  
- **⚡ ReAct Pattern**: Uses LangGraph's optimized ReAct (Reasoning + Acting) agents
- **🧠 Memory Integration**: LangMem integration for semantic, episodic, and procedural memory
- **🎯 Tool Registry**: Built-in and custom tool management with dynamic registration
- **📄 Template System**: Reusable agent configurations and deployment templates

### 🌐 REST API & Frontend Integration
- **🚀 FastAPI Backend**: High-performance async REST API with auto-documentation
- **💻 Streamlit Web UI**: Interactive web interface with real-time agent management
- **📡 WebSocket Streaming**: Real-time agent execution with streaming responses
- **🔄 Template Management**: Save, share, and deploy agent configurations via API
- **📊 Evaluation System**: Automated testing and performance metrics via API endpoints
- **🛡️ Error Handling**: Comprehensive validation and structured error responses


## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
python3 setup.py
```

This will:
- Create a virtual environment
- Install all dependencies
- Set up the `.env` file
- Test the installation
- Start the REST API server (optional)

### Option 2: Manual Setup

1. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   ```

4. **Edit the `.env` file and add your API keys**:
   ```bash
   # OpenAI API Key
   OPENAI_API_KEY=your-openai-api-key-here

   # Anthropic API Key  
   ANTHROPIC_API_KEY=your-anthropic-api-key-here

   # Google Gemini API Key
   GOOGLE_API_KEY=your-google-api-key-here

   # Groq API Key
   GROQ_API_KEY=your-groq-api-key-here
   ```

5. **Start the REST API server (recommended for full functionality)**:
   ```bash
   python3 run_api.py
   # API will be available at: http://localhost:8000
   # Documentation at: http://localhost:8000/docs
   ```

6. **Launch the Web UI (in a new terminal)**:
   ```bash
   python3 run_web_ui.py
   # Web UI will be available at: http://localhost:8501
   ```

**Note**: The `.env` file is automatically loaded by the application. Make sure to add `.env` to your `.gitignore` file to keep your API keys secure.

### Quick Test

```bash
# Test single agent (no API required)
python3 -c "
from src.core.configurable_agent import ConfigurableAgent
agent = ConfigurableAgent('configs/examples/research_agent.yml')
print(agent.run('What is artificial intelligence?')['response'])
"

# Test REST API (requires API server running)
curl -X POST "http://localhost:8000/api/v1/agents" \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Agent", "config": {"llm": {"provider": "openai"}}}'
```

## 🎯 Basic Usage

### Single Agent

```python
from src.core.configurable_agent import ConfigurableAgent

# Create agent from configuration
agent = ConfigurableAgent("configs/examples/research_agent.yml")

# Run a query
response = agent.run("What are the latest developments in quantum computing?")
print(response["response"])
```

### Async Usage

```python
import asyncio

async def main():
    agent = ConfigurableAgent("configs/examples/research_agent.yml")
    response = await agent.arun("Tell me about machine learning")
    print(response["response"])

asyncio.run(main())
```


## 🖥️ Web Interfaces & REST API

### REST API Server

**Start the FastAPI server for full functionality:**

```bash
# Start the REST API server
python3 run_api.py
# Server runs at: http://localhost:8000
# API Documentation: http://localhost:8000/docs
# Alternative docs: http://localhost:8000/redoc

# Start with custom host/port
python3 run_api.py --host 0.0.0.0 --port 8080

# Check environment configuration
python3 run_api.py --check-env
```

**REST API Features:**
- 🚀 **FastAPI Framework**: High-performance async API with automatic documentation
- 📊 **Agent Management**: Create, read, update, delete agents via HTTP endpoints
- 🔄 **Real-time Execution**: Stream agent responses via WebSocket connections
- 📝 **Configuration Templates**: Save and reuse agent configurations
- 📈 **Evaluation Integration**: Run evaluations and view metrics via API
- 🛡️ **Error Handling**: Comprehensive error handling with structured responses
- 📋 **Auto Documentation**: Interactive Swagger UI and ReDoc documentation

### Web UI with API Integration

**Launch the Streamlit web interface (requires API server):**

```bash
# Method 1: Start API server first (recommended)
python3 run_api.py         # Terminal 1: Start REST API server
python3 run_web_ui.py      # Terminal 2: Start Web UI

# Method 2: Auto-start API server
python3 run_web_ui.py      # Will prompt to start API server automatically
```

**Enhanced Web UI Features:**
- 🌐 **API Integration**: Real-time connection status and health monitoring
- 🤖 **Agent Management**: Full CRUD operations via REST API backend
- 📊 **Live Validation**: Real-time configuration validation and error handling
- 💾 **Template System**: Save, load, and share agent configurations
- 🔄 **Stream Execution**: Real-time agent response streaming
- 📄 **API Documentation**: Direct links to API docs and endpoints


### Web UI Features

- **🎯 Multi-Tab Interface**: Organized tabs for agent info, LLM config, prompts, tools, memory, etc.
- **📝 Visual Configuration**: Form-based configuration with validation and help text
- **💾 File Operations**: Load example configs, upload/download YAML files
- **🔍 Live Preview**: Real-time YAML preview with validation
- **🧪 Agent Testing**: Test your configured agent directly in the interface
- **🚀 Quick Templates**: One-click loading of pre-configured agent templates
- **📊 Performance Dashboard**: Real-time metrics and analytics
- **🗂️ Agent Library**: Searchable catalog of available agent configurations

## 🌐 REST API Reference

### API Endpoints

#### Agent Management
```bash
# Create a new agent
POST /api/v1/agents
Content-Type: application/json
{
  "name": "My Agent",
  "config": { ... },
  "description": "Agent description"
}

# Get all agents
GET /api/v1/agents

# Get specific agent
GET /api/v1/agents/{agent_id}

# Update agent
PUT /api/v1/agents/{agent_id}

# Delete agent
DELETE /api/v1/agents/{agent_id}
```

#### Agent Execution
```bash
# Execute agent synchronously
POST /api/v1/agents/{agent_id}/execute
{
  "input": "Your query here",
  "stream": false
}

# Execute agent with streaming
POST /api/v1/agents/{agent_id}/execute
{
  "input": "Your query here", 
  "stream": true
}

# WebSocket streaming endpoint
WS /api/v1/agents/{agent_id}/stream
```

#### Configuration Templates
```bash
# Get all templates
GET /api/v1/templates

# Create template from agent
POST /api/v1/templates
{
  "name": "Template Name",
  "agent_id": "source_agent_id",
  "description": "Template description"
}

# Load template
GET /api/v1/templates/{template_id}
```

#### Evaluation System
```bash
# Run evaluation
POST /api/v1/evaluations
{
  "agent_id": "agent_id",
  "test_cases": [...],
  "metrics": ["accuracy", "response_time"]
}

# Get evaluation results
GET /api/v1/evaluations/{evaluation_id}

# Get evaluation metrics
GET /api/v1/evaluations/{evaluation_id}/metrics
```


### API Response Format

```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed successfully", 
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid-here"
}
```

### Error Handling

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid configuration provided",
    "details": { ... }
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "uuid-here"
}
```

### WebSocket Events

```javascript
// Connect to agent streaming
const ws = new WebSocket('ws://localhost:8000/api/v1/agents/agent_id/stream');

// Event types
ws.on('agent_started', (data) => { ... });
ws.on('agent_thinking', (data) => { ... });
ws.on('agent_response', (data) => { ... });
ws.on('agent_error', (data) => { ... });
ws.on('agent_finished', (data) => { ... });
```

### Python API Client

```python
import requests

# Initialize API client
api_base = "http://localhost:8000/api/v1"

# Create agent
response = requests.post(f"{api_base}/agents", json={
    "name": "Research Agent",
    "config": {
        "llm": {"provider": "openai", "model": "gpt-4"},
        "tools": {"built_in": ["web_search"]}
    }
})

agent_id = response.json()["data"]["id"]

# Execute agent
result = requests.post(f"{api_base}/agents/{agent_id}/execute", json={
    "input": "Research quantum computing developments",
    "stream": False
})

print(result.json()["data"]["response"])
```

## 📁 Configuration Structure

### Basic Agent Configuration

```yaml
# Agent Information
agent:
  name: "My Agent"
  description: "Agent description"
  version: "1.0.0"

# LLM Configuration
llm:
  provider: "openai"  # or "anthropic", "google", "groq"
  model: "gpt-4.1-mini"
  temperature: 0.7
  max_tokens: 2000
  api_key_env: "OPENAI_API_KEY"

# Prompts with Variables
prompts:
  system_prompt:
    template: "You are a helpful assistant. Context: {context}"
    variables: ["context"]
  user_prompt:
    template: "User query: {query}"
    variables: ["query"]

# ReAct Configuration
react:
  max_iterations: 10
  recursion_limit: 50
```

### Memory Configuration

```yaml
memory:
  enabled: true
  provider: "langmem"
  types:
    semantic: true    # Facts and knowledge
    episodic: true    # Conversation history
    procedural: true  # Learned patterns
  storage:
    backend: "memory"  # or "postgres", "redis"
  settings:
    max_memory_size: 5000
    retention_days: 30
    background_processing: true
```

### Tools Configuration

```yaml
tools:
  built_in:
    - "web_search"
    - "calculator"
    - "file_reader"
    - "file_writer"
    - "code_executor"
  custom:
    - name: "custom_tool"
      module_path: "my.custom.tools"
      class_name: "MyTool"
      description: "My custom tool"
      parameters:
        param1: "value1"
```


## 📚 Example Configurations

### Single Agents

#### Getting Started Templates

**Empty Template** - Start from scratch:
```bash
# Copy and customize the comprehensive template
cp configs/examples/template_agent.yml configs/examples/my_agent.yml
# Edit my_agent.yml with your specific configuration
agent = ConfigurableAgent("configs/examples/my_agent.yml")
```

**Minimal Template** - Simple setup:
```bash
# Copy and customize the minimal template
cp configs/examples/minimal_agent.yml configs/examples/my_simple_agent.yml  
# Edit my_simple_agent.yml with basic configuration
agent = ConfigurableAgent("configs/examples/my_simple_agent.yml")
```

#### Research Agent
```bash
python examples/usage_examples.py
```
Specialized for research tasks with web search, information synthesis, and memory.

#### Coding Assistant
```bash
agent = ConfigurableAgent("configs/examples/coding_assistant.yml")
response = agent.run("Write a Python function to calculate fibonacci numbers")
```
Optimized for software development with code execution and file operations.

#### Customer Support
```bash
agent = ConfigurableAgent("configs/examples/customer_support.yml")
response = agent.run("I need help with my order")
```
Designed for customer service with escalation and satisfaction tracking.

#### Gemini Research Agent
```bash
agent = ConfigurableAgent("configs/examples/gemini_agent.yml")
response = agent.run("What are the latest developments in quantum computing?")
```
Powered by Google's latest Gemini 2.5 Flash model for research and analysis tasks.

#### Groq Coding Assistant
```bash
agent = ConfigurableAgent("configs/examples/groq_agent.yml")
response = agent.run("Write a Python function to calculate fibonacci numbers")
```
Powered by Groq's fast LLMs (including Meta's Llama-4-Scout) for rapid code generation and development.


## 🛠️ Advanced Features

### Custom Tools

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """Custom tool description."""
    return f"Processed: {query}"

# Register with agent
agent.tool_registry.register_function_as_tool(
    name="my_tool",
    func=my_custom_tool,
    description="My custom tool"
)
```

### Dynamic Prompt Updates

```python
# Update prompts at runtime
agent.update_prompts({
    "system_prompt": "New system prompt template"
})
```

### Memory Management

```python
# Get memory statistics
stats = agent.get_memory_stats()
print(f"Stored facts: {stats['semantic_facts']}")

# Export conversation history
history = agent.export_conversation()

# Clear memory
agent.clear_memory()
```


### Optimization and Feedback

```python
from src.optimization.prompt_optimizer import OptimizationMetric

# Record feedback for optimization
agent.prompt_optimizer.record_feedback(
    prompt_type="system_prompt",
    variant_id="variant_1",
    metrics={
        OptimizationMetric.USER_SATISFACTION: 0.9,
        OptimizationMetric.ACCURACY: 0.85
    }
)

# Run optimization
optimized_prompts = agent.prompt_optimizer.optimize_prompts()
```


## 🎯 Use Cases

### 🔬 Research & Analysis
- **Academic Research**: Multi-step research with web search, analysis, and report generation
- **Market Research**: Comprehensive market analysis with data gathering and insights  
- **Competitive Intelligence**: Automated competitor analysis and reporting with API integration

### 💻 Software Development  
- **Full-stack Development**: Requirements analysis, development, and testing with agent workflows
- **Code Review**: Automated code analysis, review, and improvement suggestions
- **DevOps**: Deployment, monitoring, and maintenance workflows with API orchestration

### ✍️ Content Creation
- **Content Marketing**: Research, writing, editing, and SEO optimization pipelines
- **Technical Documentation**: API documentation, user guides, and tutorials with template management
- **Social Media**: Content creation, scheduling, and engagement analysis via REST API

### 🎧 Customer Support
- **Intelligent Support**: Automated support with intelligent question analysis and response
- **Help Desk**: Automated ticket handling and resolution with performance tracking
- **Customer Success**: Proactive customer engagement and support with analytics integration

## 📊 Performance Monitoring

### Key Metrics Tracked
- **Response Time**: Average time for task completion
- **Success Rate**: Percentage of successfully completed tasks
- **Agent Utilization**: Workload distribution across agents
- **Decision Accuracy**: Effectiveness of agent decisions
- **System Performance**: Overall system efficiency and reliability

### Dashboard Features
- Real-time charts and visualizations
- Agent performance comparison
- Agent analytics with performance metrics
- Alert system with configurable thresholds
- Historical trend analysis
- Detailed performance reports

## 🧪 Testing

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --unit      # Unit tests only
python run_tests.py --integration  # Integration tests only
python run_tests.py --lint      # Code quality checks
python run_tests.py --quick     # Quick smoke test

# With coverage report
python run_tests.py --unit --coverage

# For detailed output
python run_tests.py --all --verbose
```

## 🏗️ Architecture

```
src/
├── api/                          # REST API Layer (NEW)
│   ├── main.py                   # FastAPI application entry point
│   ├── middleware.py             # Custom middleware (CORS, logging, etc.)
│   ├── models/                   # Pydantic request/response models
│   │   ├── agent_models.py       # Agent-related API models
│   │   ├── evaluation_models.py  # Evaluation API models
│   ├── routes/                   # API route handlers
│   │   ├── agents.py             # Agent CRUD and execution endpoints
│   │   ├── evaluations.py        # Evaluation system endpoints
│   │   ├── templates.py          # Configuration template endpoints
│   │   └── analytics.py          # Performance analytics endpoints
│   └── utils/                    # API utilities
│       ├── config.py             # API configuration management
│       └── exceptions.py         # Custom API exceptions
├── core/
│   ├── configurable_agent.py    # Main agent class
│   ├── config_loader.py         # Configuration management
├── memory/
│   └── memory_manager.py        # LangMem integration
├── evaluation/                   # Evaluation System (NEW)
│   ├── evaluator.py             # Core evaluation engine
│   ├── metrics.py               # Evaluation metrics and scoring
│   └── test_runner.py           # Test execution and reporting
├── custom_logging/              # Enhanced Logging (NEW)
│   ├── __init__.py              # Logging configuration
│   ├── logger.py                # Custom logger implementation
│   └── formatters.py            # Log formatting utilities
├── optimization/
│   ├── prompt_optimizer.py      # Prompt optimization
│   └── feedback_collector.py    # Feedback collection
├── tools/
│   └── tool_registry.py         # Tool management
└── monitoring/
    └── performance_dashboard.py # Performance monitoring

# Entry Points
├── run_api.py                   # REST API server launcher (NEW)
├── run_web_ui.py               # Streamlit web UI with API integration
├── demo_web_ui.py              # Demo/testing interface (NEW)
└── api_summary.py              # API documentation generator (NEW)
```

### Component Interactions

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web UI        │    │   REST API      │    │   Core Agents   │
│  (Streamlit)    │◄──►│   (FastAPI)     │◄──►│   (LangGraph)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Templates     │    │   Evaluation    │    │   Memory        │
│   & Config      │    │   System        │    │   (LangMem)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚨 Troubleshooting

### Common Issues

#### Configuration Validation Errors
```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('your_config.yml'))"

# Validate with schema
python -c "
from src.core.config_loader import ConfigLoader
loader = ConfigLoader()
config = loader.load_config('your_config.yml')
print('Configuration is valid!')
"
```

#### API Key Issues
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Verify in Python
python -c "import os; print('OPENAI_API_KEY:', bool(os.getenv('OPENAI_API_KEY')))"
```

#### Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check specific imports
python -c "from src.core.configurable_agent import ConfigurableAgent"
```

### Performance Issues

#### Slow Agent Response
- Optimize LLM temperature settings for faster response
- Reduce max_tokens if responses are too long
- Consider using faster LLM providers like Groq

#### High Memory Usage
- Reduce `max_memory_size` in memory configuration
- Disable unused memory types
- Implement memory cleanup policies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

We welcome contributions to the agent system! Please feel free to:

- Submit bug reports and feature requests
- Contribute new agent templates
- Enhance the web UI
- Add new monitoring capabilities
- Improve evaluation systems

## 📄 License

MIT License - see LICENSE file for details.

## 📞 Support

For issues and questions:
- Check the [examples](examples/) directory
- Review the [test files](tests/) for usage patterns
- Create an issue on GitHub
- Check the inline code documentation

## 🔮 Roadmap

### Completed ✅
- [x] **REST API Integration**: Full FastAPI backend with comprehensive endpoints
- [x] **Web UI for Agent Configuration**: Interactive Streamlit interface with API integration
- [x] **Performance Monitoring**: Real-time dashboard with analytics and metrics
- [x] **Memory Management Integration**: LangMem support for semantic/episodic/procedural memory
- [x] **Evaluation System**: Automated testing and performance evaluation framework
- [x] **Template Management**: Save, load, and share agent configurations

### In Progress 🚧
- [ ] **React Frontend**: Modern React-based web interface to replace Streamlit UI
- [ ] **Enhanced Analytics**: Advanced performance metrics and ML-powered insights
- [ ] **Template Marketplace**: Community-driven agent configuration sharing platform
- [ ] **Auto-scaling**: Dynamic agent scaling based on workload and performance metrics

### Planned 📋
- [ ] **Additional LLM Providers**: Ollama, Cohere, and other open-source model support
- [ ] **Vector Database Integration**: Advanced memory backends with embedding search
- [ ] **Visual Agent Builder**: Drag-and-drop agent configuration interface
- [ ] **Plugin System**: Extensible architecture for custom components and integrations
- [ ] **Monitoring Dashboard**: Real-time system health and performance monitoring
- [ ] **Multi-tenant Support**: Enterprise-grade multi-organization agent management

---

**Built with ❤️ for the AI agent community**

Transform your AI workflows with **configuration-driven agents**, **REST API integration**, and **real-time web interfaces**. Start building powerful, scalable agent systems today! 

🚀 **[Get Started Now](#-quick-start)** | 📚 **[API Documentation](http://localhost:8000/docs)** | 🌐 **[Web Interface](http://localhost:8501)** | 🤝 **[Contributing](#-contributing)**