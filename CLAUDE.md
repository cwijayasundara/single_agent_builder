# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation & Setup
```bash
# Automated setup (recommended)
python setup.py

# Or manual setup
pip install -r requirements.txt
cp env.example .env
# Edit .env file and add your API keys

# Verify setup
python -c "from src.core.configurable_agent import ConfigurableAgent; print('✅ Setup successful')"
```

Required environment variables:
```bash
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
GROQ_API_KEY=your-groq-api-key-here
```

### Testing
```bash
# Note: run_tests.py has been removed from current codebase
# Use pytest directly for testing

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_config_loader.py -v

# Run single test method
pytest tests/test_config_loader.py::TestConfigLoader::test_load_valid_config -v

# Quick smoke test (basic functionality)
python examples/usage_examples.py
```

### Running Examples
```bash
# Run usage examples (requires API keys)
python examples/usage_examples.py

# Launch web UI for configuration
python run_web_ui.py


# Launch REST API server (new)
python run_api.py

# Test REST API endpoints (new)
python test_api.py

# Validate API configuration
python validate_api.py

# Test specific agent configuration
python -c "
from src.core.configurable_agent import ConfigurableAgent
agent = ConfigurableAgent('configs/examples/research_agent.yml')
print(agent.run('test query'))
"
```

### REST API Server
```bash
# Start development server with auto-reload
python run_api.py

# Start server on specific host/port
python run_api.py --host 0.0.0.0 --port 8080

# Check environment configuration
python run_api.py --check-env

# Access API documentation
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

### Logging and Monitoring
```bash
# Set logging level for development
export LOG_LEVEL=DEBUG
export LOG_FORMAT=structured

# Enable file logging
export LOG_FILE=logs/agent.log
export LOG_CONSOLE=true

# Run with enhanced logging
python examples/usage_examples.py

# View structured logs
tail -f logs/agent.log

# Filter logs by component
grep "agent.core" logs/agent.log | jq .
```

## Architecture Overview

This is a **configuration-driven single agent system** using ReAct pattern agents. All behavior is defined through YAML files.

### Single Agent Architecture (ReAct Pattern)

**Configuration-First Design**: Agent behavior (LLM, prompts, tools, memory) defined in YAML. Uses LangGraph's optimized `create_react_agent()` for reasoning loops.

**Component Initialization Flow**:
1. `ConfigurableAgent` loads YAML → validates with Pydantic models  
2. `_setup_llm()` → creates LLM instance (OpenAI/Anthropic/Google/Groq)
3. `_setup_tools()` → registers built-in and custom tools via `ToolRegistry`
4. `_setup_memory()` → initializes `MemoryManager` with LangMem if enabled
5. `_setup_graph()` → uses `create_react_agent()` with LLM and tools

**ReAct Pattern**: Reasoning → Acting → Observation → Final Answer cycle. More efficient than custom graph construction.


### REST API Architecture (New)

**FastAPI Framework**: RESTful API layer built with FastAPI, providing HTTP endpoints and WebSocket support for React frontend integration.

**API Components**:
1. **src/api/main.py** - FastAPI application with middleware, CORS, and error handling
2. **src/api/routes/** - Route handlers (currently health, agents, teams available)
3. **src/api/models/** - Pydantic request/response models (check availability)
4. **src/api/middleware.py** - Custom middleware for logging, error handling, security headers
5. **src/api/utils/** - Configuration management and custom exceptions
6. **src/api/client.py** - API client utilities

**Key Features**:
- **Agent Management**: Create, read, update, delete agents via HTTP API
- **Execution APIs**: Run agents synchronously or stream via WebSocket
- **Configuration Templates**: Save and reuse agent configurations
- **Evaluation Integration**: Run evaluations and view metrics via API
- **Real-time Monitoring**: WebSocket support for streaming agent execution
- **Comprehensive Documentation**: Auto-generated OpenAPI/Swagger docs

### Core Components

**src/core/configurable_agent.py** - Main single agent orchestrator with ReAct pattern implementation

**src/core/config_loader.py** - Pydantic models for YAML validation and environment variable resolution


**src/memory/memory_manager.py** - LangMem integration (semantic/episodic/procedural memory) with retention policies

**src/tools/tool_registry.py** - Built-in tools (web_search, calculator, file_reader, file_writer, code_executor) and custom tool registration

**src/evaluation/** - New evaluation system with evaluator.py, metrics.py, test_runner.py

**src/custom_logging/** - Enhanced logging infrastructure with structured logging and formatters

**src/api/** - FastAPI REST API layer with routes, middleware, and client utilities

### Configuration Patterns

**Single Agent Configuration**:
```yaml
agent: {name, description, version}
llm: {provider, model, parameters}
prompts: {system_prompt with variables}
tools: {built_in list, custom definitions}
memory: {semantic/episodic/procedural config}
react: {max_iterations, recursion_limit}
```


**Logging Configuration**:
```yaml
logging:
  enabled: true
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "structured"  # structured or json
  console: {enabled, level, format}
  file: {enabled, path, level, max_size_mb, backup_count}
  components: {agent: INFO, evaluation: DEBUG, memory: INFO}
  correlation: {enabled, include_in_response}
  performance: {log_execution_time, log_token_usage, log_memory_operations}
  privacy: {mask_api_keys, mask_user_input, excluded_fields}
```

### Web UI Integration

**Single Agent UI** (`run_web_ui.py`): Form-based configuration with real-time YAML preview, template loading, and agent testing

**Alternative UI** (`demo_web_ui.py`): Demo interface for testing and development

### Testing Architecture

**Testing via pytest**: Use pytest directly as run_tests.py has been removed
- Unit tests for individual components in tests/ directory
- Integration tests for full agent workflows  
- Coverage reports with pytest-cov
- Manual testing via examples/ directory

**Test Patterns**:
- Mock API keys via environment variables
- Temporary YAML files for configuration testing
- Pydantic validation edge case testing
- Tool registry custom tool testing
- Memory manager multi-type testing

### Common Debugging

**Agent Initialization Issues**:
1. Check API key environment variables are set
2. Validate YAML against Pydantic models  
3. Verify custom tool import paths
4. Enable `debug_mode: true` for detailed logging

**API Server Issues**:
1. Check if API server is running on correct port (default 8000)
2. Verify environment variables are loaded
3. Check FastAPI documentation at http://localhost:8000/docs
4. Review API logs for errors

**Web UI Connection Issues**:
1. Ensure API server is started before web UI
2. Check API_BASE_URL configuration
3. Verify CORS settings in API middleware
4. Test API endpoints directly via curl or browser