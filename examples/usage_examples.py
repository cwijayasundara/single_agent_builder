"""
Comprehensive usage examples for the configurable agent system.

This module demonstrates:
- Basic agent usage and configuration
- Memory and optimization features  
- Different LLM providers (OpenAI, Anthropic, Google Gemini, Groq)
- Specialized agents (Research, Coding, Customer Support, Web Browser, Writer)
- REST API integration with web interfaces
- Performance comparisons and benchmarking
- Custom tools integration
- Async operations
- Error handling and troubleshooting

Run with: python examples/usage_examples.py
Requires: API keys set in .env file
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables from .env file
load_dotenv()

from src.core.configurable_agent import ConfigurableAgent
from src.optimization.prompt_optimizer import OptimizationMetric

# Check if API client is available
try:
    from src.api.client import ConfigurableAgentsAPIClient
    API_CLIENT_AVAILABLE = True
except ImportError:
    print("⚠️ API client modules not available. API examples will be skipped.")
    API_CLIENT_AVAILABLE = False


def basic_usage_example():
    """Basic usage of a configurable agent."""
    print("=== Basic Usage Example ===")
    
    # Create agent from configuration
    agent = ConfigurableAgent("configs/examples/research_agent.yml")
    
    # Run a simple query
    response = agent.run("What are the latest developments in quantum computing?")
    
    print("Agent Response:")
    print(response["response"])
    print(f"Iterations: {response['iteration_count']}")
    print(f"Tools Used: {list(response['tool_results'].keys())}")


def memory_example():
    """Example showing memory functionality."""
    print("\\n=== Memory Example ===")
    
    agent = ConfigurableAgent("configs/examples/coding_assistant.yml")
    
    # First interaction
    response1 = agent.run("I'm working on a Python web scraper using requests")
    print("First interaction:")
    print(response1["response"][:200] + "...")
    
    # Second interaction - agent should remember context
    response2 = agent.run("How can I add error handling to handle timeouts?")
    print("\\nSecond interaction (with memory):")
    print(response2["response"][:200] + "...")
    
    # Check memory stats
    memory_stats = agent.get_memory_stats()
    print(f"\\nMemory stats: {memory_stats}")


def optimization_example():
    """Example showing prompt optimization."""
    print("\\n=== Optimization Example ===")
    
    agent = ConfigurableAgent("configs/examples/customer_support.yml")
    
    # Simulate multiple interactions with feedback
    test_queries = [
        "My order hasn't arrived yet",
        "I need to return a product",
        "How do I reset my password?",
        "I was charged twice for the same order"
    ]
    
    for i, query in enumerate(test_queries):
        response = agent.run(query, interaction_id=f"test_{i}")
        
        # Simulate user satisfaction feedback
        satisfaction_score = 0.8 if i % 2 == 0 else 0.6
        
        print(f"Query {i+1}: {query[:30]}...")
        print(f"Response length: {len(response['response'])} chars")
        print(f"Simulated satisfaction: {satisfaction_score}")
        print()


async def async_usage_example():
    """Example of async usage."""
    print("\\n=== Async Usage Example ===")
    
    agent = ConfigurableAgent("configs/examples/research_agent.yml")
    
    # Run multiple queries concurrently
    queries = [
        "What is machine learning?",
        "Explain blockchain technology",
        "What are the benefits of renewable energy?"
    ]
    
    tasks = [agent.arun(query) for query in queries]
    responses = await asyncio.gather(*tasks)
    
    for i, (query, response) in enumerate(zip(queries, responses)):
        print(f"Query {i+1}: {query}")
        print(f"Response: {response['response'][:100]}...")
        print()


def custom_tools_example():
    """Example showing how to add custom tools."""
    print("\\n=== Custom Tools Example ===")
    
    # Create a simple custom tool function
    def database_query(query: str) -> str:
        """Mock database query tool."""
        return f"Database result for: {query}"
    
    agent = ConfigurableAgent("configs/examples/coding_assistant.yml")
    
    # Register custom tool
    agent.tool_registry.register_function_as_tool(
        name="database_query",
        func=database_query,
        description="Query the database for information"
    )
    
    print("Available tools:")
    tools = agent.get_available_tools()
    for tool_name, description in tools.items():
        print(f"- {tool_name}: {description}")


def configuration_management_example():
    """Example showing configuration management."""
    print("\\n=== Configuration Management Example ===")
    
    agent = ConfigurableAgent("configs/examples/research_agent.yml")
    
    # Get current configuration
    config = agent.get_config()
    print(f"Agent: {config.agent.name}")
    print(f"LLM: {config.llm.provider} - {config.llm.model}")
    print(f"Memory enabled: {config.memory.enabled}")
    
    # Update prompts dynamically
    new_prompts = {
        "system_prompt": "You are a specialized scientific research assistant focused on peer-reviewed sources."
    }
    agent.update_prompts(new_prompts)
    print("\\nPrompts updated successfully")
    
    # Get formatted prompt
    formatted_prompt = agent.get_prompt_template(
        "user_prompt", 
        user_input="Latest cancer research findings"
    )
    print(f"\\nFormatted prompt sample: {formatted_prompt[:100]}...")


def error_handling_example():
    """Example showing error handling."""
    print("\\n=== Error Handling Example ===")
    
    try:
        # Try to create agent with non-existent config
        agent = ConfigurableAgent("configs/nonexistent.yml")
    except FileNotFoundError as e:
        print(f"Expected error: {e}")
    
    # Create agent with valid config
    agent = ConfigurableAgent("configs/examples/research_agent.yml")
    
    # Test with invalid input
    response = agent.run("")  # Empty input
    print(f"Empty input response: {response['response'][:100]}...")
    
    if "error" in response:
        print(f"Error handled: {response['error']}")


def gemini_example():
    """Example using Gemini LLM."""
    print("\\n=== Gemini Agent Example ===")
    
    try:
        agent = ConfigurableAgent("configs/examples/gemini_agent.yml")
        
        config = agent.get_config()
        print(f"Agent: {config.agent.name}")
        print(f"LLM: {config.llm.provider} - {config.llm.model}")
        
        response = agent.run("What are the latest developments in quantum computing?")
        print(f"Response: {response['response'][:200]}...")
        
    except Exception as e:
        print(f"Gemini example failed: {e}")


def groq_example():
    """Example using Groq LLM."""
    print("\\n=== Groq Agent Example ===")
    
    try:
        agent = ConfigurableAgent("configs/examples/groq_agent.yml")
        
        config = agent.get_config()
        print(f"Agent: {config.agent.name}")
        print(f"LLM: {config.llm.provider} - {config.llm.model}")
        
        response = agent.run("Write a Python function to calculate fibonacci numbers")
        print(f"Response: {response['response'][:200]}...")
        
    except Exception as e:
        print(f"Groq example failed: {e}")


def web_browser_agent_example():
    """Example using the specialized Web Browser Agent."""
    print("\\n=== Web Browser Agent Example ===")
    
    try:
        agent = ConfigurableAgent("configs/examples/web_browser_agent.yml")
        
        config = agent.get_config()
        print(f"Agent: {config.agent.name}")
        print(f"Specialization: Web search and information gathering")
        print(f"Tools: {', '.join(config.tools.built_in)}")
        
        # Test web search capabilities
        response = agent.run("Find the latest news about renewable energy developments")
        print(f"Search Results: {response['response'][:300]}...")
        print(f"Tools Used: {list(response.get('tool_results', {}).keys())}")
        
    except Exception as e:
        print(f"Web Browser Agent example failed: {e}")


def writer_agent_example():
    """Example using the specialized Writer Agent."""
    print("\\n=== Writer Agent Example ===")
    
    try:
        agent = ConfigurableAgent("configs/examples/writer_agent.yml")
        
        config = agent.get_config()
        print(f"Agent: {config.agent.name}")
        print(f"Specialization: Content creation and writing")
        print(f"Temperature: {config.llm.temperature} (higher for creativity)")
        
        # Test content creation capabilities
        response = agent.run("Write a brief guide on sustainable living practices")
        print(f"Generated Content: {response['response'][:300]}...")
        print(f"Content Length: {len(response['response'])} characters")
        
    except Exception as e:
        print(f"Writer Agent example failed: {e}")


def api_client_example():
    """Example using the REST API client."""
    print("\\n=== REST API Client Example ===")
    
    if not API_CLIENT_AVAILABLE:
        print("❌ API client modules not available. Skipping API example.")
        return
    
    try:
        # Create API client
        print("🌐 Testing REST API Client:")
        client = ConfigurableAgentsAPIClient()
        
        # Check API health
        if not client.check_health():
            print("❌ API server not available. Make sure to run: python run_api.py")
            return
        
        print("✅ API server is running")
        
        # Create an agent via API
        print("\\n🤖 Creating Agent via API:")
        agent_data = {
            "name": "API Demo Agent",
            "description": "Agent created via REST API",
            "llm": {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "temperature": 0.7
            },
            "prompts": {
                "system_prompt": "You are a helpful research assistant."
            },
            "tools": ["web_search"]
        }
        
        agent_response = client.create_agent(agent_data)
        if agent_response:
            agent_id = agent_response.get('agent', {}).get('id')
            print(f"✅ Created agent: {agent_id}")
            
            # Run a query via API
            print("\\n📊 Running Query via API:")
            result = client.run_agent(agent_id, "What are the benefits of renewable energy?")
            if result:
                print(f"Result: {result.get('result', {}).get('response', '')[:200]}...")
            
            # Clean up - delete the agent
            if client.delete_agent(agent_id):
                print("🗑️ Agent cleaned up successfully")
        else:
            print("❌ Failed to create agent via API")
        
    except Exception as e:
        print(f"API client example failed: {e}")


def comparative_llm_example():
    """Example comparing different LLM providers on the same task."""
    print("\\n=== Comparative LLM Example ===")
    
    task = "Explain quantum computing in simple terms"
    configs = [
        ("configs/examples/research_agent.yml", "OpenAI"),
        ("configs/examples/gemini_agent.yml", "Google Gemini"),
        ("configs/examples/groq_agent.yml", "Groq")
    ]
    
    results = {}
    
    for config_file, provider_name in configs:
        try:
            agent = ConfigurableAgent(config_file)
            response = agent.run(task)
            
            results[provider_name] = {
                'response_length': len(response['response']),
                'iterations': response.get('iteration_count', 0),
                'preview': response['response'][:150] + "..."
            }
            
            print(f"\\n🤖 {provider_name}:")
            print(f"   Length: {results[provider_name]['response_length']} chars")
            print(f"   Iterations: {results[provider_name]['iterations']}")
            print(f"   Preview: {results[provider_name]['preview']}")
            
        except Exception as e:
            print(f"\\n❌ {provider_name} failed: {e}")
    
    # Summary comparison
    if results:
        print("\\n📊 Comparison Summary:")
        for provider, data in results.items():
            print(f"   {provider}: {data['response_length']} chars, {data['iterations']} iterations")


def specialized_agents_showcase():
    """Showcase different specialized agent templates."""
    print("\\n=== Specialized Agents Showcase ===")
    
    agents_to_test = [
        ("configs/examples/research_agent.yml", "Research specialist", "What is machine learning?"),
        ("configs/examples/coding_assistant.yml", "Coding assistant", "Write a Python function to sort a list"),
        ("configs/examples/customer_support.yml", "Customer support", "I need help with my account"),
        ("configs/examples/web_browser_agent.yml", "Web browser specialist", "Find news about climate change"),
        ("configs/examples/writer_agent.yml", "Writing specialist", "Write a product description for eco-friendly soap")
    ]
    
    for config_file, description, test_query in agents_to_test:
        try:
            agent = ConfigurableAgent(config_file)
            config = agent.get_config()
            
            print(f"\\n🤖 {config.agent.name} ({description}):")
            print(f"   Model: {config.llm.provider} - {config.llm.model}")
            print(f"   Temperature: {config.llm.temperature}")
            print(f"   Test: {test_query}")
            
            response = agent.run(test_query)
            print(f"   Result: {response['response'][:100]}...")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def performance_comparison_example():
    """Example showing performance comparison between agents."""
    print("\\n=== Performance Comparison Example ===")
    
    import time
    
    test_query = "Explain the concept of artificial intelligence"
    configs = [
        "configs/examples/research_agent.yml",
        "configs/examples/coding_assistant.yml"
    ]
    
    for config_file in configs:
        try:
            agent = ConfigurableAgent(config_file)
            config = agent.get_config()
            
            start_time = time.time()
            response = agent.run(test_query)
            end_time = time.time()
            
            print(f"\\n⏱️ {config.agent.name}:")
            print(f"   Response time: {end_time - start_time:.2f} seconds")
            print(f"   Response length: {len(response['response'])} characters")
            print(f"   Tools used: {len(response.get('tool_results', {}))}")
            print(f"   Iterations: {response.get('iteration_count', 0)}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")


def main():
    """Run all examples."""
    # Check for environment variables
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here":
        print("Warning: OPENAI_API_KEY not set in .env file. Some examples may fail.")
        print("Please update your .env file with your actual OpenAI API key.")
    if not os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY") == "your-anthropic-api-key-here":
        print("Warning: ANTHROPIC_API_KEY not set in .env file. Some examples may fail.")
        print("Please update your .env file with your actual Anthropic API key.")
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your-google-api-key-here":
        print("Warning: GOOGLE_API_KEY not set in .env file. Gemini examples may fail.")
        print("Please update your .env file with your actual Google API key.")
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your-groq-api-key-here":
        print("Warning: GROQ_API_KEY not set in .env file. Groq examples may fail.")
        print("Please update your .env file with your actual Groq API key.")
    
    try:
        # Core functionality examples
        basic_usage_example()
        memory_example()
        optimization_example()
        custom_tools_example()
        configuration_management_example()
        error_handling_example()
        
        # LLM provider examples
        gemini_example()
        groq_example()
        
        # New specialized agent examples
        web_browser_agent_example()
        writer_agent_example()
        
        # API client example
        api_client_example()
        
        # Comparison and showcase examples
        comparative_llm_example()
        specialized_agents_showcase()
        performance_comparison_example()
        
        # Run async example
        print("\\n=== Running Async Example ===")
        asyncio.run(async_usage_example())
        
    except Exception as e:
        print(f"Example failed (likely due to missing API keys): {e}")
        print("\\nTo run examples successfully:")
        print("1. Copy .env.example to .env: cp .env.example .env")
        print("2. Edit .env file and add your actual API keys")
        print("3. Install required dependencies: pip install -r requirements.txt")
        print("\\nSupported LLM providers:")
        print("- OpenAI (gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini)")
        print("- Anthropic (Claude-3, Claude-2)")
        print("- Google Gemini (gemini-2.5-flash-preview-05-20, gemini-1.5-pro, gemini-1.0-pro)")
        print("- Groq (meta-llama/llama-4-scout-17b-16e-instruct, llama3-8b-8192, llama3-70b-8192)")
        
        print("\\nAvailable Agent Templates:")
        print("📊 Single Agents:")
        print("- Research Agent: General research and information gathering")
        print("- Coding Assistant: Programming help and code generation")
        print("- Customer Support: Customer service and support")
        print("- Web Browser Agent: Specialized web search and information gathering")
        print("- Writer Agent: Content creation and writing specialist")
        print("- Gemini Agent: Google Gemini-powered agent")
        print("- Groq Agent: High-speed Groq inference agent")
        
        print("\\n🌐 REST API:")
        print("- API Client: Programmatic agent management via REST API")
        print("- Web UI: Browser-based agent configuration and testing")
        
        print("\\n💡 Usage Tips:")
        print("- Use REST API for programmatic agent management")
        print("- Single agents are efficient for specialized tasks")
        print("- Test different LLM providers for optimal performance")
        print("- Enable memory for context-aware conversations")


if __name__ == "__main__":
    main()