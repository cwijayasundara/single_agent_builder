#!/usr/bin/env python3
"""
Demo script showcasing the Web UI functionality for Configurable LangGraph Agents
"""

def demo_config_creation():
    """Demo of programmatic config creation that mirrors the web UI."""
    
    # This shows what the web UI creates behind the scenes
    demo_config = {
        "agent": {
            "name": "Demo Research Agent",
            "description": "A demonstration AI agent for research tasks",
            "version": "1.0.0"
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 2000,
            "api_key_env": "OPENAI_API_KEY"
        },
        "prompts": {
            "system_prompt": {
                "template": "You are a helpful research assistant. Query: {query}",
                "variables": ["query"]
            },
            "user_prompt": {
                "template": "User request: {user_input}",
                "variables": ["user_input"]
            }
        },
        "tools": {
            "built_in": ["web_search", "file_reader"],
            "custom": []
        },
        "memory": {
            "enabled": True,
            "provider": "langmem",
            "types": {
                "semantic": True,
                "episodic": True,
                "procedural": True
            },
            "storage": {
                "backend": "memory"
            },
            "settings": {
                "max_memory_size": 5000,
                "retention_days": 30,
                "background_processing": True
            }
        },
        "react": {
            "max_iterations": 10,
            "recursion_limit": 50
        },
        "optimization": {
            "enabled": False,
            "prompt_optimization": {
                "enabled": False,
                "feedback_collection": False,
                "ab_testing": False,
                "optimization_frequency": "weekly"
            },
            "performance_tracking": {
                "enabled": False,
                "metrics": ["response_time", "accuracy"]
            }
        },
        "runtime": {
            "max_iterations": 50,
            "timeout_seconds": 300,
            "retry_attempts": 3,
            "debug_mode": False
        }
    }
    
    return demo_config

def main():
    """Main demo function."""
    print("🤖 Configurable LangGraph Agents - Web UI Demo")
    print("=" * 50)
    
    print("\n📋 Demo Configuration Structure:")
    print("This is what the Web UI helps you create:")
    print()
    
    config = demo_config_creation()
    
    # Simple YAML-like output (without requiring pyyaml dependency)
    print("agent:")
    print(f"  name: {config['agent']['name']}")
    print(f"  description: {config['agent']['description']}")
    print(f"  version: {config['agent']['version']}")
    print("\nllm:")
    print(f"  provider: {config['llm']['provider']}")
    print(f"  model: {config['llm']['model']}")
    print(f"  temperature: {config['llm']['temperature']}")
    print("\n... (and all other configuration sections)")
    
    print("\n🌐 To use the Web UI:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up your API keys in .env file (copy from env.example)")
    print("3. Launch the UI: python run_web_ui.py")
    print("4. Open your browser to: http://localhost:8501")
    
    print("\n✨ Web UI Features:")
    features = [
        "🎯 Multi-tab interface for easy configuration",
        "📝 Form-based input with validation",
        "💾 Load/save YAML configurations",
        "🔍 Real-time preview and validation",
        "🧪 Test your agent directly in the browser",
        "🚀 Quick templates for common use cases",
        "🛠️ Built-in and custom tool configuration",
        "🧠 Memory management settings",
        "⚡ Performance optimization options"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print("\n🚀 Available Agent Templates:")
    templates = [
        "🔬 Research Agent",
        "💻 Coding Assistant", 
        "🎧 Customer Support",
        "🤖 Gemini Agent",
        "⚡ Groq Agent"
    ]
    
    for template in templates:
        print(f"   {template}")
    
    print("\n📖 Documentation:")
    print("   • README.md - Main project documentation")
    print("   • WEB_UI_GUIDE.md - Detailed web UI guide")
    print("   • configs/examples/ - Example configurations")
    
    print("\n🎉 Happy agent building!")

if __name__ == "__main__":
    main() 