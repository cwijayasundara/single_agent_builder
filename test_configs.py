#!/usr/bin/env python3

import yaml
import os
from pathlib import Path

def test_config(config_path):
    """Test loading a configuration file."""
    try:
        print(f'✓ Testing {config_path}...')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        agent_info = config.get('agent', {})
        llm_info = config.get('llm', {})
        memory_info = config.get('memory', {})
        optimization_info = config.get('optimization', {})
        evaluation_info = config.get('evaluation', {})
        logging_info = config.get('logging', {})
        
        print(f'  Agent name: {agent_info.get("name", "N/A")}')
        print(f'  LLM provider: {llm_info.get("provider", "N/A")}')
        print(f'  Memory enabled: {memory_info.get("enabled", False)}')
        print(f'  Optimization enabled: {optimization_info.get("enabled", False)}')
        print(f'  Evaluation enabled: {evaluation_info.get("enabled", False)}')
        print(f'  Logging enabled: {logging_info.get("enabled", False)}')
        print('  ✅ YAML loaded successfully!')
        print()
        return True
    except Exception as e:
        print(f'  ❌ Error loading {config_path}: {str(e)}')
        print()
        return False

def main():
    """Test all configuration files."""
    configs = [
        'configs/examples/minimal_agent.yml',
        'configs/examples/research_agent.yml', 
        'configs/examples/coding_assistant.yml',
        'configs/examples/customer_support.yml',
        'configs/examples/web_browser_agent.yml',
        'configs/examples/writer_agent.yml',
        'configs/examples/groq_agent.yml',
        'configs/examples/google_agent.yml',
        'configs/examples/research_agent_with_logging.yml',
        'configs/examples/evaluation_enabled_agent.yml'
    ]

    print('Testing YAML configuration files...')
    print('=' * 50)
    
    success_count = 0
    total_count = len(configs)
    
    for config_path in configs:
        if test_config(config_path):
            success_count += 1
    
    print(f'Configuration validation complete!')
    print(f'Success: {success_count}/{total_count} configurations loaded successfully')
    
    if success_count == total_count:
        print('🎉 All configurations are valid!')
    else:
        print('⚠️ Some configurations have issues')

if __name__ == '__main__':
    main()