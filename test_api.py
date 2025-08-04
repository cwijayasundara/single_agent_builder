#!/usr/bin/env python3
"""
Test script for the Configurable Agents API.
"""
import asyncio
import json
import os
from typing import Dict, Any

import httpx


async def test_api():
    """Test the API endpoints."""
    base_url = "http://127.0.0.1:8000"
    
    async with httpx.AsyncClient() as client:
        print("🧪 Testing Configurable Agents API...")
        
        # Test root endpoint
        print("\n1. Testing root endpoint...")
        try:
            response = await client.get(f"{base_url}/")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test health check
        print("\n2. Testing health endpoint...")
        try:
            response = await client.get(f"{base_url}/api/health/")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test readiness check
        print("\n3. Testing readiness endpoint...")
        try:
            response = await client.get(f"{base_url}/api/health/ready")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test liveness check
        print("\n4. Testing liveness endpoint...")
        try:
            response = await client.get(f"{base_url}/api/health/live")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test agent list
        print("\n5. Testing agent list...")
        try:
            response = await client.get(f"{base_url}/api/agents/")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test creating an agent (if API keys are available)
        if os.getenv("OPENAI_API_KEY"):
            print("\n6. Testing agent creation...")
            agent_data = {
                "name": "Test Agent",
                "description": "A test agent for API validation",
                "version": "1.0.0",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7
                },
                "prompts": {
                    "system_prompt": "You are a helpful assistant.",
                    "variables": {}
                },
                "tools": ["web_search"],
                "memory": [],
                "evaluation": {
                    "enabled": False
                },
                "react": {
                    "max_iterations": 10,
                    "recursion_limit": 25
                },
                "debug_mode": False,
                "tags": ["test", "api-validation"]
            }
            
            try:
                response = await client.post(
                    f"{base_url}/api/agents/",
                    json=agent_data
                )
                print(f"Status: {response.status_code}")
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2, default=str)}")
                
                if response.status_code == 200:
                    agent_id = result["agent"]["id"]
                    
                    # Test running the agent
                    print(f"\n7. Testing agent run (ID: {agent_id})...")
                    run_data = {
                        "query": "Hello, can you introduce yourself?",
                        "context": {},
                        "stream": False,
                        "include_evaluation": False
                    }
                    
                    try:
                        response = await client.post(
                            f"{base_url}/api/agents/{agent_id}/run",
                            json=run_data
                        )
                        print(f"Status: {response.status_code}")
                        result = response.json()
                        print(f"Response: {json.dumps(result, indent=2, default=str)}")
                    except Exception as e:
                        print(f"Error running agent: {e}")
                    
                    # Test getting agent status
                    print(f"\n8. Testing agent status (ID: {agent_id})...")
                    try:
                        response = await client.get(f"{base_url}/api/agents/{agent_id}/status")
                        print(f"Status: {response.status_code}")
                        print(f"Response: {response.json()}")
                    except Exception as e:
                        print(f"Error getting agent status: {e}")
                    
                    # Test deleting the agent
                    print(f"\n9. Cleaning up agent (ID: {agent_id})...")
                    try:
                        response = await client.delete(f"{base_url}/api/agents/{agent_id}")
                        print(f"Status: {response.status_code}")
                        print(f"Response: {response.json()}")
                    except Exception as e:
                        print(f"Error deleting agent: {e}")
                        
            except Exception as e:
                print(f"Error creating agent: {e}")
        else:
            print("\n6. Skipping agent creation test (no OPENAI_API_KEY)")
        
        print("\n🎉 API testing completed!")


def run_test():
    """Run the async test."""
    asyncio.run(test_api())


if __name__ == "__main__":
    run_test()