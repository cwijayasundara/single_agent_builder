#!/usr/bin/env python3
"""
Quick test script to verify API error handling fixes.
"""
import requests
import json
import time

API_BASE = "http://localhost:8001"

def test_api_fixes():
    """Test the API error handling fixes."""
    print("🧪 Testing API error handling fixes...")
    
    # Test 1: Create agent with missing API key (should fail gracefully)
    print("\n1. Testing agent creation with invalid configuration...")
    
    agent_config = {
        "name": "Test Agent",
        "description": "Test agent for error handling",
        "version": "1.0.0",
        "llm": {
            "provider": "invalid_provider",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "prompts": {
            "system_prompt": "You are a helpful assistant.",
            "variables": {}
        },
        "tools": ["web_search"],
        "memory": [],
        "react": {
            "max_iterations": 10,
            "recursion_limit": 50
        },
        "evaluation": {
            "enabled": False,
            "evaluators": [],
            "metrics": [],
            "auto_evaluate": False
        },
        "debug_mode": False,
        "tags": ["test"]
    }
    
    try:
        response = requests.post(f"{API_BASE}/api/agents/", json=agent_config, timeout=30)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 400:
            print("   ✅ Correctly returned 400 for invalid configuration")
        else:
            print("   ❌ Expected 400 status code")
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False
    
    # Test 2: Create valid agent
    print("\n2. Testing agent creation with valid configuration...")
    
    valid_config = agent_config.copy()
    valid_config["llm"]["provider"] = "openai"
    
    try:
        response = requests.post(f"{API_BASE}/api/agents/", json=valid_config, timeout=30)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 201:
            agent_data = response.json()
            agent_id = agent_data["agent"]["id"]
            print(f"   ✅ Successfully created agent: {agent_id}")
            
            # Test 3: Check agent status
            print(f"\n3. Testing agent status endpoint...")
            status_response = requests.get(f"{API_BASE}/api/agents/{agent_id}/status", timeout=10)
            print(f"   Status Code: {status_response.status_code}")
            print(f"   Response: {json.dumps(status_response.json(), indent=2)}")
            
            if status_response.status_code == 200:
                print("   ✅ Successfully retrieved agent status")
            else:
                print("   ❌ Failed to retrieve agent status")
            
            # Test 4: If agent is in error state, test retry functionality
            status_data = status_response.json()
            if status_data.get("data", {}).get("status") == "error":
                print(f"\n4. Testing agent retry functionality...")
                retry_response = requests.post(f"{API_BASE}/api/agents/{agent_id}/retry", timeout=30)
                print(f"   Status Code: {retry_response.status_code}")
                print(f"   Response: {json.dumps(retry_response.json(), indent=2)}")
                
                if retry_response.status_code in [200, 400]:
                    print("   ✅ Retry endpoint working correctly")
                else:
                    print("   ❌ Retry endpoint not working as expected")
            
            return True
            
        else:
            print(f"   ❌ Expected 201, got {response.status_code}")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("🚀 Testing API Error Handling Fixes")
    print("=" * 50)
    
    # Check if API is running
    if not check_api_health():
        print("❌ API server is not running on http://localhost:8001")
        print("   Please start the server with: python run_api.py --port 8001")
        exit(1)
    
    print("✅ API server is running")
    
    # Run tests
    success = test_api_fixes()
    
    if success:
        print("\n✅ API error handling fixes are working correctly!")
    else:
        print("\n❌ Some tests failed. Check the output above.")
    
    print("=" * 50)