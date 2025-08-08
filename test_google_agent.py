#!/usr/bin/env python3
"""
Test script to verify Google/Gemini agent creation works correctly.
"""
import requests
import pytest
import json
import time

API_BASE = "http://localhost:8002"

def test_google_agent_creation():
    """Test Google/Gemini agent creation with fixes."""
    print("🧪 Testing Google/Gemini agent creation...")
    # Skip if API is not running
    try:
        if requests.get(f"{API_BASE}/api/health/", timeout=3).status_code != 200:
            pytest.skip("API server not running")
    except Exception:
        pytest.skip("API server not running")
    
    # Test configuration with Google provider
    agent_config = {
        "name": "Google Test Agent",
        "description": "Test agent for Google/Gemini provider",
        "version": "1.0.0",
        "llm": {
            "provider": "google",
            "model": "gemini-1.5-flash",
            "temperature": 0.3,
            "max_tokens": 2000
        },
        "prompts": {
            "system_prompt": "You are a helpful AI assistant powered by Google's Gemini model.",
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
        "tags": ["test", "google", "gemini"]
    }
    
    print("1. Testing Google agent creation...")
    print(f"   Configuration: {json.dumps(agent_config, indent=2)}")
    
    try:
        response = requests.post(f"{API_BASE}/api/agents/", json=agent_config, timeout=60)
        print(f"   Status Code: {response.status_code}")
        response_data = response.json()
        print(f"   Response: {json.dumps(response_data, indent=2)}")
        
        if response.status_code == 201:
            agent_data = response_data
            agent_id = agent_data["agent"]["id"]
            print(f"   ✅ Successfully created Google agent: {agent_id}")
            
            # Test 2: Check agent status
            print(f"\n2. Testing agent status...")
            status_response = requests.get(f"{API_BASE}/api/agents/{agent_id}/status", timeout=10)
            print(f"   Status Code: {status_response.status_code}")
            status_data = status_response.json()
            print(f"   Response: {json.dumps(status_data, indent=2)}")
            
            agent_status = status_data.get("data", {}).get("status")
            print(f"   Agent Status: {agent_status}")
            
            if agent_status == "active":
                print("   ✅ Agent is active and ready")
                # store agent_id for next test via print marker (pytest won't share state)
                print(f"AGENT_ID={agent_id}")
                assert True
                return
            elif agent_status == "error":
                error_msg = status_data.get("data", {}).get("error_message", "Unknown error")
                print(f"   ❌ Agent is in error state: {error_msg}")
                assert False
                return
            else:
                print(f"   ⚠️  Agent is in {agent_status} state")
                return agent_id, False
                
        elif response.status_code == 400:
            print("   ❌ Configuration error - this should not happen with our fixes")
            error_detail = response_data.get("detail", {})
            print(f"   Error details: {error_detail}")
            assert False
            return
        else:
            print(f"   ❌ Unexpected status code: {response.status_code}")
            return None, False
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        assert False
        return

def test_agent_run():
    """Test running the Google agent."""
    # Skip if API is not running
    try:
        if requests.get(f"{API_BASE}/api/health/", timeout=3).status_code != 200:
            pytest.skip("API server not running")
    except Exception:
        pytest.skip("API server not running")
    # Read agent_id from status endpoint list (fallback: skip)
    try:
        agents = requests.get(f"{API_BASE}/api/agents/", timeout=10).json().get("agents", [])
        agent_id = agents[0]["id"] if agents else None
    except Exception:
        agent_id = None
    if not agent_id:
        print("   Skipping run test - no agent ID available")
        return
        
    print(f"\n3. Testing agent run...")
    
    run_request = {
        "query": "What is artificial intelligence? Give me a brief explanation.",
        "timeout": 30
    }
    
    try:
        response = requests.post(f"{API_BASE}/api/agents/{agent_id}/run", json=run_request, timeout=60)
        print(f"   Status Code: {response.status_code}")
        response_data = response.json()
        print(f"   Response: {json.dumps(response_data, indent=2)}")
        
        if response.status_code == 200:
            result = response_data.get("result", {})
            ai_response = result.get("response", "")
            execution_time = result.get("execution_time", 0)
            print(f"   ✅ Agent ran successfully!")
            print(f"   Execution time: {execution_time:.2f}s")
            print(f"   Response preview: {ai_response[:200]}...")
            assert True
            return
        else:
            print(f"   ❌ Run failed with status {response.status_code}")
            assert False
            return
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Run request failed: {e}")
        assert False
        return

def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE}/api/health/", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("🚀 Testing Google/Gemini Agent Creation Fixes")
    print("=" * 60)
    
    # Check if API is running
    if not check_api_health():
        print(f"❌ API server is not running on {API_BASE}")
        print("   Please start the server with: python run_api.py --port 8002")
        exit(1)
    
    print("✅ API server is running")
    
    # Run tests
    agent_id, creation_success = test_google_agent_creation()
    
    if creation_success:
        run_success = test_agent_run(agent_id)
        
        if run_success:
            print("\n✅ All Google/Gemini tests passed!")
            print("   - Agent creation: SUCCESS")
            print("   - Agent status: ACTIVE")  
            print("   - Agent execution: SUCCESS")
        else:
            print("\n⚠️  Google/Gemini agent created but run failed")
            print("   - Agent creation: SUCCESS")
            print("   - Agent execution: FAILED")
    else:
        print("\n❌ Google/Gemini agent creation failed")
    
    print("=" * 60)