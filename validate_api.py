#!/usr/bin/env python3
"""
Validation script for the Configurable Agents API structure.
"""
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def validate_api_structure():
    """Validate API structure and imports."""
    print("🔍 Validating API structure...")
    
    # Check file structure
    api_files = [
        "src/api/__init__.py",
        "src/api/main.py",
        "src/api/middleware.py",
        "src/api/models/__init__.py",
        "src/api/models/requests.py",
        "src/api/models/responses.py",
        "src/api/routes/__init__.py",
        "src/api/routes/agents.py",
        "src/api/routes/health.py",
        "src/api/utils/__init__.py",
        "src/api/utils/config.py",
        "src/api/utils/exceptions.py",
    ]
    
    missing_files = []
    for file_path in api_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Test basic imports
    print("\n🧪 Testing imports...")
    try:
        from src.api.models.requests import AgentCreateRequest, AgentRunRequest
        from src.api.models.responses import AgentResponse, AgentRunResponse
        from src.api.utils.config import config
        from src.api.utils.exceptions import ConfigurableAgentException
        print("✅ API models imported successfully")
        
        # Test Pydantic model creation
        agent_request = AgentCreateRequest(
            name="Test Agent",
            llm={
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "temperature": 0.7
            },
            prompts={
                "system_prompt": "You are a helpful assistant.",
                "variables": {}
            }
        )
        print("✅ Pydantic model validation works")
        
        # Test config
        print(f"✅ Config loaded: {config.environment} environment")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False
    
    print("\n🎉 API structure validation completed successfully!")
    return True


def validate_dependencies():
    """Check if key dependencies are available."""
    print("\n📦 Checking dependencies...")
    
    required_deps = [
        "pydantic",
        "pydantic_settings", 
        "enum",
        "typing",
        "datetime",
        "uuid",
        "logging",
        "os"
    ]
    
    missing_deps = []
    for dep in required_deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            missing_deps.append(dep)
            print(f"❌ {dep}")
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {missing_deps}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    return True


def main():
    """Main validation function."""
    print("🚀 Configurable Agents API Validation")
    print("=" * 50)
    
    structure_ok = validate_api_structure()
    deps_ok = validate_dependencies()
    
    if structure_ok and deps_ok:
        print("\n✅ All validations passed!")
        print("💡 You can now run: python run_api.py")
        return 0
    else:
        print("\n❌ Some validations failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())