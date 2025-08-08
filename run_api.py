#!/usr/bin/env python3
"""
Development server runner for the Configurable Agents API.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  dotenv not available, using system environment variables only")

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import uvicorn
    from src.api.main import app
    from src.api.utils.config import config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


def check_environment():
    """Check if environment is properly configured."""
    print("🔍 Checking environment configuration...")
    
    # Check API keys
    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "LANGSMITH_API_KEY": os.getenv("LANGSMITH_API_KEY"),
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
    }
    
    available_keys = [key for key, value in api_keys.items() if value]
    missing_keys = [key for key, value in api_keys.items() if not value]
    
    if available_keys:
        print(f"✅ Available API keys: {', '.join(available_keys)}")
    
    if missing_keys:
        print(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("💡 Set them in your .env file or as environment variables")
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ Found .env file")
    else:
        print("⚠️  No .env file found")
        print("💡 Copy env.example to .env and configure your API keys")
    
    print(f"🔧 API Configuration:")
    print(f"   Host: {config.host}")
    print(f"   Port: {config.port}")
    print(f"   Debug: {config.debug}")
    print(f"   Environment: {config.environment}")
    print()


def run_server(host: str = None, port: int = None, reload: bool = True, log_level: str = "info"):
    """Run the development server."""
    # Check environment for development mode, skip for production
    if os.getenv("ENVIRONMENT") != "production":
        check_environment()
    
    # Use environment variables first, then config defaults
    host = host or os.getenv("HOST") or config.host
    port = port or int(os.getenv("PORT", config.port))
    
    # Disable reload in production
    if os.getenv("ENVIRONMENT") == "production":
        reload = False
    
    environment = os.getenv("ENVIRONMENT", "development")
    workers = int(os.getenv("MAX_WORKERS", 1))
    
    print(f"🚀 Starting Configurable Agents API...")
    print(f"   📍 URL: http://{host}:{port}")
    print(f"   📚 Docs: http://{host}:{port}/docs")
    print(f"   🔄 Reload: {reload}")
    print(f"   🌍 Environment: {environment}")
    print(f"   👥 Workers: {workers}")
    print()
    
    try:
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
            access_log=True,
            reload_dirs=["src"] if reload else None,
            workers=workers if not reload else 1
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down API server...")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the Configurable Agents API development server"
    )
    parser.add_argument(
        "--host", 
        default=None, 
        help=f"Host to bind to (default: {config.host})"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=None, 
        help=f"Port to bind to (default: {config.port})"
    )
    parser.add_argument(
        "--no-reload", 
        action="store_true", 
        help="Disable auto-reload"
    )
    parser.add_argument(
        "--log-level", 
        choices=["critical", "error", "warning", "info", "debug"], 
        default="info",
        help="Log level (default: info)"
    )
    parser.add_argument(
        "--check-env", 
        action="store_true", 
        help="Check environment and exit"
    )
    
    args = parser.parse_args()
    
    if args.check_env:
        check_environment()
        return
    
    run_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()