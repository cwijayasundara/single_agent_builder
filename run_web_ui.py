#!/usr/bin/env python3
"""
Launch script for the Configurable LangGraph Agents Web UI
Now with API server health checks and integration
"""
import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def check_streamlit():
    """Check if Streamlit is available."""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def check_api_server(base_url="http://localhost:8000", timeout=5):
    """Check if API server is running and accessible."""
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout, requests.RequestException):
        return False

def start_api_server():
    """Attempt to start the API server in background."""
    try:
        print("🚀 Starting API server...")
        api_process = subprocess.Popen(
            [sys.executable, "run_api.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if server is running
        if check_api_server():
            print("✅ API server started successfully!")
            return api_process
        else:
            print("❌ API server failed to start properly")
            api_process.terminate()
            return None
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None

def activate_venv_and_run():
    """Activate virtual environment and run Streamlit."""
    venv_path = os.path.join(os.path.dirname(__file__), '.venv')
    if os.path.exists(venv_path):
        print("🔧 Activating virtual environment...")
        if sys.platform == "win32":
            # Windows
            activate_script = os.path.join(venv_path, 'Scripts', 'activate.bat')
            cmd = f'"{activate_script}" && python -m streamlit run web_ui.py'
            subprocess.run(cmd, shell=True)
        else:
            # Unix/Linux/macOS
            activate_script = os.path.join(venv_path, 'bin', 'activate')
            cmd = f'source "{activate_script}" && python -m streamlit run web_ui.py'
            subprocess.run(cmd, shell=True, executable='/bin/bash')
    else:
        print("❌ Virtual environment not found. Please run: python3 -m venv .venv")
        print("Then: source .venv/bin/activate && pip install -r requirements.txt")

def main():
    """Main function to launch the Web UI with API integration."""
    print("🚀 Launching Configurable LangGraph Agents Web UI...")
    print("🔗 Checking API server connection...")
    
    # Check API server status
    api_running = check_api_server()
    
    if not api_running:
        print("⚠️  API server is not running.")
        print("🔍 The Web UI now integrates with the REST API for full functionality.")
        
        response = input("🤔 Would you like to start the API server automatically? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            api_process = start_api_server()
            if not api_process:
                print("❌ Failed to start API server automatically.")
                print("💡 Please start it manually with: python run_api.py")
                print("🔄 Then restart this script.")
                return
        else:
            print("⚠️  Continuing without API server...")
            print("💡 Some features will be limited. Start API server with: python run_api.py")
    else:
        print("✅ API server is running and accessible!")
    
    # Check if Streamlit is available
    if check_streamlit():
        print("✅ Streamlit is available. Starting Web UI...")
        print("🌐 Web UI will be available at: http://localhost:8501")
        if api_running or check_api_server():
            print("🌐 API server is available at: http://localhost:8000")
            print("📄 API docs available at: http://localhost:8000/docs")
        
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "web_ui.py"])
        except KeyboardInterrupt:
            print("\n👋 Web UI stopped by user.")
        except Exception as e:
            print(f"❌ Error starting Web UI: {e}")
    else:
        print("❌ Streamlit is not installed.")
        print("🔧 Attempting to use virtual environment...")
        activate_venv_and_run()

if __name__ == "__main__":
    main() 