#!/usr/bin/env python3
"""
Setup script for Configurable LangGraph Agents
"""
import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected.")
    return True

def create_venv():
    """Create virtual environment if it doesn't exist."""
    venv_path = os.path.join(os.path.dirname(__file__), '.venv')
    if os.path.exists(venv_path):
        print("✅ Virtual environment already exists.")
        return True
    
    print("🔧 Creating virtual environment...")
    try:
        subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)
        print("✅ Virtual environment created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        # Activate venv and install requirements
        if sys.platform == "win32":
            # Windows
            activate_script = os.path.join('.venv', 'Scripts', 'activate.bat')
            cmd = f'"{activate_script}" && pip install -r requirements.txt'
            subprocess.run(cmd, shell=True, check=True)
        else:
            # Unix/Linux/macOS
            activate_script = os.path.join('.venv', 'bin', 'activate')
            cmd = f'source "{activate_script}" && pip install -r requirements.txt'
            subprocess.run(cmd, shell=True, executable='/bin/bash', check=True)
        
        print("✅ Dependencies installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_env_file():
    """Create .env file from example if it doesn't exist."""
    env_file = '.env'
    env_example = 'env.example'
    
    if os.path.exists(env_file):
        print("✅ .env file already exists.")
        return True
    
    if os.path.exists(env_example):
        print("📝 Creating .env file from example...")
        try:
            shutil.copy(env_example, env_file)
            print("✅ .env file created. Please edit it with your API keys.")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False
    else:
        print("⚠️ env.example not found. Please create .env file manually.")
        return False

def test_installation():
    """Test if everything is working."""
    print("🧪 Testing installation...")
    try:
        # Test imports
        if sys.platform != "win32":
            activate_script = os.path.join('.venv', 'bin', 'activate')
            cmd = f'source "{activate_script}" && python -c "import streamlit; print(\'Streamlit available\')"'
            subprocess.run(cmd, shell=True, executable='/bin/bash', check=True)
        
        print("✅ Installation test passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Configurable LangGraph Agents...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_venv():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup environment file
    setup_env_file()
    
    # Test installation
    if not test_installation():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python3 run_web_ui.py")
    print("3. Open browser to: http://localhost:8501")
    print("\n💡 For hierarchical agents:")
    print("   - Run: python3 examples/hierarchical_agent_example.py")

if __name__ == "__main__":
    main() 