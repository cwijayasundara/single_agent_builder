"""
Streamlit Web UI for Configurable LangGraph Agents
"""
import streamlit as st
import yaml
import os
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
import asyncio
from datetime import datetime
import sys

# Import project modules
try:
    from src.api.client import (
        ConfigurableAgentsAPIClient, 
        APIConnectionError, 
        APIValidationError, 
        AgentNotFoundError,
        create_api_client,
        check_api_connection
    )
    API_CLIENT_AVAILABLE = True
except ImportError as e:
    API_CLIENT_AVAILABLE = False
    API_IMPORT_ERROR = str(e)
    # Create dummy classes to avoid import errors
    class APIConnectionError(Exception): pass
    class APIValidationError(Exception): pass
    class AgentNotFoundError(Exception): pass
    
    def create_api_client(*args, **kwargs):
        raise ImportError(API_IMPORT_ERROR)
    
    def check_api_connection(*args, **kwargs):
        return False

from src.tools.tool_registry import ToolRegistry

# Page configuration
st.set_page_config(
    page_title="Configurable LangGraph Agents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to optimize sidebar width (20% of screen) and ensure content is visible
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        width: 20% !important;
        min-width: 280px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 20% !important;
        min-width: 280px !important;
    }
    [data-testid="stSidebar"] > div:first-child > div:first-child {
        width: 20% !important;
        min-width: 280px !important;
    }
    .main .block-container {
        margin-left: 22% !important;
        max-width: 78% !important;
    }
    /* Ensure sidebar content is visible and ultra-compact */
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        margin: 1px 0 !important;
        padding: 0.2rem 0.5rem !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        margin: 2px 0 !important;
    }
    [data-testid="stSidebar"] .stSubheader {
        margin-top: 12px !important;
        margin-bottom: 6px !important;
        font-size: 1rem !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div {
        font-size: 0.8rem !important;
    }
    [data-testid="stSidebar"] .stExpander {
        margin: 2px 0 !important;
    }
    [data-testid="stSidebar"] .stCaption {
        font-size: 0.75rem !important;
    }
    [data-testid="stSidebar"] .stInfo {
        padding: 0.5rem !important;
        font-size: 0.8rem !important;
    }
    
    /* Fix text area text color - ensure text is black and readable */
    .stTextArea textarea {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Fix text input text color */
    .stTextInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Fix selectbox text color */
    .stSelectbox select {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Fix number input text color */
    .stNumberInput input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Ensure all form elements have readable text */
    .stTextArea, .stTextInput, .stSelectbox, .stNumberInput {
        color: #000000 !important;
    }
    
    /* Fix any white text on white background issues */
    .stTextArea > div > div > textarea,
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Additional comprehensive text color fixes for all form elements */
    .stTextArea textarea,
    .stTextInput input,
    .stSelectbox select,
    .stNumberInput input,
    .stMultiselect select,
    .stSlider input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Target form elements specifically */
    .stForm .stTextArea textarea,
    .stForm .stTextInput input,
    .stForm .stSelectbox select,
    .stForm .stNumberInput input,
    .stForm .stMultiselect select {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Target all input elements with more specific selectors */
    input[type="text"],
    input[type="number"],
    textarea,
    select {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Force text color for all Streamlit form widgets */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] select,
    [data-testid="stNumberInput"] input,
    [data-testid="stMultiselect"] select {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Additional targeting for nested elements */
    .stTextArea > div > div > div > textarea,
    .stTextInput > div > div > div > input,
    .stSelectbox > div > div > div > select,
    .stNumberInput > div > div > div > input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Fix ONLY form inputs that are white on white */
    .stTextArea textarea,
    .stTextInput input,
    .stSelectbox select,
    .stNumberInput input,
    .stMultiselect select {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Don't affect any other text elements */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    .stMarkdown strong, .stMarkdown em, .stMarkdown small, .stMarkdown span, .stMarkdown div {
        /* Let these keep their natural colors */
    }
</style>

<script>
function fixTextColors() {
    // Fix all textarea elements
    document.querySelectorAll('textarea').forEach(function(el) {
        el.style.color = '#000000';
        el.style.backgroundColor = '#ffffff';
    });
    
    // Fix all input elements
    document.querySelectorAll('input[type="text"], input[type="number"]').forEach(function(el) {
        el.style.color = '#000000';
        el.style.backgroundColor = '#ffffff';
    });
    
    // Fix all select elements
    document.querySelectorAll('select').forEach(function(el) {
        el.style.color = '#000000';
        el.style.backgroundColor = '#ffffff';
    });
    
    // Fix only form inputs that are white on white
    document.querySelectorAll('.stTextArea textarea, .stTextInput input, .stSelectbox select, .stNumberInput input, .stMultiselect select').forEach(function(el) {
        el.style.color = '#000000';
        el.style.backgroundColor = '#ffffff';
    });
}

// Run on page load
document.addEventListener('DOMContentLoaded', fixTextColors);

// Run periodically to catch dynamically added elements
setInterval(fixTextColors, 1000);
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'config_data' not in st.session_state:
    st.session_state.config_data = {}
if 'current_config_file' not in st.session_state:
    st.session_state.current_config_file = None
if 'agent_instance' not in st.session_state:
    st.session_state.agent_instance = None
if 'api_client' not in st.session_state:
    if API_CLIENT_AVAILABLE:
        try:
            st.session_state.api_client = create_api_client()
        except Exception as e:
            st.session_state.api_client = None
            st.session_state.api_error = str(e)
    else:
        st.session_state.api_client = None
        st.session_state.api_error = API_IMPORT_ERROR if 'API_IMPORT_ERROR' in globals() else "API client not available"
if 'current_agent_id' not in st.session_state:
    st.session_state.current_agent_id = None
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False

def get_available_models():
    """Get available models for each provider."""
    return {
        "openai": [
            "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"
        ],
        "anthropic": [
            "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", 
            "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-opus-20240229"
        ],
        "google": [
            "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"
        ],
        "groq": [
            "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.1-70b-versatile", 
            "llama-3.1-8b-instant", "llama3-groq-70b-8192-tool-use-preview",
            "llama3-groq-8b-8192-tool-use-preview", "mixtral-8x7b-32768"
        ]
    }

def get_api_key_env_for_provider(provider):
    """Get the correct API key environment variable for each provider."""
    provider_to_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", 
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY"
    }
    return provider_to_env.get(provider.lower(), f"{provider.upper()}_API_KEY")

def get_built_in_tools():
    """Get list of available built-in tools."""
    tool_registry = ToolRegistry()
    return tool_registry.get_built_in_tools()

def validate_config(config_data: Dict[str, Any]) -> tuple[bool, str]:
    """Validate configuration data using API client."""
    try:
        api_client = st.session_state.api_client
        
        # Check if API client is available
        if api_client is None:
            return False, "API client not available. Please ensure the API server is running."
        
        # Check if API client is connected
        if not api_client.is_connected():
            return False, "API server not connected. Please start the API server first."
        
        validation_result = api_client.validate_config(config_data)
        
        if validation_result['valid']:
            return True, validation_result['message']
        else:
            return False, validation_result['message']
    except APIConnectionError as e:
        return False, f"API connection error: {str(e)}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def save_config_to_file(config_data: Dict[str, Any], file_path: str) -> bool:
    """Save configuration to YAML file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")
        return False

def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        return {}

def render_agent_info_form():
    """Render agent information form."""
    st.subheader("Agent Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Agent Name", 
                           value=st.session_state.config_data.get('agent', {}).get('name', ''),
                           help="A descriptive name for your agent")
        
        version = st.text_input("Version", 
                              value=st.session_state.config_data.get('agent', {}).get('version', '1.0.0'),
                              help="Version number for your agent")
    
    with col2:
        description = st.text_area("Description", 
                                 value=st.session_state.config_data.get('agent', {}).get('description', ''),
                                 help="Detailed description of what your agent does",
                                 height=100)
    
    # Update session state
    if 'agent' not in st.session_state.config_data:
        st.session_state.config_data['agent'] = {}
    
    st.session_state.config_data['agent']['name'] = name
    st.session_state.config_data['agent']['description'] = description
    st.session_state.config_data['agent']['version'] = version

def render_llm_config_form():
    """Render LLM configuration form."""
    st.subheader("LLM Configuration")
    
    models = get_available_models()
    
    col1, col2 = st.columns(2)
    
    with col1:
        provider = st.selectbox("LLM Provider", 
                              options=list(models.keys()),
                              index=list(models.keys()).index(st.session_state.config_data.get('llm', {}).get('provider', 'openai')) if st.session_state.config_data.get('llm', {}).get('provider') in models else 0,
                              help="Choose your LLM provider")
        
        # Get models for the selected provider
        provider_models = models[provider]
        current_model = st.session_state.config_data.get('llm', {}).get('model', provider_models[0])
        
        # If the current model is not in the provider's models, reset to first model
        if current_model not in provider_models:
            current_model = provider_models[0]
        
        model = st.selectbox("Model", 
                           options=provider_models,
                           index=provider_models.index(current_model),
                           help="Select the specific model to use")
        
        # Auto-update API key environment variable when provider changes
        default_api_key_env = get_api_key_env_for_provider(provider)
        current_api_key_env = st.session_state.config_data.get('llm', {}).get('api_key_env', default_api_key_env)
        
        # If API key env doesn't match the provider, suggest the correct one
        if current_api_key_env != default_api_key_env and not current_api_key_env.startswith(provider.upper()):
            current_api_key_env = default_api_key_env
        
        api_key_env = st.text_input("API Key Environment Variable", 
                                  value=current_api_key_env,
                                  help=f"Environment variable for {provider.title()} API key (usually {default_api_key_env})")
        
        # Show a helpful note about the expected environment variable
        if api_key_env != default_api_key_env:
            st.info(f"💡 Standard env var for {provider.title()} is: `{default_api_key_env}`")
    
    with col2:
        temperature = st.slider("Temperature", 
                              min_value=0.0, max_value=2.0, 
                              value=st.session_state.config_data.get('llm', {}).get('temperature', 0.7),
                              step=0.1,
                              help="Controls randomness in responses")
        
        max_tokens = st.number_input("Max Tokens", 
                                   min_value=1, max_value=128000,
                                   value=st.session_state.config_data.get('llm', {}).get('max_tokens', 4000),
                                   help="Maximum number of tokens in response")
        
        base_url = st.text_input("Base URL (Optional)", 
                               value=st.session_state.config_data.get('llm', {}).get('base_url', ''),
                               help="Custom base URL for API endpoint")
    
    # Update session state
    if 'llm' not in st.session_state.config_data:
        st.session_state.config_data['llm'] = {}
    
    st.session_state.config_data['llm']['provider'] = provider
    st.session_state.config_data['llm']['model'] = model
    st.session_state.config_data['llm']['temperature'] = temperature
    st.session_state.config_data['llm']['max_tokens'] = max_tokens
    st.session_state.config_data['llm']['api_key_env'] = api_key_env
    if base_url:
        st.session_state.config_data['llm']['base_url'] = base_url

def render_prompts_config_form():
    """Render prompts configuration form."""
    st.subheader("Prompts Configuration")
    
    # System Prompt
    st.write("**System Prompt**")
    system_template = st.text_area("System Prompt Template", 
                                 value=st.session_state.config_data.get('prompts', {}).get('system_prompt', {}).get('template', ''),
                                 height=150,
                                 help="The system prompt that defines the agent's role and behavior")
    
    # Helper function to safely get variables as string
    def get_variables_string(variables):
        if isinstance(variables, dict):
            return ', '.join(variables.keys())
        elif isinstance(variables, list):
            return ', '.join(variables)
        else:
            return ''
    
    system_variables = st.text_input("System Prompt Variables (comma-separated)", 
                                   value=get_variables_string(st.session_state.config_data.get('prompts', {}).get('system_prompt', {}).get('variables', {})),
                                   help="Variables that can be substituted in the system prompt")
    
    # User Prompt
    st.write("**User Prompt**")
    user_template = st.text_area("User Prompt Template", 
                                value=st.session_state.config_data.get('prompts', {}).get('user_prompt', {}).get('template', ''),
                                height=100,
                                help="Template for formatting user inputs")
    
    user_variables = st.text_input("User Prompt Variables (comma-separated)", 
                                 value=get_variables_string(st.session_state.config_data.get('prompts', {}).get('user_prompt', {}).get('variables', {})),
                                 help="Variables that can be substituted in the user prompt")
    
    # Tool Prompt (Optional)
    st.write("**Tool Prompt (Optional)**")
    tool_template = st.text_area("Tool Prompt Template", 
                                value=st.session_state.config_data.get('prompts', {}).get('tool_prompt', {}).get('template', ''),
                                height=100,
                                help="Template for tool usage instructions")
    
    tool_variables = st.text_input("Tool Prompt Variables (comma-separated)", 
                                 value=get_variables_string(st.session_state.config_data.get('prompts', {}).get('tool_prompt', {}).get('variables', {})) if st.session_state.config_data.get('prompts', {}).get('tool_prompt') else '',
                                 help="Variables that can be substituted in the tool prompt")
    
    # Update session state
    if 'prompts' not in st.session_state.config_data:
        st.session_state.config_data['prompts'] = {}
    
    st.session_state.config_data['prompts']['system_prompt'] = {
        'template': system_template,
        'variables': {v.strip(): '' for v in system_variables.split(',') if v.strip()}
    }
    
    st.session_state.config_data['prompts']['user_prompt'] = {
        'template': user_template,
        'variables': {v.strip(): '' for v in user_variables.split(',') if v.strip()}
    }
    
    if tool_template:
        st.session_state.config_data['prompts']['tool_prompt'] = {
            'template': tool_template,
            'variables': {v.strip(): '' for v in tool_variables.split(',') if v.strip()}
        }

def render_tools_config_form():
    """Render tools configuration form."""
    st.subheader("Tools Configuration")
    
    # Built-in Tools
    st.write("**Built-in Tools**")
    available_tools = get_built_in_tools()
    current_built_in = st.session_state.config_data.get('tools', {}).get('built_in', [])
    
    selected_tools = st.multiselect("Select Built-in Tools", 
                                  options=available_tools,
                                  default=[tool for tool in current_built_in if tool in available_tools],
                                  help="Choose from available built-in tools")
    
    # Custom Tools
    st.write("**Custom Tools**")
    
    # Initialize custom tools in session state
    if 'tools' not in st.session_state.config_data:
        st.session_state.config_data['tools'] = {}
    if 'custom' not in st.session_state.config_data['tools']:
        st.session_state.config_data['tools']['custom'] = []
    
    custom_tools = st.session_state.config_data['tools']['custom']
    
    # Add new custom tool
    with st.expander("Add Custom Tool"):
        new_tool_name = st.text_input("Tool Name", key="new_tool_name")
        new_tool_module = st.text_input("Module Path", key="new_tool_module")
        new_tool_class = st.text_input("Class Name", key="new_tool_class")
        new_tool_description = st.text_area("Description", key="new_tool_description")
        new_tool_params = st.text_area("Parameters (JSON format)", 
                                     value="{}",
                                     key="new_tool_params",
                                     help="JSON object with tool parameters")
        
        if st.button("Add Custom Tool"):
            try:
                params = json.loads(new_tool_params) if new_tool_params.strip() else {}
                new_tool = {
                    'name': new_tool_name,
                    'module_path': new_tool_module,
                    'class_name': new_tool_class,
                    'description': new_tool_description,
                    'parameters': params
                }
                custom_tools.append(new_tool)
                st.success(f"Added custom tool: {new_tool_name}")
                st.rerun()
            except json.JSONDecodeError:
                st.error("Invalid JSON format in parameters")
    
    # Display existing custom tools
    if custom_tools:
        st.write("**Existing Custom Tools**")
        for i, tool in enumerate(custom_tools):
            with st.expander(f"Custom Tool: {tool['name']}"):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**Module:** {tool['module_path']}")
                    st.write(f"**Class:** {tool['class_name']}")
                    st.write(f"**Description:** {tool['description']}")
                    if tool.get('parameters'):
                        st.write(f"**Parameters:** {json.dumps(tool['parameters'], indent=2)}")
                with col2:
                    if st.button("Remove", key=f"remove_tool_{i}"):
                        custom_tools.pop(i)
                        st.rerun()
    
    # Update session state
    st.session_state.config_data['tools']['built_in'] = selected_tools
    st.session_state.config_data['tools']['custom'] = custom_tools

def render_memory_config_form():
    """Render memory configuration form."""
    st.subheader("Memory Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        memory_enabled = st.checkbox("Enable Memory", 
                                   value=st.session_state.config_data.get('memory', {}).get('enabled', False),
                                   help="Enable memory functionality for the agent")
        
        if memory_enabled:
            provider = st.selectbox("Memory Provider", 
                                  options=["langmem", "custom"],
                                  index=0 if st.session_state.config_data.get('memory', {}).get('provider', 'langmem') == 'langmem' else 1,
                                  help="Choose memory provider")
            
            st.write("**Memory Types**")
            semantic = st.checkbox("Semantic Memory", 
                                 value=st.session_state.config_data.get('memory', {}).get('types', {}).get('semantic', True),
                                 help="Store facts and knowledge")
            
            episodic = st.checkbox("Episodic Memory", 
                                 value=st.session_state.config_data.get('memory', {}).get('types', {}).get('episodic', True),
                                 help="Store conversation history")
            
            procedural = st.checkbox("Procedural Memory", 
                                   value=st.session_state.config_data.get('memory', {}).get('types', {}).get('procedural', True),
                                   help="Store learned patterns")
    
    with col2:
        if memory_enabled:
            st.write("**Storage Configuration**")
            backend = st.selectbox("Storage Backend", 
                                 options=["memory", "postgres", "redis"],
                                 index=["memory", "postgres", "redis"].index(st.session_state.config_data.get('memory', {}).get('storage', {}).get('backend', 'memory')),
                                 help="Choose storage backend")
            
            connection_string = st.text_input("Connection String (Optional)", 
                                            value=st.session_state.config_data.get('memory', {}).get('storage', {}).get('connection_string', ''),
                                            help="Database connection string if using external storage")
            
            st.write("**Memory Settings**")
            max_memory_size = st.number_input("Max Memory Size", 
                                            min_value=1000, max_value=100000,
                                            value=st.session_state.config_data.get('memory', {}).get('settings', {}).get('max_memory_size', 10000),
                                            help="Maximum number of memory items")
            
            retention_days = st.number_input("Retention Days", 
                                           min_value=1, max_value=365,
                                           value=st.session_state.config_data.get('memory', {}).get('settings', {}).get('retention_days', 30),
                                           help="How long to keep memories")
            
            background_processing = st.checkbox("Background Processing", 
                                              value=st.session_state.config_data.get('memory', {}).get('settings', {}).get('background_processing', True),
                                              help="Enable background memory processing")
    
    # Update session state
    if 'memory' not in st.session_state.config_data:
        st.session_state.config_data['memory'] = {}
    
    st.session_state.config_data['memory']['enabled'] = memory_enabled
    
    if memory_enabled:
        st.session_state.config_data['memory']['provider'] = provider
        st.session_state.config_data['memory']['types'] = {
            'semantic': semantic,
            'episodic': episodic,
            'procedural': procedural
        }
        st.session_state.config_data['memory']['storage'] = {
            'backend': backend
        }
        if connection_string:
            st.session_state.config_data['memory']['storage']['connection_string'] = connection_string
        
        st.session_state.config_data['memory']['settings'] = {
            'max_memory_size': max_memory_size,
            'retention_days': retention_days,
            'background_processing': background_processing
        }

def render_react_config_form():
    """Render ReAct configuration form."""
    st.subheader("ReAct Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_iterations = st.number_input("Max Iterations", 
                                       min_value=1, max_value=100,
                                       value=st.session_state.config_data.get('react', {}).get('max_iterations', 10),
                                       help="Maximum reasoning/acting cycles")
    
    with col2:
        recursion_limit = st.number_input("Recursion Limit", 
                                        min_value=10, max_value=200,
                                        value=st.session_state.config_data.get('react', {}).get('recursion_limit', 50),
                                        help="Maximum recursion depth")
    
    # Update session state
    if 'react' not in st.session_state.config_data:
        st.session_state.config_data['react'] = {}
    
    st.session_state.config_data['react']['max_iterations'] = max_iterations
    st.session_state.config_data['react']['recursion_limit'] = recursion_limit

def render_optimization_config_form():
    """Render optimization configuration form."""
    st.subheader("Optimization Configuration")
    
    optimization_enabled = st.checkbox("Enable Optimization", 
                                     value=st.session_state.config_data.get('optimization', {}).get('enabled', False),
                                     help="Enable optimization features")
    
    if optimization_enabled:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Prompt Optimization**")
            prompt_opt_enabled = st.checkbox("Enable Prompt Optimization", 
                                           value=st.session_state.config_data.get('optimization', {}).get('prompt_optimization', {}).get('enabled', False))
            
            feedback_collection = st.checkbox("Feedback Collection", 
                                            value=st.session_state.config_data.get('optimization', {}).get('prompt_optimization', {}).get('feedback_collection', False))
            
            ab_testing = st.checkbox("A/B Testing", 
                                   value=st.session_state.config_data.get('optimization', {}).get('prompt_optimization', {}).get('ab_testing', False))
            
            optimization_frequency = st.selectbox("Optimization Frequency", 
                                                options=["daily", "weekly", "monthly"],
                                                index=["daily", "weekly", "monthly"].index(st.session_state.config_data.get('optimization', {}).get('prompt_optimization', {}).get('optimization_frequency', 'weekly')))
        
        with col2:
            st.write("**Performance Tracking**")
            perf_tracking_enabled = st.checkbox("Enable Performance Tracking", 
                                              value=st.session_state.config_data.get('optimization', {}).get('performance_tracking', {}).get('enabled', False))
            
            if perf_tracking_enabled:
                available_metrics = ["response_time", "accuracy", "user_satisfaction", "source_quality", "resolution_rate", "customer_satisfaction", "escalation_rate", "code_quality", "execution_success"]
                current_metrics = st.session_state.config_data.get('optimization', {}).get('performance_tracking', {}).get('metrics', ["response_time", "accuracy", "user_satisfaction"])
                
                selected_metrics = st.multiselect("Performance Metrics", 
                                                options=available_metrics,
                                                default=[metric for metric in current_metrics if metric in available_metrics])
    
    # Update session state
    if 'optimization' not in st.session_state.config_data:
        st.session_state.config_data['optimization'] = {}
    
    st.session_state.config_data['optimization']['enabled'] = optimization_enabled
    
    if optimization_enabled:
        st.session_state.config_data['optimization']['prompt_optimization'] = {
            'enabled': prompt_opt_enabled,
            'feedback_collection': feedback_collection,
            'ab_testing': ab_testing,
            'optimization_frequency': optimization_frequency
        }
        
        st.session_state.config_data['optimization']['performance_tracking'] = {
            'enabled': perf_tracking_enabled
        }
        
        if perf_tracking_enabled:
            st.session_state.config_data['optimization']['performance_tracking']['metrics'] = selected_metrics

def render_runtime_config_form():
    """Render runtime configuration form."""
    st.subheader("Runtime Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_iterations = st.number_input("Max Iterations", 
                                       min_value=1, max_value=200,
                                       value=st.session_state.config_data.get('runtime', {}).get('max_iterations', 50),
                                       help="Maximum iterations for agent execution")
        
        timeout_seconds = st.number_input("Timeout (seconds)", 
                                        min_value=10, max_value=3600,
                                        value=st.session_state.config_data.get('runtime', {}).get('timeout_seconds', 300),
                                        help="Maximum execution time in seconds")
    
    with col2:
        retry_attempts = st.number_input("Retry Attempts", 
                                       min_value=0, max_value=10,
                                       value=st.session_state.config_data.get('runtime', {}).get('retry_attempts', 3),
                                       help="Number of retry attempts on failure")
        
        debug_mode = st.checkbox("Debug Mode", 
                               value=st.session_state.config_data.get('runtime', {}).get('debug_mode', False),
                               help="Enable debug mode for detailed logging")
    
    # Update session state
    if 'runtime' not in st.session_state.config_data:
        st.session_state.config_data['runtime'] = {}
    
    st.session_state.config_data['runtime']['max_iterations'] = max_iterations
    st.session_state.config_data['runtime']['timeout_seconds'] = timeout_seconds
    st.session_state.config_data['runtime']['retry_attempts'] = retry_attempts
    st.session_state.config_data['runtime']['debug_mode'] = debug_mode


def render_evaluation_config_form():
    """Render evaluation configuration form."""
    st.subheader("🔬 Evaluation Configuration")
    
    st.info("Configure LangSmith evaluation to measure and improve your agent's performance")
    
    # Main evaluation toggle
    evaluation_enabled = st.checkbox("Enable Evaluation", 
                                   value=st.session_state.config_data.get('evaluation', {}).get('enabled', False),
                                   help="Enable evaluation functionality for performance measurement")
    
    if evaluation_enabled:
        # LangSmith Configuration
        st.write("**🔗 LangSmith Integration**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            langsmith_enabled = st.checkbox("Enable LangSmith", 
                                          value=st.session_state.config_data.get('evaluation', {}).get('langsmith', {}).get('enabled', False),
                                          help="Enable LangSmith integration for advanced evaluation features")
            
            if langsmith_enabled:
                api_key_env = st.text_input("LangSmith API Key Environment Variable", 
                                          value=st.session_state.config_data.get('evaluation', {}).get('langsmith', {}).get('api_key_env', 'LANGSMITH_API_KEY'),
                                          help="Environment variable containing your LangSmith API key")
                
                project_name = st.text_input("Project Name", 
                                            value=st.session_state.config_data.get('evaluation', {}).get('langsmith', {}).get('project_name', ''),
                                            help="LangSmith project name for organizing evaluations")
        
        with col2:
            if langsmith_enabled:
                endpoint = st.text_input("LangSmith Endpoint", 
                                       value=st.session_state.config_data.get('evaluation', {}).get('langsmith', {}).get('endpoint', 'https://api.smith.langchain.com'),
                                       help="LangSmith API endpoint")
                
                tracing = st.checkbox("Enable Tracing", 
                                    value=st.session_state.config_data.get('evaluation', {}).get('langsmith', {}).get('tracing', True),
                                    help="Enable automatic tracing of agent interactions")
        
        st.divider()
        
        # Evaluators Configuration
        st.write("**📊 Evaluators**")
        
        # Initialize evaluators in session state
        if 'evaluation' not in st.session_state.config_data:
            st.session_state.config_data['evaluation'] = {}
        if 'evaluators' not in st.session_state.config_data['evaluation']:
            st.session_state.config_data['evaluation']['evaluators'] = []
        
        evaluators = st.session_state.config_data['evaluation']['evaluators']
        
        # Built-in evaluators quick setup
        st.write("**Quick Setup - Built-in Evaluators**")
        
        quick_col1, quick_col2 = st.columns(2)
        
        with quick_col1:
            if st.button("➕ Add Correctness Evaluator"):
                evaluators.append({
                    'name': 'correctness',
                    'type': 'llm_as_judge',
                    'prompt': 'Evaluate the factual correctness and accuracy of the response.',
                    'model': 'openai:gpt-4o-mini',
                    'enabled': True
                })
                st.rerun()
            
            if st.button("➕ Add Response Time Evaluator"):
                evaluators.append({
                    'name': 'response_time',
                    'type': 'heuristic',
                    'parameters': {'target_time': 5.0, 'max_time': 20.0},
                    'enabled': True
                })
                st.rerun()
        
        with quick_col2:
            if st.button("➕ Add Helpfulness Evaluator"):
                evaluators.append({
                    'name': 'helpfulness',
                    'type': 'llm_as_judge',
                    'prompt': 'Evaluate how helpful and relevant the response is to the user.',
                    'model': 'openai:gpt-4o-mini',
                    'enabled': True
                })
                st.rerun()
            
            if st.button("➕ Add Tool Usage Evaluator"):
                evaluators.append({
                    'name': 'tool_usage',
                    'type': 'heuristic',
                    'parameters': {},
                    'enabled': True
                })
                st.rerun()
        
        # Custom evaluator creation
        with st.expander("➕ Add Custom Evaluator"):
            new_eval_name = st.text_input("Evaluator Name", key="new_eval_name")
            new_eval_type = st.selectbox("Evaluator Type", 
                                       options=["llm_as_judge", "heuristic", "custom"],
                                       key="new_eval_type")
            
            if new_eval_type == "llm_as_judge":
                new_eval_prompt = st.text_area("Evaluation Prompt", 
                                             value="Evaluate the quality of the agent's response.",
                                             key="new_eval_prompt",
                                             help="Prompt to guide the LLM evaluator")
                
                new_eval_model = st.selectbox("Evaluation Model", 
                                            options=["openai:gpt-4o-mini", "openai:gpt-4o", "anthropic:claude-3-sonnet-20240229"],
                                            key="new_eval_model")
            else:
                new_eval_prompt = ""
                new_eval_model = ""
            
            new_eval_params = st.text_area("Parameters (JSON format)", 
                                         value="{}",
                                         key="new_eval_params",
                                         help="JSON object with evaluator parameters")
            
            if st.button("Add Custom Evaluator"):
                try:
                    params = json.loads(new_eval_params) if new_eval_params.strip() else {}
                    new_evaluator = {
                        'name': new_eval_name,
                        'type': new_eval_type,
                        'enabled': True
                    }
                    
                    if new_eval_prompt:
                        new_evaluator['prompt'] = new_eval_prompt
                    if new_eval_model:
                        new_evaluator['model'] = new_eval_model
                    if params:
                        new_evaluator['parameters'] = params
                    
                    evaluators.append(new_evaluator)
                    st.success(f"Added evaluator: {new_eval_name}")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in parameters")
        
        # Display existing evaluators
        if evaluators:
            st.write("**Configured Evaluators**")
            for i, evaluator in enumerate(evaluators):
                with st.expander(f"📊 {evaluator['name']} ({evaluator['type']})"):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**Type:** {evaluator['type']}")
                        if 'prompt' in evaluator:
                            st.write(f"**Prompt:** {evaluator['prompt'][:100]}...")
                        if 'model' in evaluator:
                            st.write(f"**Model:** {evaluator['model']}")
                        if 'parameters' in evaluator:
                            st.write(f"**Parameters:** {json.dumps(evaluator['parameters'], indent=2)}")
                        st.write(f"**Enabled:** {'✅' if evaluator.get('enabled', True) else '❌'}")
                    with col2:
                        if st.button("🗑️ Remove", key=f"remove_eval_{i}"):
                            evaluators.pop(i)
                            st.rerun()
        
        st.divider()
        
        # Evaluation Settings
        st.write("**⚙️ Evaluation Settings**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_evaluate = st.checkbox("Auto-Evaluate", 
                                      value=st.session_state.config_data.get('evaluation', {}).get('auto_evaluate', False),
                                      help="Automatically evaluate each agent response")
            
            evaluation_frequency = st.selectbox("Evaluation Frequency", 
                                              options=["manual", "per_run", "daily", "weekly"],
                                              index=["manual", "per_run", "daily", "weekly"].index(st.session_state.config_data.get('evaluation', {}).get('evaluation_frequency', 'manual')),
                                              help="How often to run evaluations")
            
            batch_size = st.number_input("Batch Size", 
                                       min_value=1, max_value=100,
                                       value=st.session_state.config_data.get('evaluation', {}).get('batch_size', 10),
                                       help="Number of evaluations to run in parallel")
        
        with col2:
            max_concurrency = st.number_input("Max Concurrency", 
                                            min_value=1, max_value=10,
                                            value=st.session_state.config_data.get('evaluation', {}).get('max_concurrency', 2),
                                            help="Maximum concurrent evaluation threads")
            
            # Metrics selection
            available_metrics = ["correctness", "helpfulness", "response_time", "tool_usage", "coherence", "user_satisfaction"]
            current_metrics = st.session_state.config_data.get('evaluation', {}).get('metrics', ["correctness", "helpfulness", "response_time"])
            
            selected_metrics = st.multiselect("Evaluation Metrics", 
                                            options=available_metrics,
                                            default=[metric for metric in current_metrics if metric in available_metrics],
                                            help="Metrics to track and analyze")
        
        st.divider()
        
        # Dataset Configuration
        st.write("**📋 Evaluation Datasets**")
        
        # Initialize datasets in session state
        if 'datasets' not in st.session_state.config_data['evaluation']:
            st.session_state.config_data['evaluation']['datasets'] = []
        
        datasets = st.session_state.config_data['evaluation']['datasets']
        
        # Add dataset
        with st.expander("➕ Add Evaluation Dataset"):
            new_dataset_name = st.text_input("Dataset Name", key="new_dataset_name")
            new_dataset_description = st.text_input("Dataset Description", key="new_dataset_description")
            
            st.write("**Test Examples**")
            new_dataset_examples = st.text_area("Examples (JSON format)", 
                                               value='[{"inputs": {"query": "What is AI?"}, "outputs": {"answer": "Artificial Intelligence"}}]',
                                               key="new_dataset_examples",
                                               help="JSON array of test examples with inputs and expected outputs")
            
            if st.button("Add Dataset"):
                try:
                    examples = json.loads(new_dataset_examples) if new_dataset_examples.strip() else []
                    new_dataset = {
                        'name': new_dataset_name,
                        'description': new_dataset_description,
                        'examples': examples
                    }
                    datasets.append(new_dataset)
                    st.success(f"Added dataset: {new_dataset_name}")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in examples")
        
        # Display existing datasets
        if datasets:
            st.write("**Configured Datasets**")
            for i, dataset in enumerate(datasets):
                with st.expander(f"📋 {dataset['name']}"):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**Description:** {dataset.get('description', 'No description')}")
                        st.write(f"**Examples:** {len(dataset.get('examples', []))}")
                        if dataset.get('examples'):
                            st.write("**Sample Example:**")
                            st.json(dataset['examples'][0])
                    with col2:
                        if st.button("🗑️ Remove", key=f"remove_dataset_{i}"):
                            datasets.pop(i)
                            st.rerun()
        
        # Evaluation status and actions
        st.divider()
        st.write("**🚀 Evaluation Actions**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧪 Test Evaluators", help="Test configured evaluators with sample data"):
                if evaluators:
                    st.write("**Testing Evaluators...**")
                    
                    # Create a temporary config for testing
                    temp_config = st.session_state.config_data.copy()
                    
                    # Ensure evaluation is enabled for testing
                    temp_config['evaluation']['enabled'] = True
                    
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                            yaml.dump(temp_config, f, default_flow_style=False)
                            temp_config_file = f.name
                        
                        from src.core.configurable_agent import ConfigurableAgent
                        
                        # Create test agent
                        test_agent = ConfigurableAgent(temp_config_file)
                        
                        # Test with sample data
                        test_queries = [
                            "What is the capital of France?",
                            "Explain photosynthesis briefly",
                            "What is 2 + 2?"
                        ]
                        
                        for i, query in enumerate(test_queries):
                            st.write(f"**Test {i+1}: {query}**")
                            
                            with st.spinner(f"Running test {i+1}..."):
                                # Run the agent
                                result = test_agent.run(query)
                                
                                # Show response
                                st.write(f"Response: {result['response'][:100]}...")
                                
                                # Manual evaluation if auto-eval is off
                                if 'evaluation' not in result and test_agent.evaluation_manager is not None:
                                    try:
                                        eval_result = test_agent.evaluation_manager.evaluate_single(
                                            input_data={"query": query},
                                            output_data=result
                                        )
                                        result['evaluation'] = eval_result
                                    except Exception as eval_error:
                                        st.warning(f"Evaluation failed: {str(eval_error)}")
                                
                                # Show evaluation results
                                if 'evaluation' in result:
                                    eval_cols = st.columns(len(result['evaluation']))
                                    for j, (eval_name, eval_data) in enumerate(result['evaluation'].items()):
                                        if isinstance(eval_data, dict) and 'score' in eval_data:
                                            with eval_cols[j]:
                                                score = eval_data['score']
                                                color = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                                                st.metric(f"{color} {eval_name}", f"{score:.2f}")
                            
                            st.divider()
                        
                        # Clean up
                        if os.path.exists(temp_config_file):
                            os.unlink(temp_config_file)
                        
                        st.success("✅ Evaluator testing completed!")
                        
                    except Exception as e:
                        st.error(f"Error testing evaluators: {str(e)}")
                        if 'temp_config_file' in locals() and os.path.exists(temp_config_file):
                            os.unlink(temp_config_file)
                else:
                    st.warning("No evaluators configured. Please add evaluators first.")
        
        with col2:
            if st.button("📊 View Metrics", help="View evaluation metrics and analytics"):
                if st.session_state.agent_instance and hasattr(st.session_state.agent_instance, 'evaluation_manager'):
                    try:
                        metrics = st.session_state.agent_instance.get_evaluation_metrics()
                        
                        if metrics.get('total_evaluations', 0) > 0:
                            st.write("**📈 Evaluation Metrics Summary**")
                            
                            # Overall metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Evaluations", metrics.get('total_evaluations', 0))
                            with col2:
                                overall = metrics.get('overall_metrics', {})
                                avg_score = overall.get('average_score', 0)
                                st.metric("Average Score", f"{avg_score:.3f}")
                            with col3:
                                trends = metrics.get('trends', {})
                                direction = trends.get('direction', 'stable')
                                trend_icon = "📈" if direction == "improving" else "📉" if direction == "declining" else "➡️"
                                st.metric("Trend", f"{trend_icon} {direction.title()}")
                            
                            # Per-evaluator metrics
                            evaluator_metrics = metrics.get('evaluator_metrics', {})
                            if evaluator_metrics:
                                st.write("**Per-Evaluator Performance:**")
                                for eval_name, eval_metrics in evaluator_metrics.items():
                                    with st.expander(f"📊 {eval_name.title()}"):
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Count", eval_metrics.get('count', 0))
                                        with col2:
                                            st.metric("Average", f"{eval_metrics.get('average_score', 0):.3f}")
                                        with col3:
                                            st.metric("Min/Max", f"{eval_metrics.get('min_score', 0):.2f}/{eval_metrics.get('max_score', 0):.2f}")
                                        with col4:
                                            st.metric("Std Dev", f"{eval_metrics.get('std_deviation', 0):.3f}")
                                        
                                        # Score distribution
                                        dist = eval_metrics.get('score_distribution', {})
                                        if dist:
                                            st.write("**Score Distribution:**")
                                            for range_key, count in dist.items():
                                                st.write(f"  {range_key}: {count} evaluations")
                        else:
                            st.info("No evaluation metrics available yet. Run some evaluations first!")
                    
                    except Exception as e:
                        st.error(f"Error retrieving metrics: {str(e)}")
                else:
                    st.warning("No agent instance available. Please initialize an agent in the Test tab first.")
        
        with col3:
            if st.button("⚡ Run Batch Eval", help="Run batch evaluation on datasets"):
                # Check if evaluation is enabled in configuration
                evaluation_enabled = st.session_state.config_data.get('evaluation', {}).get('enabled', False)
                if not evaluation_enabled:
                    st.error("❌ Evaluation is not enabled. Please enable evaluation in the Evaluation tab first.")
                    return
                
                if datasets and st.session_state.agent_instance:
                    # Ensure we have the API client in scope
                    api_client = st.session_state.api_client
                    st.write("**Running Batch Evaluation...**")
                    
                    try:
                        # Check if evaluation is enabled in configuration
                        evaluation_enabled = st.session_state.config_data.get('evaluation', {}).get('enabled', False)
                        if not evaluation_enabled:
                            st.error("❌ Evaluation is not enabled. Please enable evaluation in the Evaluation tab first.")
                            return
                        
                        # Check if evaluators are configured
                        evaluators = st.session_state.config_data.get('evaluation', {}).get('evaluators', [])
                        if not evaluators:
                            st.warning("⚠️ No evaluators configured. Please add evaluators in the Evaluation tab.")
                            st.info("💡 You can add built-in evaluators like Correctness, Helpfulness, Response Time, etc.")
                            return
                        
                        # For API-based agents, we'll run evaluations through the API
                        st.info("🔄 Running batch evaluation through API...")
                        
                        # Create test cases from datasets
                        all_test_cases = []
                        for dataset in datasets:
                            for example in dataset.get('examples', []):
                                test_case = {
                                    'input': example.get('inputs', {}).get('query', ''),
                                    'expected_output': example.get('outputs', {}),
                                    'dataset': dataset['name']
                                }
                                if test_case['input']:  # Only add if there's an input
                                    all_test_cases.append(test_case)
                        
                        if not all_test_cases:
                            st.warning("No valid test cases found in datasets. Please check your dataset examples.")
                            return
                        
                        # Run evaluations through API
                        with st.spinner(f"Running batch evaluation on {len(all_test_cases)} test cases..."):
                            results = []
                            successful_cases = 0
                            
                            for i, test_case in enumerate(all_test_cases):
                                try:
                                    # Run agent via API
                                    run_response = api_client.run_agent(
                                        agent_id=st.session_state.current_agent_id,
                                        query=test_case['input'],
                                        include_evaluation=True
                                    )
                                    
                                    result = {
                                        'test_case': test_case,
                                        'agent_result': run_response.get('result', {}),
                                        'evaluation': run_response.get('evaluation', {}),
                                        'success': True
                                    }
                                    results.append(result)
                                    successful_cases += 1
                                    
                                    # Show progress
                                    if (i + 1) % 5 == 0:
                                        st.write(f"Processed {i + 1}/{len(all_test_cases)} test cases...")
                                        
                                except Exception as e:
                                    st.warning(f"Failed to evaluate test case {i + 1}: {str(e)}")
                                    result = {
                                        'test_case': test_case,
                                        'agent_result': {},
                                        'evaluation': {},
                                        'success': False,
                                        'error': str(e)
                                    }
                                    results.append(result)
                        
                        # Calculate summary metrics
                        total_cases = len(all_test_cases)
                        success_rate = successful_cases / total_cases if total_cases > 0 else 0
                        
                        # Calculate evaluation metrics
                        summary_metrics = {}
                        for evaluator in evaluators:
                            eval_name = evaluator.get('name', 'unknown')
                            scores = []
                            
                            for result in results:
                                if result['success'] and 'evaluation' in result['agent_result']:
                                    eval_data = result['agent_result']['evaluation']
                                    if eval_name in eval_data and isinstance(eval_data[eval_name], dict):
                                        score = eval_data[eval_name].get('score', 0)
                                        scores.append(score)
                            
                            if scores:
                                summary_metrics[eval_name] = {
                                    'count': len(scores),
                                    'average': sum(scores) / len(scores),
                                    'min': min(scores),
                                    'max': max(scores),
                                    'scores_above_threshold': {
                                        '0.7': len([s for s in scores if s >= 0.7])
                                    }
                                }
                        
                        batch_result = {
                            'total_cases': total_cases,
                            'successful_cases': successful_cases,
                            'success_rate': success_rate,
                            'duration_seconds': 0,  # We don't track duration for API calls
                            'summary_metrics': summary_metrics,
                            'results': results
                        }
                        
                        st.write("**🎯 Batch Evaluation Results:**")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Cases", batch_result['total_cases'])
                        with col2:
                            st.metric("Successful", batch_result['successful_cases'])
                        with col3:
                            st.metric("Success Rate", f"{batch_result['success_rate']:.1%}")
                        with col4:
                            st.metric("Duration", f"{batch_result['duration_seconds']:.1f}s")
                        
                        # Summary metrics by evaluator
                        summary_metrics = batch_result.get('summary_metrics', {})
                        if summary_metrics:
                            st.write("**📊 Evaluator Performance:**")
                            for eval_name, metrics in summary_metrics.items():
                                with st.expander(f"📈 {eval_name.title()}"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Average Score", f"{metrics.get('average', 0):.3f}")
                                    with col2:
                                        st.metric("Min/Max", f"{metrics.get('min', 0):.2f}/{metrics.get('max', 0):.2f}")
                                    with col3:
                                        high_score_count = metrics.get('scores_above_threshold', {}).get('0.7', 0)
                                        total_count = metrics.get('count', 1)
                                        st.metric("High Quality (≥0.7)", f"{high_score_count}/{total_count}")
                        
                        # Show individual results
                        if batch_result.get('results'):
                            with st.expander(f"📋 Individual Results ({len(batch_result['results'])} cases)"):
                                for i, result in enumerate(batch_result['results'][:10]):  # Show first 10
                                    test_case = result['test_case']
                                    agent_result = result['agent_result']
                                    evaluation = result['evaluation']
                                    
                                    st.write(f"**Case {i+1}:** {test_case['input'][:50]}...")
                                    st.write(f"Dataset: {test_case.get('dataset', 'Unknown')}")
                                    
                                    # Show evaluation scores
                                    eval_cols = st.columns(min(len(evaluation), 4))
                                    for j, (eval_name, eval_data) in enumerate(evaluation.items()):
                                        if isinstance(eval_data, dict) and 'score' in eval_data:
                                            with eval_cols[j % 4]:
                                                score = eval_data['score']
                                                color = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                                                st.write(f"{color} {eval_name}: {score:.2f}")
                                    
                                    st.divider()
                        
                        st.success("✅ Batch evaluation completed!")
                    
                    except Exception as e:
                        st.error(f"Error running batch evaluation: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
                
                elif not datasets:
                    st.warning("No datasets configured. Please add evaluation datasets first.")
                elif not st.session_state.agent_instance:
                    st.warning("No agent instance available. Please initialize an agent in the Test tab first.")
    
    # Update session state
    st.session_state.config_data['evaluation'] = {
        'enabled': evaluation_enabled
    }
    
    if evaluation_enabled:
        # LangSmith config
        langsmith_config = {
            'enabled': langsmith_enabled if 'langsmith_enabled' in locals() else False
        }
        
        if langsmith_enabled:
            langsmith_config.update({
                'api_key_env': api_key_env if 'api_key_env' in locals() else 'LANGSMITH_API_KEY',
                'project_name': project_name if 'project_name' in locals() else '',
                'endpoint': endpoint if 'endpoint' in locals() else 'https://api.smith.langchain.com',
                'tracing': tracing if 'tracing' in locals() else True
            })
        
        st.session_state.config_data['evaluation'].update({
            'langsmith': langsmith_config,
            'evaluators': evaluators,
            'datasets': datasets,
            'auto_evaluate': auto_evaluate if 'auto_evaluate' in locals() else False,
            'evaluation_frequency': evaluation_frequency if 'evaluation_frequency' in locals() else 'manual',
            'batch_size': batch_size if 'batch_size' in locals() else 10,
            'max_concurrency': max_concurrency if 'max_concurrency' in locals() else 2,
            'metrics': selected_metrics if 'selected_metrics' in locals() else ["correctness", "helpfulness", "response_time"]
        })

def render_yaml_preview():
    """Render YAML preview and validation."""
    st.subheader("Configuration Preview")
    
    # Generate YAML
    try:
        yaml_content = yaml.dump(st.session_state.config_data, default_flow_style=False, sort_keys=False, indent=2)
        
        # Validate configuration
        is_valid, validation_message = validate_config(st.session_state.config_data)
        
        if is_valid:
            st.success(validation_message)
        else:
            st.error(validation_message)
        
        # Display YAML
        st.code(yaml_content, language='yaml')
        
        # Download button
        st.download_button(
            label="Download Configuration",
            data=yaml_content,
            file_name=f"agent_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yml",
            mime="text/yaml"
        )
        
        return yaml_content
        
    except Exception as e:
        st.error(f"Error generating YAML: {str(e)}")
        return None

def render_file_operations():
    """Render file load/save operations."""
    st.subheader("File Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Load Configuration**")
        
        st.info("💡 Agent templates are available in the sidebar under 'Quick Templates'")
        
        # Upload custom config
        
        # Upload custom config
        uploaded_file = st.file_uploader("Upload Configuration File", type=['yml', 'yaml'])
        if uploaded_file is not None:
            try:
                config_data = yaml.safe_load(uploaded_file.read())
                st.session_state.config_data = config_data
                st.session_state.current_config_file = uploaded_file.name
                st.success(f"Loaded configuration from {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        st.write("**Save Configuration**")
        
        save_filename = st.text_input("Save as filename", 
                                    value=f"my_agent_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yml")
        
        if st.button("Save Configuration"):
            if st.session_state.config_data:
                # Create configs directory if it doesn't exist
                save_dir = Path("configs/custom")
                save_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = save_dir / save_filename
                if save_config_to_file(st.session_state.config_data, str(save_path)):
                    st.success(f"Configuration saved to {save_path}")
                    st.session_state.current_config_file = str(save_path)
            else:
                st.warning("No configuration to save")

def render_agent_testing():
    """Render agent testing interface using API."""
    st.subheader("Test Agent")
    
    # Check API connection
    api_client = st.session_state.api_client
    
    if api_client is None:
        st.error("⚠️ API client not available. Please ensure the API server is running.")
        st.info("💡 Start the API server with: `python run_api.py`")
        if st.button("🔄 Retry Connection"):
            st.rerun()
        return
    
    if not api_client.is_connected():
        st.error("⚠️ Cannot connect to API server. Please ensure the API server is running.")
        st.info("💡 Start the API server with: `python run_api.py`")
        if st.button("🔄 Retry Connection"):
            st.rerun()
        return
    
    if not st.session_state.config_data:
        st.warning("Please configure your agent first")
        return
    
    # Validate configuration
    is_valid, validation_message = validate_config(st.session_state.config_data)
    if not is_valid:
        st.error(f"❌ Configuration validation error: {validation_message}")
        st.info("💡 Please check your configuration and ensure all required fields are filled.")
        return
    
    # Initialize session storage for last test response (for full-width rendering)
    if 'test_last_response' not in st.session_state:
        st.session_state.test_last_response = None
    
    # Initialize agent via API
    if st.button("Initialize Agent"):
        with st.spinner("Creating agent via API..."):
            try:
                response = api_client.create_agent(st.session_state.config_data)
                st.session_state.current_agent_id = response['agent']['id']
                st.session_state.agent_instance = {
                    'id': response['agent']['id'],
                    'name': response['agent']['name'],
                    'status': response['agent']['status']
                }
                st.success(f"✅ Agent '{response['agent']['name']}' created successfully!")
                st.info(f"🆔 Agent ID: {response['agent']['id']}")
            except APIConnectionError as e:
                st.error(f"❌ API connection error: {str(e)}")
            except APIValidationError as e:
                st.error(f"❌ Configuration validation error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error creating agent: {str(e)}")
    
    # Test interface
    if st.session_state.agent_instance and st.session_state.current_agent_id:
        agent_info = st.session_state.agent_instance
        st.write(f"**🤖 Agent Ready: {agent_info['name']}**")
        st.caption(f"Status: {agent_info['status']} | ID: {agent_info['id']}")
        
        test_input = st.text_area("Enter your test message:", 
                                height=100,
                                placeholder="Ask your agent a question or give it a task...")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("Send Message", type="primary"):
                if test_input:
                    with st.spinner("🤖 Agent is thinking..."):
                        try:
                            # Run agent via API
                            run_response = api_client.run_agent(
                                agent_id=st.session_state.current_agent_id,
                                query=test_input,
                                include_evaluation=True
                            )
                            
                            result = run_response.get('result', {})
                            # Store the response to render full-width below the controls
                            st.session_state.test_last_response = {
                                'run_response': run_response,
                                'result': result
                            }
                            st.info("Response received. See full-width details below.")
                            
                        except AgentNotFoundError:
                            st.error("❌ Agent not found. Please initialize the agent first.")
                        except APIConnectionError as e:
                            st.error(f"❌ API connection error: {str(e)}")
                        except Exception as e:
                            st.error(f"❌ Error running agent: {str(e)}")
                else:
                    st.warning("Please enter a message")
        
        with col2:
            if st.button("🗑️ Delete Agent"):
                try:
                    api_client.delete_agent(st.session_state.current_agent_id)
                    st.session_state.agent_instance = None
                    st.session_state.current_agent_id = None
                    st.session_state.test_last_response = None
                    st.success("✅ Agent deleted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error deleting agent: {str(e)}")
        
        # Render the agent response and details in full width below the controls
        if st.session_state.test_last_response:
            run_response = st.session_state.test_last_response.get('run_response', {})
            result = st.session_state.test_last_response.get('result', {})
            
            st.write("**🤖 Agent Response:**")
            if isinstance(result, dict):
                response_text = result.get('response', 'No response')
            else:
                response_text = str(result)
            st.write(response_text)
            
            # Show execution metrics
            if isinstance(result, dict):
                col_metric1, col_metric2 = st.columns(2)
                with col_metric1:
                    exec_time = result.get('execution_time', 0)
                    st.metric("⏱️ Execution Time", f"{exec_time:.2f}s")
                with col_metric2:
                    token_usage = result.get('token_usage', {})
                    total_tokens = sum(token_usage.values()) if token_usage else 0
                    st.metric("🔤 Tokens Used", total_tokens)
            
            # Show evaluation results if available
            evaluation_results = run_response.get('evaluation', [])
            if evaluation_results:
                st.write("**📊 Evaluation Results:**")
                eval_cols = st.columns(min(len(evaluation_results), 4))
                for i, eval_result in enumerate(evaluation_results):
                    with eval_cols[i % 4]:
                        evaluator = eval_result.get('evaluator', 'Unknown')
                        score = eval_result.get('score', 0)
                        score_color = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                        st.metric(f"{score_color} {evaluator.title()}", f"{score:.2f}")
                
                # Show detailed evaluation in expander
                with st.expander("Detailed Evaluation Results"):
                    for eval_result in evaluation_results:
                        evaluator = eval_result.get('evaluator', 'Unknown')
                        score = eval_result.get('score', 0)
                        details = eval_result.get('details', {})
                        st.write(f"**{evaluator.title()}:**")
                        st.write(f"  Score: {score:.3f}")
                        if details:
                            st.json(details)
            
            # Show additional details in expander
            with st.expander("Full Response Details"):
                st.json(run_response)
    
    # Show existing agents
    with st.expander("🔍 View Existing Agents"):
        try:
            agents_response = api_client.list_agents(page_size=10)
            agents = agents_response.get('agents', [])
            
            if agents:
                st.write(f"**Found {len(agents)} agents:**")
                for agent in agents:
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"**{agent['name']}** ({agent['version']})")
                        if agent.get('description'):
                            st.caption(agent['description'])
                    with col2:
                        status_color = "🟢" if agent['status'] == 'active' else "🟡" if agent['status'] == 'configuring' else "🔴"
                        st.write(f"{status_color} {agent['status']}")
                    with col3:
                        if st.button("Use", key=f"use_{agent['id']}"):
                            st.session_state.current_agent_id = agent['id']
                            st.session_state.agent_instance = agent
                            st.success(f"Selected agent: {agent['name']}")
                            st.rerun()
            else:
                st.info("No agents found. Create an agent to get started.")
        except APIConnectionError:
            st.error("Unable to fetch agents - API connection issue")
        except Exception as e:
            st.error(f"Error fetching agents: {str(e)}")

def render_template_selector():
    """Render compact template selector in sidebar."""
    st.sidebar.subheader("🤖 Agent Templates")
    
    # Define all available templates
    templates = {
        "📋 Template Agent": {
            "file": "template_agent.yml",
            "description": "Comprehensive template with all options"
        },
        "🚀 Minimal Agent": {
            "file": "minimal_agent.yml",
            "description": "Simple template to get started quickly"
        },
        "🔬 Research Agent": {
            "file": "research_agent.yml",
            "description": "Web research and information gathering"
        },
        "💻 Coding Assistant": {
            "file": "coding_assistant.yml", 
            "description": "Code generation and programming help"
        },
        "🎧 Customer Support": {
            "file": "customer_support.yml",
            "description": "Customer service and support agent"
        },
        "🤖 Gemini Agent": {
            "file": "gemini_agent.yml",
            "description": "Google Gemini-powered agent"
        },
        "⚡ Groq Agent": {
            "file": "groq_agent.yml", 
            "description": "High-speed Groq inference agent"
        },
        "🔍 Web Browser Agent": {
            "file": "web_browser_agent.yml",
            "description": "Specialized web search and information gathering"
        },
        "✍️ Writer Agent": {
            "file": "writer_agent.yml",
            "description": "Content creation and writing specialist"
        },
        "🔬 Evaluation Demo Agent": {
            "file": "evaluation_enabled_agent.yml",
            "description": "Agent with comprehensive evaluation capabilities"
        }
    }
    
    # Get available templates
    available_templates = []
    for template_name, template_info in templates.items():
        example_path = Path(f"configs/examples/{template_info['file']}")
        if example_path.exists():
            available_templates.append(template_name)
    
    if available_templates:
        # Template selection dropdown
        selected_template = st.sidebar.selectbox(
            "Choose a template:",
            options=["Select template..."] + available_templates,
            help="Select a pre-configured agent template to load"
        )
        
        # Show description for selected template
        if selected_template != "Select template..." and selected_template in templates:
            st.sidebar.caption(f"📝 {templates[selected_template]['description']}")
        
        # Load button
        col1, col2 = st.sidebar.columns([2, 1])
        with col1:
            load_disabled = selected_template == "Select template..."
            if st.button("Load Template", disabled=load_disabled, use_container_width=True):
                if selected_template in templates:
                    template_info = templates[selected_template]
                    example_path = Path(f"configs/examples/{template_info['file']}")
                    config_data = load_config_from_file(str(example_path))
                    if config_data:
                        st.session_state.config_data = config_data
                        st.session_state.current_config_file = str(example_path)
                        st.sidebar.success(f"✅ Loaded!")
                        st.rerun()
        
        with col2:
            st.sidebar.caption(f"{len(available_templates)} available")
    else:
        st.sidebar.warning("No templates found")


def main():
    """Main Streamlit application."""
    st.title("🤖 Configurable LangGraph Agents")
    st.write("Create and configure AI agents with a user-friendly web interface")
    
    # API Connection Status
    api_client = st.session_state.api_client
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if not API_CLIENT_AVAILABLE:
            st.error("🚫 API Client Unavailable")
            st.session_state.api_connected = False
        elif api_client is None:
            st.error("🚫 API Client Error")
            st.session_state.api_connected = False
        elif api_client.is_connected():
            st.success("🌐 API Connected")
            st.session_state.api_connected = True
        else:
            st.error("🔌 API Disconnected")
            st.session_state.api_connected = False
    
    with col2:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    with col3:
        if not st.session_state.api_connected:
            if not API_CLIENT_AVAILABLE:
                st.info("Install: `pip install httpx`")
            else:
                st.info("Start API: `python run_api.py`")
    
    if not st.session_state.api_connected:
        if not API_CLIENT_AVAILABLE:
            st.error("⚠️ API client dependencies missing. Please install requirements: `pip install -r requirements.txt`")
            if 'api_error' in st.session_state:
                with st.expander("Error Details"):
                    st.code(st.session_state.api_error)
        else:
            st.warning("⚠️ Some features require API connection. Please start the API server to use agent testing and management features.")
        st.divider()
    
    # Sidebar navigation with compact title
    st.sidebar.title("🔧 Configuration")
    
    # Show current configuration status compactly
    if st.session_state.current_config_file:
        st.sidebar.success(f"📄 {Path(st.session_state.current_config_file).name}")
    else:
        st.sidebar.info("No configuration loaded")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "Agent Info", "LLM Config", "Prompts", "Tools", "Memory", 
        "ReAct", "Optimization", "Runtime", "Preview", "Test", "Evaluation"
    ])
    
    with tab1:
        render_agent_info_form()
    
    with tab2:
        render_llm_config_form()
    
    with tab3:
        render_prompts_config_form()
    
    with tab4:
        render_tools_config_form()
    
    with tab5:
        render_memory_config_form()
    
    with tab6:
        render_react_config_form()
    
    with tab7:
        render_optimization_config_form()
    
    with tab8:
        render_runtime_config_form()
    
    with tab9:
        render_yaml_preview()
        st.divider()
        render_file_operations()
    
    with tab10:
        render_agent_testing()
    
    with tab11:
        render_evaluation_config_form()
    
    # Render compact template selector
    render_template_selector()
    
    # Sidebar tools with better organization
    st.sidebar.divider()
    
    # Agent Management - More compact layout
    st.sidebar.subheader("⚙️ Agent Management")
    
    # Two-column layout for actions
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🗑️ Clear", help="Clear all configuration", use_container_width=True):
            st.session_state.config_data = {}
            st.session_state.current_config_file = None
            st.session_state.agent_instance = None
            st.rerun()
    
    with col2:
        if st.button("💾 Save", help="Save current configuration", use_container_width=True):
            if st.session_state.config_data:
                save_filename = f"my_agent_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yml"
                save_dir = Path("configs/custom")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / save_filename
                if save_config_to_file(st.session_state.config_data, str(save_path)):
                    st.sidebar.success(f"✅ Saved!")
            else:
                st.sidebar.warning("No configuration to save")
    
    # Test agent - full width with API awareness
    if st.sidebar.button("🧪 Test Agent", use_container_width=True):
        if not st.session_state.api_connected:
            st.sidebar.error("API not connected! Start with: python run_api.py")
        elif st.session_state.config_data:
            st.sidebar.success("Agent ready! Go to the 'Test' tab.")
        else:
            st.sidebar.warning("No agent configured yet")
    
    # Debug section - collapsible with compact styling
    st.sidebar.divider()
    with st.sidebar.expander("🔧 Debug Info"):
        st.caption(f"Config: {Path(st.session_state.current_config_file).name if st.session_state.current_config_file else 'None'}")
        st.caption(f"Data: {'✅' if st.session_state.config_data else '❌'}")
        st.caption(f"Agent: {'✅' if st.session_state.agent_instance else '❌'}")
        st.caption(f"API: {'✅' if st.session_state.api_connected else '❌'}")
        if st.session_state.current_agent_id:
            st.caption(f"Agent ID: {st.session_state.current_agent_id[:8]}...")
        
        # API health check button
        if st.button("🚑 Check API Health", use_container_width=True):
            try:
                health = st.session_state.api_client.health_check()
                st.success(f"API Status: {health.get('status', 'Unknown')}") 
            except Exception as e:
                st.error(f"API Error: {str(e)[:50]}...")

if __name__ == "__main__":
    main() 