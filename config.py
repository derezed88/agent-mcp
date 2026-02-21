import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_PROMPT_FILE = os.path.join(BASE_DIR, ".system_prompt")
DRIVE_TOKEN_FILE = os.path.join(BASE_DIR, "token.json")
DRIVE_CREDS_FILE = os.path.join(BASE_DIR, "credentials.json")
LLM_MODELS_FILE = os.path.join(BASE_DIR, "llm-models.json")
PLUGINS_ENABLED_FILE = os.path.join(BASE_DIR, "plugins-enabled.json")

# Google Drive
DRIVE_FOLDER_ID = os.getenv("FOLDER_ID")
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]

# Logging (setup early so load_llm_registry can use it)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AISvc] %(levelname)s %(message)s",
)
log = logging.getLogger("AISvc")


def load_default_model():
    """Load default model from llm-models.json."""
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = json.load(f)
        return data.get('default_model', '')
    except (FileNotFoundError, Exception) as e:
        log.warning(f"Could not load default_model from llm-models.json: {e}")
        return ''


def save_default_model(model_key: str) -> bool:
    """Persist default_model to llm-models.json. Returns True on success."""
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = json.load(f)
        data['default_model'] = model_key
        with open(LLM_MODELS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        log.error(f"save_default_model({model_key}): {e}")
        return False


def load_llm_registry():
    """Load enabled models from llm-models.json with API keys from .env."""
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = json.load(f)

        registry = {}
        for name, config in data.get('models', {}).items():
            # Only include enabled models
            if not config.get('enabled', True):
                continue

            # Get API key from environment if specified
            api_key = None
            env_key = config.get('env_key')
            if env_key:
                api_key = os.getenv(env_key)
            elif config.get('host') and 'localhost' not in config.get('host', ''):
                # Local models don't need keys
                api_key = "local-no-key-required"

            registry[name] = {
                "model_id": config.get('model_id'),
                "type": config.get('type'),
                "host": config.get('host'),
                "key": api_key,
                "max_context": config.get('max_context', 50),
                "description": config.get('description', ''),
                "tool_call_available": config.get('tool_call_available', False),
                "llm_call_timeout": config.get('llm_call_timeout', 60),
                "system_prompt_folder": config.get('system_prompt_folder', ''),
            }

        return registry
    except FileNotFoundError:
        log.warning(f"llm-models.json not found, using fallback registry")
        return FALLBACK_LLM_REGISTRY
    except Exception as e:
        log.error(f"Error loading llm-models.json: {e}, using fallback")
        return FALLBACK_LLM_REGISTRY


def save_llm_model_field(model_name: str, field: str, value) -> bool:
    """Persist a single field for a model in llm-models.json. Returns True on success."""
    try:
        with open(LLM_MODELS_FILE, 'r') as f:
            data = json.load(f)
        if model_name not in data.get('models', {}):
            log.error(f"save_llm_model_field: model '{model_name}' not found in JSON")
            return False
        data['models'][model_name][field] = value
        with open(LLM_MODELS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        log.error(f"save_llm_model_field({model_name}, {field}): {e}")
        return False


def load_rate_limits() -> dict:
    """Load rate_limits section from plugins-enabled.json."""
    defaults = {
        "llm_call": {"calls": 3,  "window_seconds": 20, "auto_disable": True},
        "search":   {"calls": 5,  "window_seconds": 10, "auto_disable": False},
        "drive":    {"calls": 10, "window_seconds": 60, "auto_disable": False},
        "db":       {"calls": 20, "window_seconds": 60, "auto_disable": False},
        "system":   {"calls": 0,  "window_seconds": 0,  "auto_disable": False},
    }
    try:
        with open(PLUGINS_ENABLED_FILE, 'r') as f:
            config = json.load(f)
        loaded = config.get('rate_limits', {})
        # Merge with defaults so missing keys always have a value
        for tool_type, def_cfg in defaults.items():
            if tool_type not in loaded:
                loaded[tool_type] = def_cfg
            else:
                for k, v in def_cfg.items():
                    loaded[tool_type].setdefault(k, v)
        return loaded
    except Exception as e:
        log.warning(f"Could not load rate_limits from plugins-enabled.json: {e}, using defaults")
        return defaults


def save_rate_limit(tool_type: str, field: str, value) -> bool:
    """Persist a single rate_limit field in plugins-enabled.json. Returns True on success."""
    try:
        with open(PLUGINS_ENABLED_FILE, 'r') as f:
            data = json.load(f)
        if 'rate_limits' not in data:
            data['rate_limits'] = {}
        if tool_type not in data['rate_limits']:
            data['rate_limits'][tool_type] = {}
        data['rate_limits'][tool_type][field] = value
        with open(PLUGINS_ENABLED_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        log.error(f"save_rate_limit({tool_type}, {field}): {e}")
        return False


# Fallback LLM Registry (used if llm-models.json doesn't exist)
FALLBACK_LLM_REGISTRY = {
    "grok4": {
        "model_id": "grok-4-1-fast-reasoning",
        "type": "OPENAI",
        "host": "https://api.x.ai/v1",
        "key": os.getenv("XAI_API_KEY"),
        "max_context": 50,
        "description": "xAI Grok-4 with fast reasoning",
        "tool_call_available": False,
        "llm_call_timeout": 60,
    },
    "gpt52": {
        "model_id": "gpt-5.2",
        "type": "OPENAI",
        "host": "https://api.openai.com/v1",
        "key": os.getenv("OPENAI_API_KEY"),
        "max_context": 100,
        "description": "OpenAI GPT-5.2",
        "tool_call_available": False,
        "llm_call_timeout": 60,
    },
    "gemini25": {
        "model_id": "gemini-2.5-flash",
        "type": "GEMINI",
        "host": None,
        "key": os.getenv("GEMINI_API_KEY"),
        "max_context": 100,
        "description": "Google Gemini 2.5 Flash",
        "tool_call_available": False,
        "llm_call_timeout": 60,
    },
}

def load_limits() -> dict:
    """Load depth/iteration limits from llm-models.json 'limits' section."""
    try:
        with open(LLM_MODELS_FILE, "r") as f:
            data = json.load(f)
        return data.get("limits", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_limit_field(key: str, value: int) -> bool:
    """Persist a single limits field to llm-models.json."""
    try:
        with open(LLM_MODELS_FILE, "r") as f:
            data = json.load(f)
        data.setdefault("limits", {})[key] = value
        with open(LLM_MODELS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        log.error(f"save_limit_field failed: key={key} value={value} err={e}")
        return False


# Load LLM Registry from JSON (only enabled models)
LLM_REGISTRY = load_llm_registry()

# Load default model from plugins-enabled.json
DEFAULT_MODEL = load_default_model()
MAX_TOOL_ITERATIONS = 10

_limits = load_limits()
MAX_AT_LLM_DEPTH = int(_limits.get("max_at_llm_depth", 1))
MAX_AGENT_CALL_DEPTH = int(_limits.get("max_agent_call_depth", 1))

# Rate limits by tool type — loaded from plugins-enabled.json
RATE_LIMITS = load_rate_limits()

# Lightweight model used for the one-shot topic classifier call.
# Must be a key in LLM_REGISTRY.  Gemini Flash is ideal — fast, cheap, tiny output.
CLASSIFIER_MODEL = "gemini_flash"