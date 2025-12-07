"""
Configuration for RepoSpeak
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-5-20250929")  



# Analysis Configuration
SKIP_PATTERNS = ['test', 'tests', 'venv', '__pycache__', '.git', 'node_modules', '.pytest_cache']


