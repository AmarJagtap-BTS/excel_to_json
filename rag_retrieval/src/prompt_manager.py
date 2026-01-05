"""
Prompt Manager Module
Handles loading and caching of prompts from prompts.json
"""
import json
import os
from typing import Dict, Any, Optional


class PromptManager:
    """Manages prompt templates and caching"""
    
    _instance: Optional['PromptManager'] = None
    _prompts: Optional[Dict[str, Any]] = None
    
    def __new__(cls):
        """Singleton pattern to ensure prompts are loaded only once"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize prompt manager"""
        if self._prompts is None:
            self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load prompts from JSON file with error handling"""
        try:
            prompts_path = self._get_prompts_path()
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self._prompts = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"prompts.json not found. Please ensure it exists in the rag_retrieval directory."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompts.json: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error loading prompts: {str(e)}")
    
    @staticmethod
    def _get_prompts_path() -> str:
        """Get the path to prompts.json"""
        # Try multiple locations
        current_dir = os.path.dirname(__file__)
        
        # Location 1: ../prompts.json (from src/)
        path1 = os.path.join(current_dir, '..', 'prompts.json')
        if os.path.exists(path1):
            return path1
        
        # Location 2: ./prompts.json (same directory)
        path2 = os.path.join(current_dir, 'prompts.json')
        if os.path.exists(path2):
            return path2
        
        # Default to path1 if neither exists (will raise FileNotFoundError)
        return path1
    
    def get_system_prompt(self) -> str:
        """Build and return the system prompt"""
        if not self._prompts:
            raise RuntimeError("Prompts not loaded")
        sp = self._prompts['system_prompt']
        
        prompt_parts = [
            f"{sp['role']} {sp['job']}",
            "",
            f"CATALOG SCOPE: {sp.get('catalog_scope', '')}",
            "",
            "IMPORTANT INSTRUCTIONS:"
        ]
        
        for inst in sp['important_instructions']:
            prompt_parts.append(f"- {inst}")
        
        prompt_parts.append("")
        prompt_parts.append("Guidelines:")
        
        for guide in sp['guidelines']:
            prompt_parts.append(f"- {guide}")
        
        if 'response_structure' in sp:
            prompt_parts.append("")
            prompt_parts.append("Response Structure:")
            for struct in sp['response_structure']:
                prompt_parts.append(f"- {struct}")
        
        return "\n".join(prompt_parts)
    
    def get_user_prompt(self, query: str, context: str) -> str:
        """Build and return the user prompt"""
        if not self._prompts:
            raise RuntimeError("Prompts not loaded")
        up = self._prompts['user_prompt_template']
        return f"""{up['query_label']}: {query}

{up['context_label']}:
{context}

{up['instruction']}"""
    
    def get_ui_message(self, key: str) -> str:
        """Get a UI message by key"""
        if not self._prompts:
            return ''
        return self._prompts['ui_messages'].get(key, '')
    
    def get_sidebar_examples(self) -> list:
        """Get sidebar example queries"""
        if not self._prompts:
            return []
        return self._prompts['sidebar_examples']
    
    def get_all_prompts(self) -> Dict[str, Any]:
        """Get all prompts (for debugging/inspection)"""
        if not self._prompts:
            return {}
        return self._prompts.copy()
    
    def reload(self) -> None:
        """Reload prompts from file (useful for development)"""
        self._prompts = None
        self._load_prompts()
