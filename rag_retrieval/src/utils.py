"""
Utility functions for the RAG retrieval system
"""
import os
import logging
from typing import Optional


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging for a module
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger


def get_project_root() -> str:
    """Get the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def safe_get(dictionary: dict, key: str, default: any = None) -> any:
    """
    Safely get a value from a dictionary
    
    Args:
        dictionary: Dictionary to get value from
        key: Key to look up
        default: Default value if key not found
        
    Returns:
        Value or default
    """
    return dictionary.get(key, default)


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + suffix
