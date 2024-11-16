"""Argument parsing for CogVideoX training."""

import argparse

def get_args():
    """Get default arguments for testing."""
    parser = argparse.ArgumentParser(description="CogVideoX training arguments")
    
    # Add default arguments needed for testing
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--use_8bit_adam", action="store_true")
    
    return parser.parse_args([])  # Return default args for testing
