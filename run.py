#!/usr/bin/env python3
"""
Entry point to run the On-The-Fly Adaptive Agent
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from adaptive_agent import main

if __name__ == "__main__":
    main()