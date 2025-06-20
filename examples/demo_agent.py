#!/usr/bin/env python3
"""
Demo script for the Adaptive Agent
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from adaptive_agent import AdaptiveAgent

def run_demo():
    """Run a demonstration of the agent's capabilities"""
    print("🤖 Adaptive Agent Demo")
    print("=" * 40)
    
    try:
        agent = AdaptiveAgent(provider_type='auto')
        print("✅ Agent initialized successfully!\n")
        
        # Test cases
        test_cases = [
            ("Greeting", "Hello, how are you?"),
            ("Basic Math", "calculate 15 + 27"),
            ("Complex Math", "solve sqrt(25) * 4 + 10"),
            ("Math with Functions", "calculate sin(pi/2)"),
            ("Help Request", "help"),
            ("Thank You", "thank you for your help"),
        ]
        
        for category, prompt in test_cases:
            print(f"🔤 {category}: {prompt}")
            result = agent.run(prompt)
            print(f"🤖 Response: {result}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    run_demo()