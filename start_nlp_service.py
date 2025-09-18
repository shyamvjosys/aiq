#!/usr/bin/env python3
"""
Simple startup script for the Josys NLP Web Interface
"""

import os
import sys
import subprocess

def main():
    print("🚀 Starting Josys NLP Web Interface Service")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('josys_data.db'):
        print("❌ Database not found!")
        print("💡 Make sure you're in the /Users/shyamvasudevan/py/aiq directory")
        print("💡 Or run: cd /Users/shyamvasudevan/py/aiq")
        return
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("💡 Create a .env file with your OpenAI API key:")
        print("   echo 'OPENAI_API_KEY=your_openai_key_here' > .env")
        return
    
    print("✅ Database found")
    print("✅ .env file found")
    print("\n🤖 Starting OpenAI NLP interface...")
    
    try:
        # Start the NLP interface
        subprocess.run([sys.executable, 'nlp_openai_interface.py'])
    except KeyboardInterrupt:
        print("\n👋 NLP service stopped.")
    except Exception as e:
        print(f"❌ Error starting service: {e}")

if __name__ == "__main__":
    main()
