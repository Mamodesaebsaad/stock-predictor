#!/usr/bin/env python3
"""
Stock Predictor API Server Runner

This script provides an easy way to start the FastAPI server with optimal settings.
Usage: python run.py
"""

import uvicorn
import os
import sys
from pathlib import Path

def main():
    """Start the FastAPI server with development settings"""
    
    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Check if main.py exists
    if not Path("main.py").exists():
        print("❌ Error: main.py not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    # Check if config.json exists
    if not Path("config.json").exists():
        print("⚠️  Warning: config.json not found")
        print("Creating default config.json...")
        
        default_config = {
            "data_source": "yfinance",
            "custom_api_url": "https://api.example.com/stock/{ticker}"
        }
        
        import json
        with open("config.json", "w") as f:
            json.dump(default_config, f, indent=2)
        
        print("✅ Default config.json created")
    
    print("🚀 Starting Stock Predictor API...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API docs available at: http://localhost:8000/docs")
    print("🔄 Auto-reload enabled for development")
    print("\n" + "="*50)
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["./"],
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
