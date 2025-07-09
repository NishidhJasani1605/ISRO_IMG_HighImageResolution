#!/usr/bin/env python3
import os
import sys
import subprocess

def main():
    """
    Run the Streamlit app with the correct Python path
    """
    # Get the absolute path of the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Make sure the project root is in the Python path
    sys.path.append(project_root)
    
    # Get the path to the app
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    
    # Print information
    print(f"Starting ISRO Satellite Super-Resolution App...")
    print(f"Project root: {project_root}")
    
    # Run the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port=8502"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 