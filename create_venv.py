import os
import sys
import subprocess

# Define the name of the virtual environment folder
VENV_DIR = ".venv"

def get_venv_python():
    """Returns the path to the python executable inside the venv based on OS."""
    if sys.platform == "win32":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")

def is_in_venv():
    """Checks if the script is currently running inside the venv."""
    # sys.prefix changes when a virtual environment is active
    return sys.prefix == os.path.abspath(VENV_DIR)

def setup_and_restart():
    """Creates the venv, installs requirements, and re-runs this script."""
    print("Virtual environment not found or not active.")
    
    # 1. Create the venv if it doesn't exist
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment in '{VENV_DIR}'...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    
    venv_python = get_venv_python()
    
    # 2. Upgrade pip and install required packages
    print("Installing/verifying dependencies...")
    # Replace 'requests' with any packages your script actually needs
    required_packages = ["requests"] 
    
    subprocess.check_call([venv_python, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([venv_python, "-m", "pip", "install"] + required_packages)
    
    # 3. Re-execute this exact script using the venv's Python interpreter
    print("\n--- Restarting script inside the virtual environment ---\n")
    os.execv(venv_python, [venv_python] + sys.argv)

def main():
    """Your actual program logic goes here."""
    print("Success! The script is running safely inside the isolated environment.")
    print(f"Active Python interpreter: {sys.executable}")
    
    # Example using an installed library
    import requests
    response = requests.get("https://github.com")
    print(f"API Test Status Code: {response.status_code}")

if __name__ == "__main__":
    # Check if we are already running inside the venv
    if not is_in_venv():
        setup_and_restart()
    else:
        main()
