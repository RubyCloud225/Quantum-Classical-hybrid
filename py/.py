import sys
import subprocess
import os

def create_and_install_venv(venv_path="venv", requirements_file="requirements.txt"):
    # Create virtual environment
    subprocess.check_call([sys.executable, "-m", "venv", venv_path])

    # Path to pip inside the venv
    if os.name == "nt":
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
        activate_command = f"{venv_path}\\Scripts\\activate.bat"
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")
        activate_command = f"source {venv_path}/bin/activate"

    print(f"Virtual environment created at '{venv_path}'.")
    print(f"To activate, run:\n{activate_command}")

    # Install dependencies
    if os.path.exists(requirements_file):
        print(f"Installing dependencies from {requirements_file}...")
        subprocess.check_call([pip_path, "install", "-r", requirements_file])
        print("Dependencies installed.")
    else:
        print(f"No requirements.txt found at '{requirements_file}'. Skipping dependency installation.")

if __name__ == "__main__":
    create_and_install_venv()