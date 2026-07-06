import subprocess
import sys
import os
import shutil
import stat

def run_git_cmd(args, cwd=None):
    print(f"Running: git {' '.join(args)}")
    result = subprocess.run(["git"] + args, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}")
        return False, result.stdout, result.stderr
    print(result.stdout.strip())
    return True, result.stdout, result.stderr

def safe_rmtree(path):
    """
    Sicherer rmtree, der auch schreibgeschuetzte (read-only) Dateien
    unter Windows loeschen kann (sehr haeufig bei .git Ordnern).
    """
    def remove_readonly(func, path_to_remove, exc_info):
        try:
            os.chmod(path_to_remove, stat.S_IWRITE)
            func(path_to_remove)
        except Exception as e:
            # Falls es immer noch blockiert ist (z.B. durch OneDrive oder einen Editor)
            pass

    if os.path.exists(path):
        try:
            # Python 3.12+ verwendet onexc
            shutil.rmtree(path, onexc=remove_readonly)
        except TypeError:
            # Aeltere Python-Versionen verwenden onerror
            shutil.rmtree(path, onerror=remove_readonly)

def main():
    # Make sure we run in the directory where the script is located (Semesterarbeit)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working in directory: {script_dir}\n")

    repo_name = "g15-qcodes"
    repo_path = os.path.join(script_dir, repo_name)
    
    https_url = "https://gitlab.ethz.ch/qpg/g15-qcodes.git"
    ssh_url = "git@gitlab.ethz.ch:qpg/g15-qcodes.git"
    
    # 1. Ask for protocol choice
    print("Select the remote protocol you want to use for downloading:")
    print("1) HTTPS: https://gitlab.ethz.ch/qpg/g15-qcodes.git (requires GitLab username/password or token)")
    print("2) SSH:   git@gitlab.ethz.ch:qpg/g15-qcodes.git (requires SSH key setup on GitLab)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    remote_url = ssh_url if choice == "2" else https_url
    
    # 2. Check if the directory already exists
    if os.path.exists(repo_path):
        print(f"\nThe folder '{repo_name}' already exists.")
        print("What would you like to do?")
        print("1) Update existing files (git pull)")
        print("2) Completely replace folder (delete and fresh git clone)")
        print("3) Cancel")
        
        action = input("Enter choice (1, 2, or 3): ").strip()
        if action == "3":
            print("Aborted.")
            return
            
        elif action == "1":
            # Check if it is a Git repo
            if not os.path.exists(os.path.join(repo_path, ".git")):
                print(f"Error: '{repo_name}' is not a Git repository (no .git folder).")
                confirm = input("Would you like to delete the folder and clone it from scratch? (y/n): ").strip().lower()
                if confirm == 'y':
                    action = "2"
                else:
                    return
            
            if action == "1":
                # Update remote URL in case it changed
                run_git_cmd(["remote", "set-url", "origin", remote_url], cwd=repo_path)
                
                # Run git pull
                print(f"\nPulling latest changes from GitLab into '{repo_name}'...")
                try:
                    subprocess.run(["git", "pull", "origin"], cwd=repo_path, check=True)
                    print("\nSuccessfully updated code from GitLab!")
                except subprocess.CalledProcessError as e:
                    print(f"\nFailed to pull: {e}")
                return
                
        if action == "2":
            confirm = input(f"Are you sure you want to delete '{repo_path}' and all its contents? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Aborted.")
                return
            print(f"Deleting '{repo_path}'...")
            safe_rmtree(repo_path)
            
            # Double check if directory was removed (sometimes OneDrive/Explorer locks it momentarily)
            if os.path.exists(repo_path):
                print(f"Warning: The folder could not be deleted automatically because a process (e.g. OneDrive, VS Code or Explorer) is blocking it.")
                print(f"Please delete the folder '{repo_path}' manually and run this script again.")
                return

    # 3. Clone repository
    print(f"\nCloning repository to '{repo_name}'...")
    try:
        subprocess.run(["git", "clone", remote_url, repo_name], check=True)
        print("\nSuccessfully downloaded code from GitLab!")
    except subprocess.CalledProcessError as e:
        print(f"\nFailed to clone: {e}")

if __name__ == "__main__":
    main()
