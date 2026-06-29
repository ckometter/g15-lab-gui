import subprocess
import sys
import os

def run_git_cmd(args):
    print(f"Running: git {' '.join(args)}")
    result = subprocess.run(["git"] + args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}")
        return False, result.stdout, result.stderr
    print(result.stdout.strip())
    return True, result.stdout, result.stderr

def main():
    # Make sure we run in the repository directory
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_dir)
    print(f"Working in repository directory: {repo_dir}\n")

    # 1. Check if git is installed and initialized
    if not os.path.exists(".git"):
        print("Error: .git directory not found. Please run this script inside the Git repository.")
        return

    # 2. Check current remote URL
    success, stdout, _ = run_git_cmd(["remote", "-v"])
    if not success:
        return
        
    print("\nSelect the remote protocol you want to use:")
    print("1) HTTPS: https://gitlab.ethz.ch/qpg/g15-qcodes.git (requires GitLab username/password or token)")
    print("2) SSH:   git@gitlab.ethz.ch:qpg/g15-qcodes.git (requires SSH key setup on GitLab)")
    print("3) Keep current configuration")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    if choice == "1":
        run_git_cmd(["remote", "set-url", "origin", "https://gitlab.ethz.ch/qpg/g15-qcodes.git"])
    elif choice == "2":
        run_git_cmd(["remote", "set-url", "origin", "git@gitlab.ethz.ch:qpg/g15-qcodes.git"])
    
    # 3. Git Status check
    print("\nCurrent git status:")
    run_git_cmd(["status"])
    
    confirm = input("\nDo you want to stage all modified/new files and push them? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    # 4. Add files
    success, _, _ = run_git_cmd(["add", "."])
    if not success:
        print("Failed to stage files.")
        return

    # 5. Commit message
    commit_msg = input("\nEnter commit message (or press enter for default): ").strip()
    if not commit_msg:
        commit_msg = "Add Thorlabs PM100D powermeter support, Scalar Map tab, and GUI updates"
        
    success, _, _ = run_git_cmd(["commit", "-m", commit_msg])
    if not success:
        print("Nothing to commit or commit failed.")
    
    # 6. Push to remote
    print("\nPushing to GitLab repository...")
    # Get current branch
    _, stdout, _ = run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"])
    branch = stdout.strip() if stdout.strip() else "main"
    
    # Run interactive push so user can input credentials if needed
    print(f"Pushing to origin {branch}...")
    try:
        subprocess.run(["git", "push", "origin", branch], check=True)
        print("\nSuccessfully uploaded code to GitLab!")
    except subprocess.CalledProcessError as e:
        print(f"\nFailed to push: {e}")

if __name__ == "__main__":
    main()
