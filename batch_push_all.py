import subprocess
import os
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def batch_push_staged():
    # 1. Get list of staged files
    res = run_cmd('git diff --name-only --cached')
    files = res.stdout.strip().split('\n')
    files = [f.strip() for f in files if f.strip()]
    
    if not files:
        print("No staged files to push.")
        # Try status just in case some aren't staged yet but should be
        res = run_cmd('git status --porcelain')
        lines = res.stdout.strip().split('\n')
        files = [line[3:].strip() for line in lines if line.strip() and (line.startswith(' M') or line.startswith('??') or line.startswith('A '))]
        if not files:
            print("No modified/untracked files found either.")
            return
        print(f"Adding {len(files)} files to staging...")
        run_cmd(f"git add {' '.join(['\"' + f + '\"' for f in files])}")

    batch_size = 50 # Smaller batch size for safety with large files
    total = len(files)
    print(f"Found {total} files to process.")

    # We need to unstage everything first to push in batches
    # or just work with what we have if we want to push in increments.
    # Actually, the best way to batch push is to commit and push increments.
    
    # Let's unstage all first
    run_cmd("git reset")

    for i in range(0, total, batch_size):
        batch = files[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} files)")
        
        # Add batch
        add_cmd = f"git add {' '.join(['\"' + f + '\"' for f in batch])}"
        run_cmd(add_cmd)
        
        # Commit batch
        commit_msg = f"Upload assets batch {i//batch_size + 1}"
        commit_cmd = f'git commit -m "{commit_msg}"'
        run_cmd(commit_cmd)
        
        # Push batch
        push_cmd = "git push origin master"
        push_res = run_cmd(push_cmd)
        
        if push_res.returncode != 0:
            print("Push failed. Manual intervention might be needed.")
            # Optional: break if push fails consistently
            # break

if __name__ == "__main__":
    batch_push_staged()
