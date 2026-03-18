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
        print("No staged files found. Checking for all changed files...")
        res = run_cmd('git status --porcelain')
        lines = res.stdout.strip().split('\n')
        files = [line[3:].strip() for line in lines if line.strip()]
        if not files:
            print("Everything clean.")
            return

    batch_size = 100
    total = len(files)
    print(f"Found {total} files/changes to process.")
    
    # Unstage all to allow controlled batching
    run_cmd("git reset")

    for i in range(0, total, batch_size):
        batch = files[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} files)")
        
        # Add batch (wrapping filenames in quotes to handle spaces)
        quoted_files = ' '.join([f'"{f}"' for f in batch])
        add_cmd = f"git add {quoted_files}"
        run_cmd(add_cmd)
        
        # Commit batch
        commit_msg = f"CFC Update Batch {i//batch_size + 1}"
        commit_cmd = f'git commit -m "{commit_msg}"'
        run_cmd(commit_cmd)
        
        # Push batch
        push_cmd = "git push origin master"
        push_res = run_cmd(push_cmd)
        
        if push_res.returncode != 0:
            print("Push failed. Remote might be rejecting the batch or there's a conflict.")
            # If rejected, try smaller batch or stop
            sys.exit(1)

if __name__ == "__main__":
    batch_push_staged()
