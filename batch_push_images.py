import subprocess
import os

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result

def batch_push():
    # 1. Get list of modified/untracked PNGs
    res = run_cmd('git status --porcelain | grep ".png$"')
    lines = res.stdout.strip().split('\n')
    files = [line[3:].strip() for line in lines if line.strip()]
    
    if not files:
        print("No PNG files to push.")
        return

    batch_size = 100
    total = len(files)
    print(f"Found {total} files to process.")

    for i in range(0, total, batch_size):
        batch = files[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} files)")
        
        # Add batch
        add_cmd = f"git add {' '.join(batch)}"
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
    batch_push()
