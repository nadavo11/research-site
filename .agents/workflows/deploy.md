---
description: robust push and deploy changes for large asset updates
---
This workflow provides a segment-based strategy to synchronize large repository updates (e.g., thousands of visual assets) with the remote, avoiding the standard `git push` timeouts and memory issues.

### 1. Build Verification
Before deploying, always ensure the latest site reports are generated and consistent with the experiment data.
// turbo
```bash
python3 build_multi_dataset_report.py
```

### 2. Staging and Analysis
Stage all changes and identify the volume of the update.
// turbo
```bash
git add -A && git status
```
> [!NOTE]
> If `git status` shows thousands of new or modified files, **do NOT** run a standard `git push`. Use the batching strategy in Step 3.

### 3. Segmented Batch Push
If the update is large (>500 files or several GBs), execute this segmented push script to synchronize the repository in controlled batches.

// turbo
```python
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
    files = [f.strip() for f in res.stdout.strip().split('\n') if f.strip()]
    
    if not files:
        print("No staged files. Syncing with porcelain state...")
        res = run_cmd('git status --porcelain')
        files = [line[3:].strip() for line in res.stdout.strip().split('\n') if line.strip()]
        if not files:
            print("Everything clean.")
            return

    batch_size = 100
    total = len(files)
    print(f"Found {total} files/changes to process. Using batch size {batch_size}.")
    
    # 2. Unstage to allow controlled batching
    run_cmd("git reset")

    # 3. Process batches
    for i in range(0, total, batch_size):
        batch = files[i:i + batch_size]
        print(f"\n--- Batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} files) ---")
        
        quoted_files = ' '.join([f'"{f}"' for f in batch])
        run_cmd(f"git add {quoted_files}")
        run_cmd(f'git commit -m "Deployment Update: Segment {i//batch_size + 1}"')
        
        push_res = run_cmd("git push origin master")
        if push_res.returncode != 0:
            print("CRITICAL: Push abandoned due to error. Resolve conflicts or connectivity and restart.")
            sys.exit(1)

if __name__ == "__main__":
    batch_push_staged()
```

### 4. Final Verification
Check `git status` one last time to ensure all segments were successfully synchronized.
// turbo
```bash
git status
```
