#!/bin/bash

# Check if commit message is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a commit message"
    echo "Usage: $0 \"your commit message\""
    exit 1
fi

git add . && git commit -m "$1" && git push origin main

# SSH into remote server and execute commands
sshpass -p 'ML4lyfe' ssh redmond@204.15.42.29 << 'EOF'
    cd cogvideox-factory/
    # Kill existing training session if it exists
    tmux kill-session -t training 2>/dev/null || true
    # Create new detached session
    tmux new-session -d -s training
    # Send commands to the session
    tmux send-keys -t training "source myenv/bin/activate" C-m
    tmux send-keys -t training "git stash" C-m
    tmux send-keys -t training "git pull" C-m
    tmux send-keys -t training "chmod +x train_video_inpainting_sft.sh" C-m
    tmux send-keys -t training "./train_video_inpainting_sft.sh" C-m
    echo "Training started in tmux session 'training'. Attach with: tmux attach -t training"
EOF