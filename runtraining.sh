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
    tmux
    source myenv/bin/activate
    git stash
    git pull
    chmod +x train_video_inpainting_sft.sh
    ./train_video_inpainting_sft.sh
EOF