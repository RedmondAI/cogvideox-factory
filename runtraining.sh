#!/bin/bash

# Check if commit message is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a commit message"
    echo "Usage: $0 \"your commit message\""
    exit 1
fi

# Set commit message
COMMIT_MESSAGE="fix(inpainting): Fix tensor handling in transformer forward pass for video inpainting training:
- Replace text embeddings with zero conditioning tensors
- Fix tensor permutations before/after transformer (B,C,T,H,W <-> B,T,C,H,W)
- Update rotary embeddings to use correct video dimensions
- Ensure consistent tensor formats for loss computation"

git add . && git commit -m "$COMMIT_MESSAGE" && git push origin main
# SSH into remote server and execute commands
sshpass -p 'ML4lyfe' ssh redmond@204.15.42.29 << 'EOF'
    cd cogvideox-factory/
    source myenv/bin/activate
    git stash
    git pull
    chmod +x train_video_inpainting_sft.sh
    cd training
    ./train_video_inpainting_sft.sh "$COMMIT_MESSAGE"
EOF