#!/bin/bash

# Setup Script for Agent 2: Hybrid Segmentation
# This script downloads necessary repositories and pretrained weights.

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${BASE_DIR}/pretrained_weights"
REPOS_DIR="${BASE_DIR}/repos"

mkdir -p "$WEIGHTS_DIR"
mkdir -p "$REPOS_DIR"

echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}   Agent 2: Hybrid Segmentation Setup (SuPreM + SAM)  ${NC}"
echo -e "${BLUE}======================================================${NC}"
echo ""

# ------------------------------------------------------------------
# 1. Setup SuPreM (AbdomenAtlas 1.1)
# ------------------------------------------------------------------
echo -e "${YELLOW}[1/4] Setting up SuPreM (AbdomenAtlas 1.1)...${NC}"

# 1.1 Clone Repo (Optional, mostly for reference implementation)
if [ ! -d "${REPOS_DIR}/SuPreM" ]; then
    echo "Cloning SuPreM repository..."
    git clone https://github.com/MrGiovanni/SuPreM.git "${REPOS_DIR}/SuPreM"
else
    echo "SuPreM repo already exists."
fi

# 1.2 Download Weights (Swin UNETR)
SUPREM_WEIGHTS="${WEIGHTS_DIR}/supervised_suprem_swinunetr_2100.pth"
if [ ! -f "$SUPREM_WEIGHTS" ]; then
    echo "Downloading SuPreM (Swin UNETR) weights..."
    # Note: Link might need updating if changed by author. Using verified link from analysis.
    wget -O "$SUPREM_WEIGHTS" "https://github.com/MrGiovanni/SuPreM/releases/download/v0.1/supervised_suprem_swinunetr_2100.pth"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}SuPreM weights downloaded successfully.${NC}"
    else
        echo -e "${RED}Failed to download SuPreM weights. Please check internet connection or link.${NC}"
    fi
else
    echo -e "${GREEN}SuPreM weights already exist.${NC}"
fi

# ------------------------------------------------------------------
# 2. Setup SAM-Med3D
# ------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[2/4] Setting up SAM-Med3D...${NC}"

# 2.1 Clone Repo
if [ ! -d "${REPOS_DIR}/SAM-Med3D" ]; then
    echo "Cloning SAM-Med3D repository..."
    git clone https://github.com/uni-medical/SAM-Med3D.git "${REPOS_DIR}/SAM-Med3D"
else
    echo "SAM-Med3D repo already exists."
fi

# 2.2 Download Weights (Turbo)
SAM_WEIGHTS="${WEIGHTS_DIR}/sam_med3d_turbo.pth"
if [ ! -f "$SAM_WEIGHTS" ]; then
    echo "Downloading SAM-Med3D-turbo weights..."
    # Link from HuggingFace
    wget -O "$SAM_WEIGHTS" "https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
    
     if [ $? -eq 0 ]; then
        echo -e "${GREEN}SAM-Med3D-turbo weights downloaded successfully.${NC}"
    else
        echo -e "${RED}Failed to download SAM-Med3D weights.${NC}"
    fi
else
    echo -e "${GREEN}SAM-Med3D-turbo weights already exist.${NC}"
fi

# ------------------------------------------------------------------
# 3. Install Dependencies
# ------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[3/4] Checking Python Dependencies...${NC}"
echo "Installing/Updating required packages: monai, torch, simpleitk..."
pip install monai torch torchvision torchaudio simpleitk nibabel

# ------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------
echo ""
echo -e "${BLUE}======================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "Weights are located in: ${WEIGHTS_DIR}"
echo -e "Repos are located in:   ${REPOS_DIR}"
echo -e "${BLUE}======================================================${NC}"
