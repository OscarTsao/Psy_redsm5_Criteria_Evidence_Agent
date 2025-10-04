#!/bin/bash
# Git workflow helper script for DSM-5 BERT classification project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAIN_BRANCH="spanbert-criteria-classification"
TRAINING_OUTPUT_DIR="outputs/training"
EXPERIMENT_PREFIX="experiment"

function print_usage() {
    echo "Usage: $0 {new-experiment|commit|push|merge|clean-outputs|status}"
    echo ""
    echo "Commands:"
    echo "  new-experiment <name>  - Create new experiment branch from main"
    echo "  commit <message>       - Add, commit, and tag with experiment info"
    echo "  push                   - Push current branch to remote"
    echo "  merge                  - Merge current experiment to main (after review)"
    echo "  clean-outputs          - Clean old training outputs (keep last 5)"
    echo "  status                 - Show current git and project status"
    echo ""
    echo "Examples:"
    echo "  $0 new-experiment focal-loss-optimization"
    echo "  $0 commit 'Implement adaptive focal loss with early stopping'"
    echo "  $0 push"
    echo "  $0 clean-outputs"
}

function check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo -e "${RED}Error: Not in a git repository${NC}"
        exit 1
    fi
}

function get_current_branch() {
    git branch --show-current
}

function new_experiment() {
    local experiment_name="$1"
    if [[ -z "$experiment_name" ]]; then
        echo -e "${RED}Error: Experiment name is required${NC}"
        print_usage
        exit 1
    fi

    local branch_name="${EXPERIMENT_PREFIX}/${experiment_name}"

    echo -e "${BLUE}Creating new experiment branch: ${branch_name}${NC}"

    # Ensure we're on main branch and up to date
    git checkout "$MAIN_BRANCH"
    git pull origin "$MAIN_BRANCH" 2>/dev/null || echo "Warning: Could not pull from remote"

    # Create and checkout new branch
    git checkout -b "$branch_name"

    echo -e "${GREEN}✓ Created and switched to branch: ${branch_name}${NC}"
    echo -e "${YELLOW}Remember to:${NC}"
    echo "  1. Make your experimental changes"
    echo "  2. Run experiments and validate results"
    echo "  3. Use './scripts/git_workflow.sh commit \"message\"' to commit"
    echo "  4. Use './scripts/git_workflow.sh push' to push to remote"
}

function commit_changes() {
    local commit_message="$1"
    if [[ -z "$commit_message" ]]; then
        echo -e "${RED}Error: Commit message is required${NC}"
        print_usage
        exit 1
    fi

    local current_branch=$(get_current_branch)
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo -e "${BLUE}Committing changes on branch: ${current_branch}${NC}"

    # Add all tracked and relevant untracked files
    git add -A

    # Create commit with additional metadata
    local full_message="${commit_message}

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
Branch: ${current_branch}
Timestamp: ${timestamp}"

    git commit -m "$full_message"

    echo -e "${GREEN}✓ Changes committed successfully${NC}"
    echo -e "${YELLOW}Next steps:${NC}"
    echo "  - Run tests: python test_setup.py"
    echo "  - Run training: python train.py"
    echo "  - Push changes: ./scripts/git_workflow.sh push"
}

function push_branch() {
    local current_branch=$(get_current_branch)

    echo -e "${BLUE}Pushing branch: ${current_branch}${NC}"

    # Set upstream and push
    git push -u origin "$current_branch"

    echo -e "${GREEN}✓ Branch pushed successfully${NC}"
    echo -e "${YELLOW}You can now create a PR at:${NC}"
    echo "https://github.com/your-repo/compare/${MAIN_BRANCH}...${current_branch}"
}

function merge_to_main() {
    local current_branch=$(get_current_branch)

    if [[ "$current_branch" == "$MAIN_BRANCH" ]]; then
        echo -e "${RED}Error: Already on main branch${NC}"
        exit 1
    fi

    echo -e "${BLUE}Merging ${current_branch} to ${MAIN_BRANCH}${NC}"

    # Switch to main and update
    git checkout "$MAIN_BRANCH"
    git pull origin "$MAIN_BRANCH" 2>/dev/null || echo "Warning: Could not pull from remote"

    # Merge the experiment branch
    git merge "$current_branch" --no-ff -m "Merge experiment: $current_branch"

    echo -e "${GREEN}✓ Successfully merged to main${NC}"
    echo -e "${YELLOW}Recommended next steps:${NC}"
    echo "  - Push main: git push origin $MAIN_BRANCH"
    echo "  - Delete experiment branch: git branch -d $current_branch"
    echo "  - Delete remote branch: git push origin --delete $current_branch"
}

function clean_outputs() {
    echo -e "${BLUE}Cleaning old training outputs...${NC}"

    if [[ ! -d "$TRAINING_OUTPUT_DIR" ]]; then
        echo -e "${YELLOW}No training outputs directory found${NC}"
        return
    fi

    # Keep only the 5 most recent runs
    local runs=$(find "$TRAINING_OUTPUT_DIR" -maxdepth 1 -type d -name "20*" | sort -r)
    local count=0

    while IFS= read -r run; do
        count=$((count + 1))
        if [[ $count -gt 5 ]]; then
            echo "Removing old run: $(basename "$run")"
            rm -rf "$run"
        fi
    done <<< "$runs"

    local optimization_dir="outputs/optimization"
    if [[ -d "$optimization_dir" ]]; then
        echo -e "${BLUE}Cleaning old optimization studies...${NC}"
        local opt_runs=$(find "$optimization_dir" -maxdepth 1 -type d -name "20*" | sort -r)
        local opt_count=0
        while IFS= read -r run; do
            opt_count=$((opt_count + 1))
            if [[ $opt_count -gt 5 ]]; then
                echo "Removing old study: $(basename "$run")"
                rm -rf "$run"
            fi
        done <<< "$opt_runs"
    fi

    echo -e "${GREEN}✓ Cleanup completed${NC}"
}

function show_status() {
    local current_branch=$(get_current_branch)

    echo -e "${BLUE}=== Project Status ===${NC}"
    echo -e "${YELLOW}Current branch:${NC} $current_branch"
    echo -e "${YELLOW}Git status:${NC}"
    git status --short

    echo ""
    echo -e "${YELLOW}Recent commits:${NC}"
    git log --oneline -5

    echo ""
    echo -e "${YELLOW}Training outputs:${NC}"
    if [[ -d "$TRAINING_OUTPUT_DIR" ]]; then
        ls -la "$TRAINING_OUTPUT_DIR" | head -10
    else
        echo "No training outputs found"
    fi
}

# Main script logic
check_git_repo

case "$1" in
    "new-experiment")
        new_experiment "$2"
        ;;
    "commit")
        commit_changes "$2"
        ;;
    "push")
        push_branch
        ;;
    "merge")
        merge_to_main
        ;;
    "clean-outputs")
        clean_outputs
        ;;
    "status")
        show_status
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
