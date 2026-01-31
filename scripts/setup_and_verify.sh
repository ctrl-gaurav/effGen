#!/bin/bash

################################################################################
# effGen Master Setup & Verification Script
#
# This script orchestrates the complete installation and verification process
# for the effGen framework with beautiful, smooth animations.
#
# Usage:
#   ./setup_and_verify.sh [OPTIONS]
#
# Options:
#   --skip-install         Skip installation, only verify
#   --skip-verify          Only install, skip verification
#   --install-vllm         Install vLLM for faster inference
#   --download-models      Download popular SLM models
#   --dev                  Install development dependencies
#   --verbose              Verbose verification output
#   --help                 Show this help message
################################################################################

set -e

################################################################################
# Colors and Formatting
################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

BOLD='\033[1m'
DIM='\033[2m'

################################################################################
# Configuration
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for inherited settings from parent script
QUICK_MODE="${EFFGEN_QUICK_MODE:-false}"

# Animation Settings (disabled in quick mode)
if [ "$QUICK_MODE" = true ]; then
    CLEAR_BETWEEN_PHASES=false
    ANIM_DELAY=0
else
    CLEAR_BETWEEN_PHASES=true
    ANIM_DELAY=0.1  # Base animation delay in seconds
fi

SKIP_INSTALL=false
SKIP_VERIFY=false
INSTALL_VLLM=false
DOWNLOAD_MODELS=false
INSTALL_DEV=false
VERBOSE=false

################################################################################
# Helper Functions
################################################################################

# Smooth delay function - skips in quick mode
delay() {
    if [ "$QUICK_MODE" = false ] && [ -n "$1" ]; then
        sleep "$1"
    fi
}

# Short delay for line-by-line animation
line_delay() {
    delay "$ANIM_DELAY"
}

# Medium delay for section transitions
section_delay() {
    delay "$(echo "$ANIM_DELAY * 3" | bc -l 2>/dev/null || echo 0.3)"
}

# Long delay for major transitions
phase_delay() {
    delay "$(echo "$ANIM_DELAY * 5" | bc -l 2>/dev/null || echo 0.5)"
}

print_master_banner() {
    if [ "$CLEAR_BETWEEN_PHASES" = true ]; then
        clear
    fi
    echo ""
    phase_delay
    echo -e "${MAGENTA}${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
    line_delay
    echo -e "${MAGENTA}${BOLD}║${NC}                                                                ${MAGENTA}${BOLD}║${NC}"
    line_delay
    echo -e "${MAGENTA}${BOLD}║${NC}           ${CYAN}${BOLD}🚀  EFFGEN MASTER SETUP  🚀${NC}            ${MAGENTA}${BOLD}║${NC}"
    line_delay
    echo -e "${MAGENTA}${BOLD}║${NC}                                                                ${MAGENTA}${BOLD}║${NC}"
    line_delay
    echo -e "${MAGENTA}${BOLD}║${NC}     ${WHITE}Complete Installation & Verification Pipeline${NC}      ${MAGENTA}${BOLD}║${NC}"
    line_delay
    echo -e "${MAGENTA}${BOLD}║${NC}                                                                ${MAGENTA}${BOLD}║${NC}"
    line_delay
    echo -e "${MAGENTA}${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    section_delay

    # Pipeline visualization
    echo -e "${YELLOW}${BOLD}📋 Pipeline Steps:${NC}"
    line_delay

    if [ "$SKIP_INSTALL" = false ]; then
        echo -e "  ${GREEN}1.${NC} ${CYAN}Installation${NC}    ${DIM}→ Setup framework & dependencies${NC}"
        line_delay
    fi

    if [ "$SKIP_VERIFY" = false ]; then
        echo -e "  ${GREEN}2.${NC} ${CYAN}Verification${NC}   ${DIM}→ Validate installation${NC}"
        line_delay
    fi

    echo -e "  ${GREEN}3.${NC} ${CYAN}Ready to Use${NC}   ${DIM}→ Start building agents!${NC}"
    echo ""
    phase_delay
}

print_phase_transition() {
    local from=$1
    local to=$2

    # In quick mode, just print a simple transition
    if [ "$QUICK_MODE" = true ]; then
        echo ""
        echo -e "${GREEN}✓${NC} ${from} complete"
        echo -e "${CYAN}→${NC} Starting ${to}..."
        echo ""
        return
    fi

    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    phase_delay

    # Success checkmark
    echo -e "  ${GREEN}${BOLD}✓${NC} ${from} Complete"
    section_delay
    echo ""

    # Animated dots
    printf "  ${YELLOW}${BOLD}Preparing next phase"
    for i in {1..5}; do
        printf "."
        delay 0.2
    done
    echo -e "${NC}"
    echo ""
    section_delay

    # Progress bar animation
    printf "  ${CYAN}["
    for i in {1..20}; do
        printf "█"
        delay 0.05
    done
    echo -e "]${NC} Done"
    echo ""
    section_delay

    # Transition message
    echo -e "  ${WHITE}${BOLD}▶${NC} ${GREEN}${from}${NC} → ${CYAN}${to}${NC}"
    echo ""
    section_delay

    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    phase_delay
}

print_phase_header() {
    if [ "$CLEAR_BETWEEN_PHASES" = true ]; then
        clear
    fi
    echo ""
    section_delay
    echo -e "${CYAN}${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}                                                                ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}                      ${WHITE}${BOLD}$1${NC}                      ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}                                                                ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    section_delay
}

print_completion_celebration() {
    if [ "$CLEAR_BETWEEN_PHASES" = true ]; then
        clear
    fi
    echo ""
    section_delay

    # Big celebration box
    echo -e "${GREEN}${BOLD}"
    line_delay
    echo "    ╔═══════════════════════════════════════════════════════════════╗"
    line_delay
    echo "    ║                                                               ║"
    line_delay
    echo "    ║            🎉  SETUP COMPLETE - ALL SYSTEMS GO!  🎉           ║"
    line_delay
    echo "    ║                                                               ║"
    line_delay
    echo "    ╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    section_delay

    # Success message
    echo -e "${GREEN}${BOLD}✨ EffGen Framework is Ready! ✨${NC}"
    echo ""
    line_delay

    # Status indicators
    echo -e "${WHITE}${BOLD}Status:${NC}"
    echo ""
    line_delay
    if [ "$SKIP_INSTALL" = false ]; then
        echo -e "  ${GREEN}✓${NC} Installation    ${DIM}[COMPLETE]${NC}"
        line_delay
    fi
    if [ "$SKIP_VERIFY" = false ]; then
        echo -e "  ${GREEN}✓${NC} Verification    ${DIM}[PASSED]${NC}"
        line_delay
    fi
    echo -e "  ${GREEN}✓${NC} System Ready    ${DIM}[ALL SYSTEMS GO]${NC}"
    echo ""
    section_delay
}

print_error() {
    echo ""
    echo -e "${RED}${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}${BOLD}║${NC}                                                                ${RED}${BOLD}║${NC}"
    echo -e "${RED}${BOLD}║${NC}                     ${WHITE}❌  ERROR  ❌${NC}                     ${RED}${BOLD}║${NC}"
    echo -e "${RED}${BOLD}║${NC}                                                                ${RED}${BOLD}║${NC}"
    echo -e "${RED}${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${RED}$1${NC}"
    echo ""
}

show_help() {
    echo "EffGen Master Setup & Verification Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick                Fast mode (no animations, CI-friendly)"
    echo "  --skip-install         Skip installation, only verify"
    echo "  --skip-verify          Only install, skip verification"
    echo "  --install-vllm         Install vLLM for faster inference"
    echo "  --download-models      Download popular SLM models"
    echo "  --dev                  Install development dependencies"
    echo "  --verbose              Verbose verification output"
    echo "  --help                 Show this help message"
    echo ""
    exit 0
}

################################################################################
# Argument Parsing
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            CLEAR_BETWEEN_PHASES=false
            ANIM_DELAY=0
            shift
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        --install-vllm)
            INSTALL_VLLM=true
            shift
            ;;
        --download-models)
            DOWNLOAD_MODELS=true
            shift
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

################################################################################
# Main Execution
################################################################################

main() {
    # Show master banner
    print_master_banner

    # Build install command
    INSTALL_CMD="$SCRIPT_DIR/install_effgen.sh"
    INSTALL_ARGS=""

    if [ "$INSTALL_VLLM" = true ]; then
        INSTALL_ARGS="$INSTALL_ARGS --install-vllm"
    fi
    if [ "$DOWNLOAD_MODELS" = true ]; then
        INSTALL_ARGS="$INSTALL_ARGS --download-models"
    fi
    if [ "$INSTALL_DEV" = true ]; then
        INSTALL_ARGS="$INSTALL_ARGS --dev"
    fi

    # Build verify command
    VERIFY_CMD="$SCRIPT_DIR/verify.sh"
    VERIFY_ARGS=""

    if [ "$VERBOSE" = true ]; then
        VERIFY_ARGS="$VERIFY_ARGS --verbose"
    fi

    # Phase 1: Installation
    if [ "$SKIP_INSTALL" = false ]; then
        print_phase_header "PHASE 1: INSTALLATION"

        if [ -f "$INSTALL_CMD" ]; then
            bash "$INSTALL_CMD" $INSTALL_ARGS
            INSTALL_EXIT=$?

            if [ $INSTALL_EXIT -ne 0 ]; then
                print_error "Installation failed with exit code $INSTALL_EXIT"
                exit $INSTALL_EXIT
            fi
        else
            print_error "Installation script not found: $INSTALL_CMD"
            exit 1
        fi

        # Transition animation
        if [ "$SKIP_VERIFY" = false ]; then
            print_phase_transition "Installation" "Verification"
        fi
    fi

    # Phase 2: Verification
    if [ "$SKIP_VERIFY" = false ]; then
        print_phase_header "PHASE 2: VERIFICATION"

        if [ -f "$VERIFY_CMD" ]; then
            bash "$VERIFY_CMD" $VERIFY_ARGS
            VERIFY_EXIT=$?

            if [ $VERIFY_EXIT -ne 0 ]; then
                print_error "Verification failed with exit code $VERIFY_EXIT"
                exit $VERIFY_EXIT
            fi
        else
            print_error "Verification script not found: $VERIFY_CMD"
            exit 1
        fi
    fi

    # Final celebration
    print_completion_celebration

    # Quick start guide
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${WHITE}${BOLD}🚀 Quick Start Commands:${NC}"
    echo ""
    echo -e "  ${CYAN}1.${NC} Activate environment:"
    echo -e "     ${DIM}conda activate effgen${NC}"
    echo ""
    echo -e "  ${CYAN}2.${NC} Try an example:"
    echo -e "     ${DIM}python examples/basic_agent.py${NC}"
    echo ""
    echo -e "  ${CYAN}3.${NC} Use the CLI:"
    echo -e "     ${DIM}effgen run \"What is 2+2?\"${NC}"
    echo ""
    echo -e "  ${CYAN}4.${NC} Read the docs:"
    echo -e "     ${DIM}https://effgen.org/docs/${NC}"
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Final message
    echo -e "${GREEN}${BOLD}Happy building with EffGen! 🤖✨${NC}"
    echo ""
}

################################################################################
# Execute
################################################################################

main
