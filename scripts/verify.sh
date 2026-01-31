#!/bin/bash

################################################################################
# effGen Verification Script
#
# This script verifies that effGen framework is properly installed and
# functional. It performs comprehensive checks on all components with beautiful
# CLI output.
#
# Usage:
#   ./setup/verify.sh [OPTIONS]
#
# Options:
#   --verbose              Show detailed verification output
#   --skip-optional        Skip optional dependency checks
#   --quick                Run only essential verifications
#   --help                 Show this help message
#
# Requirements:
#   - effGen must be installed (pip install -e .)
################################################################################

set -e  # Exit on error

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
NC='\033[0m' # No Color

BOLD='\033[1m'
DIM='\033[2m'
UNDERLINE='\033[4m'

################################################################################
# Script Configuration
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check for inherited settings from parent script
QUICK_MODE="${EFFGEN_QUICK_MODE:-false}"

# Animation delay (0 in quick mode)
if [ "$QUICK_MODE" = true ]; then
    ANIM_DELAY=0
else
    ANIM_DELAY=0.1
fi

VERBOSE=false
SKIP_OPTIONAL=false

################################################################################
# Helper Functions
################################################################################

# Smooth delay function - skips in quick mode
delay() {
    if [ "$QUICK_MODE" = false ] && [ -n "$1" ]; then
        sleep "$1"
    fi
}

line_delay() { delay "$ANIM_DELAY"; }
section_delay() { delay "$(echo "$ANIM_DELAY * 3" | bc -l 2>/dev/null || echo 0.3)"; }
phase_delay() { delay "$(echo "$ANIM_DELAY * 5" | bc -l 2>/dev/null || echo 0.5)"; }

print_banner() {
    if [ "$QUICK_MODE" = false ]; then
        clear
    fi
    echo ""
    section_delay
    echo -e "${CYAN}${BOLD}╔════════════════════════════════════════════════════════════════╗${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}                                                                ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}             ${MAGENTA}${BOLD}🔍  EFFGEN VERIFIER  🔍${NC}             ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}                                                                ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}         ${WHITE}Comprehensive Framework Validation Suite${NC}         ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}║${NC}                                                                ${CYAN}${BOLD}║${NC}"
    line_delay
    echo -e "${CYAN}${BOLD}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    line_delay
    echo -e "${YELLOW}${BOLD}🎯 Verification Tests:${NC}"
    line_delay
    echo -e "  ${GREEN}▸${NC} Module Imports"
    line_delay
    echo -e "  ${GREEN}▸${NC} Core Components"
    line_delay
    echo -e "  ${GREEN}▸${NC} Tool System"
    line_delay
    echo -e "  ${GREEN}▸${NC} Model Loading"
    echo ""
    section_delay
}

print_header() {
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_subheader() {
    echo ""
    echo -e "${MAGENTA}${BOLD}▶ $1${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}${BOLD}✓${NC} ${GREEN}$1${NC}"
}

print_success_big() {
    echo ""
    echo -e "${GREEN}${BOLD}"
    echo "    ╔═══════════════════════════════════════════════════════════╗"
    echo "    ║                                                           ║"
    echo "    ║              ${WHITE}✨  $1  ✨${GREEN}              ║"
    echo "    ║                                                           ║"
    echo "    ╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

print_error() {
    echo -e "${RED}${BOLD}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}${BOLD}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}${BOLD}ℹ${NC} $1"
}

print_progress() {
    echo -e "${CYAN}  ▸${NC} $1"
}

show_spinner() {
    local pid=$1
    local message=$2
    local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'

    while kill -0 $pid 2>/dev/null; do
        local temp=${spinstr#?}
        printf " ${CYAN}${BOLD}[%c]${NC} %s" "$spinstr" "$message"
        local spinstr=$temp${spinstr%"$temp"}
        sleep 0.1
        printf "\r"
    done
    printf "    \r"
}

display_summary() {
    local passed=$1
    local failed=$2
    local warnings=$3
    local total=$((passed + failed + warnings))

    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${WHITE}${BOLD}                      VERIFICATION SUMMARY${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${GREEN}${BOLD}Passed:${NC}    ${passed}/${total} tests"
    echo -e "  ${RED}${BOLD}Failed:${NC}    ${failed}/${total} tests"
    echo -e "  ${YELLOW}${BOLD}Warnings:${NC}  ${warnings}/${total} tests"
    echo ""

    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}${BOLD}✓ ALL VERIFICATIONS PASSED!${NC}"
        echo ""
        echo -e "${GREEN}effGen framework is properly installed and functional.${NC}"
    else
        echo -e "${RED}${BOLD}✗ SOME VERIFICATIONS FAILED!${NC}"
        echo ""
        echo -e "${YELLOW}Please review the errors above and fix them.${NC}"
    fi
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

################################################################################
# Argument Parsing
################################################################################

show_help() {
    echo "effGen Verification Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --verbose           Show detailed verification output"
    echo "  --skip-optional     Skip optional dependency checks"
    echo "  --quick             Run only essential verifications"
    echo "  --help              Show this help message"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --skip-optional)
            SKIP_OPTIONAL=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            ANIM_DELAY=0
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

################################################################################
# Main Verification Process
################################################################################

main() {
    if [ "$QUICK_MODE" = false ]; then
        clear
    fi
    print_banner

    # Check if we're in the correct directory
    print_header "📍 Checking Environment"

    cd "$PROJECT_ROOT"
    print_success "Project root: $PROJECT_ROOT"

    # Activate conda environment if it exists
    print_subheader "🐍 Verifying Python Environment"

    if command -v conda &> /dev/null; then
        # Check if effgen environment exists
        if conda env list | grep -q "^effgen "; then
            print_progress "Activating conda environment 'effgen'..."
            eval "$(conda shell.bash hook)"
            conda activate effgen
            print_success "Conda environment 'effgen' activated"
        else
            print_info "Conda environment 'effgen' not found, using system Python"
        fi
    fi

    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION detected"

    # Check if effGen is importable
    print_progress "Checking effGen installation..."

    if ! python3 -c "import effgen" &> /dev/null; then
        print_error "effGen not importable"
        echo ""
        print_info "Make sure you've activated the conda environment:"
        print_info "  conda activate effgen"
        print_info "Or install with: pip install -e ."
        exit 1
    fi

    print_success "effGen is importable"

    # Build verification command
    print_header "🔍 Running Comprehensive Verification"

    VERIFY_CMD="python3 $PROJECT_ROOT/scripts/verify.py"

    if [ "$VERBOSE" = true ]; then
        VERIFY_CMD="$VERIFY_CMD --verbose"
    fi

    if [ "$SKIP_OPTIONAL" = true ]; then
        VERIFY_CMD="$VERIFY_CMD --skip-optional"
    fi

    echo ""
    print_info "Executing: $VERIFY_CMD"
    echo ""

    # Run verification script
    if $VERIFY_CMD; then
        VERIFY_EXIT_CODE=0
    else
        VERIFY_EXIT_CODE=$?
    fi

    # Parse results from verify.py output (it prints summary at the end)
    echo ""

    if [ $VERIFY_EXIT_CODE -eq 0 ]; then
        print_header "🎉 Verification Complete"
        print_success "All verifications passed successfully!"
        echo ""
        echo -e "${GREEN}${BOLD}effGen is ready to use!${NC}"
        echo ""
        print_info "Quick start guide:"
        echo -e "  ${CYAN}1.${NC} Check the documentation: ${BLUE}https://effgen.org/docs/${NC}"
        echo -e "  ${CYAN}2.${NC} Explore examples: ${DIM}cd examples${NC}"
        echo -e "  ${CYAN}3.${NC} Run your first agent: ${DIM}python examples/basic_agent.py${NC}"
        echo ""
    else
        print_header "⚠ Verification Issues Detected"
        print_warning "Some checks failed or returned warnings"
        echo ""
        print_info "For detailed information, run with --verbose flag"
        echo -e "  ${DIM}./setup/verify.sh --verbose${NC}"
        echo ""
    fi

    exit $VERIFY_EXIT_CODE
}

################################################################################
# Execute Main Function
################################################################################

main
