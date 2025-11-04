#!/bin/bash
# Quick test runner for all dev scripts

cd "$(dirname "$0")"

echo "======================================================================"
echo "  HITL Dev Scripts - Quick Test Runner"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test scripts in order
scripts=(
    "test_utils.py"
    "test_serialization.py"
    "test_database.py"
    "test_schema_registry.py"
    "test_artifacts.py"
    "test_full_workflow.py"
)

passed=0
failed=0

for script in "${scripts[@]}"; do
    echo ""
    echo "----------------------------------------------------------------------"
    echo "  Running: $script"
    echo "----------------------------------------------------------------------"
    
    if python "$script"; then
        echo -e "${GREEN}✓ PASSED${NC}: $script"
        ((passed++))
    else
        echo -e "${RED}✗ FAILED${NC}: $script"
        ((failed++))
    fi
done

# Summary
echo ""
echo "======================================================================"
echo "  Summary"
echo "======================================================================"
echo ""
echo "  Tests run:    ${#scripts[@]}"
echo -e "  Passed:       ${GREEN}$passed${NC}"
if [ $failed -gt 0 ]; then
    echo -e "  Failed:       ${RED}$failed${NC}"
else
    echo -e "  Failed:       $failed"
fi
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}✓ ALL DEV SCRIPTS PASSED${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}✗ SOME DEV SCRIPTS FAILED${NC}"
    echo ""
    exit 1
fi
