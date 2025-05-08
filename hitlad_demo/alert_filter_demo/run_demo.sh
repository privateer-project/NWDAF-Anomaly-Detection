#!/bin/bash
# Run script for Alert Filter Demo

# Set up colored output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}=== Alert Filter Demo Runner ====${NC}"
echo -e "${CYAN}======================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed.${NC}"
    exit 1
fi

# Function to install dependencies
install_dependencies() {
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install dependencies.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Dependencies installed successfully.${NC}"
}

# Function to set up the demo
setup_demo() {
    echo -e "${YELLOW}Setting up the demo...${NC}"
    python3 setup_demo.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to set up the demo.${NC}"
        exit 1
    fi
    
    # Check if autoencoder model exists
    model_path=$(grep -A 2 "autoencoder:" config.yaml | grep "model_path" | awk '{print $2}' | tr -d '"')
    if [ ! -f "$model_path" ]; then
        echo -e "${RED}Autoencoder model not found at: $model_path${NC}"
        echo -e "${YELLOW}Please ensure you have a trained autoencoder model at this location.${NC}"
        echo -e "${YELLOW}You may need to adjust the model path in config.yaml.${NC}"
        
        read -p "Do you want to continue anyway? (y/n): " continue_anyway
        if [[ $continue_anyway != "y" && $continue_anyway != "Y" ]]; then
            echo -e "${RED}Exiting.${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}Found autoencoder model at: $model_path${NC}"
    fi
    
    echo -e "${GREEN}Demo setup completed successfully.${NC}"
}

# Function to run the demo
run_demo() {
    echo -e "${YELLOW}Running the demo...${NC}"
    echo -e "${BLUE}Choose demo type:${NC}"
    echo -e "1. ${CYAN}Automatic demo${NC} (uses true labels as feedback)"
    echo -e "2. ${CYAN}Interactive demo${NC} (you provide feedback)"
    
    read -p "Enter your choice (1/2): " choice
    
    case $choice in
        1)
            echo -e "${YELLOW}Running automatic demo...${NC}"
            python3 alert_filter_demo.py
            ;;
        2)
            echo -e "${YELLOW}Running interactive demo...${NC}"
            python3 interactive_demo.py
            ;;
        *)
            echo -e "${RED}Invalid choice. Exiting.${NC}"
            exit 1
            ;;
    esac
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Demo execution failed.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Demo completed successfully.${NC}"
}

# Main execution
echo -e "${YELLOW}Do you want to install dependencies? (y/n)${NC}"
read -p "Enter your choice: " install_deps

if [[ $install_deps == "y" || $install_deps == "Y" ]]; then
    install_dependencies
fi

echo -e "${YELLOW}Do you want to set up the demo? (y/n)${NC}"
read -p "Enter your choice: " setup

if [[ $setup == "y" || $setup == "Y" ]]; then
    setup_demo
fi

run_demo

echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}=== Demo Runner Complete ====${NC}"
echo -e "${CYAN}======================================${NC}"
