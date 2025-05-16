#!/bin/bash
# Run script for Alert Filter Demo UI

# Set up colored output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}=== Alert Filter Demo UI Runner ====${NC}"
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
    echo -e "${YELLOW}Installing UI dependencies...${NC}"
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install UI dependencies.${NC}"
        exit 1
    fi
    
    # Install parent directory dependencies
    echo -e "${YELLOW}Installing main project dependencies...${NC}"
    pip3 install -r ../requirements.txt
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install main project dependencies.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}All dependencies installed successfully.${NC}"
}

# Function to check setup
check_setup() {
    echo -e "${YELLOW}Checking setup...${NC}"
    
    # Check if autoencoder model exists
    model_path=$(grep -A 2 "autoencoder_model_path" ../config.yaml | grep "models/autoencoder_model.pt" | awk '{print $2}' | tr -d '"')
    if [ ! -f "../$model_path" ]; then
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
    
    echo -e "${GREEN}Setup check completed successfully.${NC}"
}

# Function to run the UI server
run_server() {
    echo -e "${YELLOW}Starting the UI server...${NC}"
    echo -e "${BLUE}The demo interface will be available at:${NC} ${CYAN}http://localhost:5000${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
    
    python3 app.py
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Server execution failed.${NC}"
        exit 1
    fi
}

# Main execution
echo -e "${YELLOW}Do you want to install dependencies? (y/n)${NC}"
read -p "Enter your choice: " install_deps

if [[ $install_deps == "y" || $install_deps == "Y" ]]; then
    install_dependencies
fi

echo -e "${YELLOW}Do you want to check the setup? (y/n)${NC}"
read -p "Enter your choice: " check

if [[ $check == "y" || $check == "Y" ]]; then
    check_setup
fi

run_server

echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}=== UI Server Stopped ====${NC}"
echo -e "${CYAN}======================================${NC}"
