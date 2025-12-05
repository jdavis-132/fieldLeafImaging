#!/bin/bash
#
# Script to process remaining SAM3 embeddings for devices 3-8 sequentially
# This script waits for devices 1 and 2 to complete, then processes the rest
#

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SAM3 Embedding Sequential Processor${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to check if a process is running
check_process() {
    local pattern=$1
    pgrep -f "$pattern" > /dev/null 2>&1
    return $?
}

# Function to wait for a process to complete
wait_for_process() {
    local pattern=$1
    local device_name=$2

    echo -e "${YELLOW}Waiting for $device_name to complete...${NC}"
    while check_process "$pattern"; do
        sleep 30
    done
    echo -e "${GREEN}✓ $device_name completed!${NC}"
}

# Wait for devices 1 and 2 to complete
echo ""
echo -e "${YELLOW}Checking status of currently running processes...${NC}"

if check_process "device1/cropped/.*\.png.*sam3_embeddings_device1.csv"; then
    wait_for_process "device1/cropped/.*\.png.*sam3_embeddings_device1.csv" "Device 1"
else
    echo -e "${GREEN}✓ Device 1 already completed${NC}"
fi

if check_process "device2/cropped/.*\.png.*sam3_embeddings_device2.csv"; then
    wait_for_process "device2/cropped/.*\.png.*sam3_embeddings_device2.csv" "Device 2"
else
    echo -e "${GREEN}✓ Device 2 already completed${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting sequential processing${NC}"
echo -e "${GREEN}========================================${NC}"

# Array of remaining devices
devices=(1 2 3 4 5 6 7 8)

# Process each device sequentially
for device in "${devices[@]}"; do
    echo ""
    echo -e "${YELLOW}[Device $device] Starting extraction...${NC}"

    input_pattern="data/processed/ne2025/device${device}/cropped/*.png"
    output_file="output/sam3_embeddings_device${device}.csv"

    # Check if output file already exists and is non-empty
    if [ -f "$output_file" ] && [ -s "$output_file" ]; then
        echo -e "${YELLOW}[Device $device] Output file already exists. Skipping...${NC}"
        continue
    fi

    # Run the extraction
    echo -e "${YELLOW}[Device $device] Processing: $input_pattern${NC}"
    echo -e "${YELLOW}[Device $device] Output: $output_file${NC}"

    if python src/sam3/extract_embeddings.py "$input_pattern" -o "$output_file"; then
        echo -e "${GREEN}[Device $device] ✓ Completed successfully!${NC}"
    else
        echo -e "${RED}[Device $device] ✗ Failed with error code $?${NC}"
        # Continue with next device instead of exiting
        continue
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All devices processed!${NC}"
echo -e "${GREEN}========================================${NC}"

# Combine all CSV files
echo ""
echo -e "${YELLOW}Combining all device embeddings into final CSV...${NC}"

# Check if all device CSV files exist
all_exist=true
for i in {1..8}; do
    if [ ! -f "output/sam3_embeddings_device${i}.csv" ]; then
        echo -e "${RED}Warning: output/sam3_embeddings_device${i}.csv not found${NC}"
        all_exist=false
    fi
done

if [ "$all_exist" = true ]; then
    # Combine CSVs: take header from first file, then data from all files
    echo -e "${YELLOW}Merging CSV files...${NC}"

    # Get header from first file
    head -n 1 output/sam3_embeddings_device1.csv > output/sam3_embeddings.csv

    # Append data (skip header) from all device files
    for i in {1..8}; do
        tail -n +2 output/sam3_embeddings_device${i}.csv >> output/sam3_embeddings.csv
    done

    # Count total lines
    total_lines=$(wc -l < output/sam3_embeddings.csv)
    total_images=$((total_lines - 1))  # Subtract header

    echo -e "${GREEN}✓ Successfully combined embeddings!${NC}"
    echo -e "${GREEN}  Total images: ${total_images}${NC}"
    echo -e "${GREEN}  Output file: output/sam3_embeddings.csv${NC}"
else
    echo -e "${RED}Cannot combine CSV files - some device outputs are missing${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Processing complete!${NC}"
echo -e "${GREEN}========================================${NC}"
