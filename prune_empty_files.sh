#!/bin/bash

# Script to remove empty files from a directory and its subdirectories
# Usage: ./prune_empty_files.sh [directory_path]

# Default to current directory if no argument provided
TARGET_DIR="${1:-.}"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

echo "Pruning empty files from: $TARGET_DIR"
echo "Searching for empty files..."

# Find and count empty files first
EMPTY_FILES=$(find "$TARGET_DIR" -type f -empty)
EMPTY_COUNT=$(echo "$EMPTY_FILES" | grep -c '^' 2>/dev/null || echo "0")

if [ "$EMPTY_COUNT" -eq 0 ]; then
    echo "No empty files found!"
    exit 0
fi

echo "Found $EMPTY_COUNT empty files:"
echo "$EMPTY_FILES"
echo

# Ask for confirmation
read -p "Do you want to delete these $EMPTY_COUNT empty files? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled."
    exit 0
fi

# Remove empty files
echo "Removing empty files..."
find "$TARGET_DIR" -type f -empty -delete

# Verify deletion
REMAINING_EMPTY=$(find "$TARGET_DIR" -type f -empty | wc -l)
DELETED_COUNT=$((EMPTY_COUNT - REMAINING_EMPTY))

echo "Done! Deleted $DELETED_COUNT empty files."

if [ "$REMAINING_EMPTY" -gt 0 ]; then
    echo "Warning: $REMAINING_EMPTY empty files could not be deleted."
else
    echo "All empty files successfully removed."
fi