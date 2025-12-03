#!/bin/bash

ZIP_NAME="hagrid_light.zip"
TARGET_DIR="hagrid_light"

if [ ! -f "$ZIP_NAME" ]; then
    echo "File not found: $ZIP_NAME"
    exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y unzip
fi

mkdir -p "$TARGET_DIR"

echo "Extracting $ZIP_NAME ..."

unzip "$ZIP_NAME" -d "$TARGET_DIR"

echo "Done. Extracted to: $TARGET_DIR"