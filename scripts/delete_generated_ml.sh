#!/bin/bash

REPOS_FOLDER="${1:-repos}" # Default to repos directory if not specified

find "$REPOS_FOLDER" -type f -name "GENAIGREENML*" -delete

echo "Deleted files with prefix 'GENAIGREENML' from $REPOS_FOLDER"
