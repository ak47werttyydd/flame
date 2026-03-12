#!/bin/bash

# Delete all profile_trace directories under exp/

set -e

TARGET=$(find exp/ -type d -name "checkpoint" 2>/dev/null)

if [ -z "$TARGET" ]; then
    echo "No checkpoint directories found under exp/."
    exit 0
fi

echo "The following directories will be deleted:"
echo "$TARGET"
echo

read -p "Confirm? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo "$TARGET" | xargs rm -rf

echo "Done."