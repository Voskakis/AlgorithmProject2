#!/bin/sh
set -e

# Optional: run any initialization logic here
echo "Container started. Running indefinitely..."

# Trap SIGTERM and SIGINT so the container stops cleanly
trap "echo 'Shutting down...'; exit 0" TERM INT

# Sleep in a loop forever
while true; do
    sleep 3600 &
    wait $!
done
