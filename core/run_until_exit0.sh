#!/bin/bash

while true; do
    python your_script.py

    # Check the exit code of the Python script
    if [ $? -eq 0 ]; then
        echo "Python script exited without errors. Exiting."
        break
    else
        echo "Python script exited with an error. Restarting..."
    fi

    sleep 1
done
