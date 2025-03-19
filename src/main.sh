#!/bin/bash

# This script will run the EZDiffusionModel and then execute the simulate and recover process.

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Run the EZDiffusionModel script
python3 EZDiffusionModel.py

# Check if the first script was successful
if [ $? -eq 0 ]; then
    echo "EZ Diffusion Model ran successfully!"
else
    echo "Error: EZ Diffusion Model did not run successfully."
    exit 1
fi

# Now, run the simulate and recover process
python3 SimulateAndRecover.py

# Check if the second script was successful
if [ $? -eq 0 ]; then
    echo "Simulation and recovery ran successfully!"
else
    echo "Error: Simulation and recovery did not run successfully."
    exit 1
fi

echo "OK"
