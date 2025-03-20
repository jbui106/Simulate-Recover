#!/bin/bash

# Run the EZDiffusionModel script
python3 EZDiffusionModel.py

# Check if the first script was successful
if [ $? -eq 0 ]; then
    echo "EZ Diffusion Model ran successfully!"
else
    echo "Error: EZ Diffusion Model did not run successfully."
    exit 1
fi

# Run the SimulateAndRecover script
python3 SimulateAndRecover.py

# Check if the second script was successful
if [ $? -eq 0 ]; then
    echo "Simulation and recovery ran successfully!"
else
    echo "Error: Simulation and recovery did not run successfully."
    exit 1
fi

echo "Finished"
