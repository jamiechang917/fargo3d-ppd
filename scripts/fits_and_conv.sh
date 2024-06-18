#!/bin/bash
module purge
module load radmc3d
module load casa

python3 get_fits.py
cd ../rt_output
mkdir -p fits && cd fits
mv ../lines.fits .
casa < ../../scripts/conv_steps
echo "Done!"
