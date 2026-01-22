#!/bin/bash
#SBATCH -J d_ortos
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --time 24:00:00

python3 imagery_downloader.py
