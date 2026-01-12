#!/bin/bash
#SBATCH -J md_tune_2
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s191684@student.pg.edu.pl

module load trytonp/apptainer/1.3.0
singularity exec --nv docker://ultralytics/ultralytics:latest python3 yolo_fine_tune.py
