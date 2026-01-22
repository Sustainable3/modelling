#!/bin/bash
#SBATCH -J pv_inf
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --time 24:00:00

module load trytonp/apptainer/1.3.0
singularity exec --nv docker://ultralytics/ultralytics:latest python3 yolo_pv_segment_sum.py
