#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=deeplearning_dynamics       #Set the job name to "JobExample1"
#SBATCH --time=24:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=10G                 #Request 2560MB (2.5GB) per node
#SBATCH --gres=gpu:a100:1
#SBATCH --output=dld.%j      #Send stdout/err to "Example1Out.[jobID]"

#First Executable Line
cd $SCRATCH
cd RL
#module load GCCcore/12.2.0 Python/3.10.8
module load GCC/11.3.0 OpenMPI/4.1.4
module load matplotlib/3.5.2
module load PyTorch/1.12.1-CUDA-11.7.0
source sb3env/bin/activate
cd $SCRATCH/RL_partially_observed/src
# python sac_rendezvous.py
# python sac_attitudedyn.py
# python sac_orbital.py
python run_pretrained_sac_model.py
# python plot_training_loss.py