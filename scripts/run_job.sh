#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=6000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:2
#We expect that our program should not run langer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=12:00:00

 #your script, in this case: write the hostname and the ids of the chosen gpus.
   hostname
   echo $CUDA_VISIBLE_DEVICES
   source /home/vtx829/.env/bin/activate
   python -m test.test_statenet --train_domains taxi --eval_domains taxi --epochs 100 --train_strict --gpu 1 --elmo --pooled -n test_slurm_taxi > test_slurm_taxi.log 2> test_slurm_taxi.err

