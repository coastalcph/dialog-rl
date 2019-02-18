#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=6000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:titanx:1
#We expect that our program should not run langer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=12:00:00

#your script, in this case: write the hostname and the ids of the chosen gpus.


hostname
echo $CUDA_VISIBLE_DEVICES

source /home/grn762/projects/dialog-rl/dialog_env/bin/activate

pretrain=(taxi)
gpu=1

echo "=============================="
echo "  Pretrain:" ${pretrain[@]};
python -m run --train_domains ${pretrain[@]} --train_strict --eval_domains ${pretrain[@]} --gpu $gpu -n pretrain-$pretrain-single --delexicalize_labels --epochs 200 --elmo --pooled 2> logs/pretrain-$pretrain-single-stdout.log > logs/pretrain-$pretrain-single-stderr.log

