#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=6000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:1
#We expect that our program should not run langer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=20:00:00

hostname
echo $CUDA_VISIBLE_DEVICES

source /home/grn762/projects/dialog-rl/dialog_env/bin/activate

pretrain=(taxi restaurant attraction hotel train)
name=pretrain-all-no-elmo
bsize=32
gpu=1

echo "=============================="
echo "  Pretrain:" ${pretrain[@]};
python -m run --train_domains ${pretrain[@]} --batch_size $bsize --train_strict --eval_domains ${pretrain[@]} --gpu $gpu -n $name --delexicalize_labels --epochs 100 2> logs/$name-stdout.log > logs/$name-stderr.log

