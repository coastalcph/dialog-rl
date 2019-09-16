#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=6000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:2
# We expect that our program should not run langer than 4 hours
# Note that a program will be killed once it exceeds this time!
#SBATCH --time=120:00:00
   
#your script, in this case: write the hostname and the ids of the chosen gpus.
hostname
echo $CUDA_VISIBLE_DEVICES
source /home/vtx829/.env/bin/activate

DOMAINS=(taxi hotel restaurant attraction train);
#DOMAINS=(attraction);
gpu=1

for domain in ${DOMAINS[@]};
do
        echo "=============================="
        date;
        echo "Target domain: " $domain;
        echo "  Pretrained model:" ${pretrain[@]};
        python -m run --train_domains $domain --eval_domains $domain --lr 0.0001 --epochs 1000 --gpu $gpu -n p/finetune-$domain --max_dev_dialogs 150 --delexicalize_labels --elmo --pooled --batch_size 16 --train_strict --patience 10 --train_strict --reinforce --resume exp/statenet/p/pretrain-$domain  &> logs/finetune-$domain.log 

        ((gpu++)) ;
done


