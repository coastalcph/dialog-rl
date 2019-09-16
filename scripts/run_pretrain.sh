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
#DOMAINS=(hotel);
gpu=1

for domain in ${DOMAINS[@]};
do
        pretrain=(${DOMAINS[@]//*$domain*});  # all but $domain
        echo "=============================="
        date;
        echo "Target domain: " $domain;
        echo "  Pretrain:" ${pretrain[@]};
        python -m run --train_domains ${pretrain[@]} --eval_domains $pretrain --lr 0.00003 --epochs 1000 --gpu $gpu -n p/pretrain-$domain --max_dev_dialogs 150 --delexicalize_labels --elmo --pooled --batch_size 16 --patience 10 --train_strict &> logs/pretrain-$domain.log 

        ((gpu++)) ;
done


