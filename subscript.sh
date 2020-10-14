#!/bin/bash
#SBATCH  --output=/home/cajad/sbatch_outputs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=20G

# call your calculation executable, redirect output
source /home/cajad/apps/miniconda/etc/profile.d/conda.sh
conda activate kpts
python --version 

#python -u train.py --dataset 'ShapeNet' --category 'airplane' --ckpt_model 'airplane_october1chf' --node_num 14 --node_knn_k_1 3 --basis_num 10 --surface_normal_len 0 --input_pc_num 1600 "$@"
python -u train.py --dataset 'ModelNet10' --category 'chair' --ckpt_model 'chair_october_14' --node_num 14 --node_knn_k_1 3 --basis_num 10 --surface_normal_len 3 --input_pc_num 3000 "$@"

# python -u test.py --dataset 'ShapeNet' --category 'motorbike' --ckpt_model 'motorbike_10b_10cov_03bsh' --node_num 18 --node_knn_k_1 3 --basis_num 10 --surface_normal_len 0 --input_pc_num 1600
# python test.py --dataset 'ShapeNet' --category 'airplane' --ckpt_model 'airplane_newcov' --node_num 14 --node_knn_k_1 3 --basis_num 10 --surface_normal_len 0 --input_pc_num 1600
# python test.py --dataset 'ShapeNet' --category 'car' --ckpt_model 'car_5b_10cov_03bsh_183' --node_num 18 --node_knn_k_1 3 --basis_num 5 --surface_normal_len 0 --input_pc_num 1600

# python test.py --dataset 'ModelNet10' --category 'chair' --ckpt_model 'chair_new_cov' --node_num 14 --node_knn_k_1 3 --basis_num 10 --surface_normal_len 3 --input_pc_num 3000
# table: table_123_5b_5cov
# bed: bed_143_4_05bs_0505npl

# sbatch --output=sbatch_log/%j.out --gres=gpu:1 --mem=30G subscript.sh
# squeue -u cajad
