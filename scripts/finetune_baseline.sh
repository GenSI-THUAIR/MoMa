#!/bin/bash
# system configs
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

# GPU configs
devices="0"
n_gpus=1

# tasks configs
few_shot_datapath='./data'
ckpt_path='./jmp-l.pt' # NOTE: check jmp-l.pt

datasets=('piezoelectric_tensor' 'expt_eform' 'mp_poly_total' 'mp_poisson_ratio' 'jarvis_2d_dielectric_opt' 'jarvis_2d_eform' 'jarvis_2d_exfoliation' 'jarvis_2d_gap_opt' 'mp_elastic_anisotropy' 'mp_eps_electronic' 'mp_eps_total' 'mp_phonons' 'mp_poly_electronic' 'expt_bandgap' 'jarvis_3d_eps_tbmbj' 'jarvis_3d_gap_tbmbj' 'mp_dielectric')

for data_name in $datasets; do
for split in {'0','1','2','3','4'}; do

dict_mean=$(python3 -c "import json; import sys; data_name='$data_name'; print(json.load(open('json/stage2_mean.json'))[data_name])")
dict_std=$(python3 -c "import json; import sys; data_name='$data_name'; print(json.load(open('json/stage2_std.json'))[data_name])")

echo "mean of $data_name is $dict_mean"
echo "std of $data_name is $dict_std"

WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$devices python -m torch.distributed.run --nproc_per_node=$n_gpus --master-port=1234 main.py \
        --num-gpus $n_gpus --distributed \
        --mode train \
        --task.data_name=$data_name \
        --config-yml configs/stage2_base.yml \
        --dataset.train.config.src="$few_shot_datapath/$data_name/$split/train" \
        --dataset.val[0].config.src="$few_shot_datapath/$data_name/$split/val" \
        --dataset.test[0].config.src="$few_shot_datapath/$data_name/$split/test" \
        --dataset.normalization.y.mean=${dict_mean[$data_name]} \
        --dataset.normalization.y.std=${dict_std[$data_name]} \
        --identifier Stg2_$data_name+_split$split+_E$expert_num+K$K \
        --task.base_checkpoint.src="$ckpt_path" \

done
done