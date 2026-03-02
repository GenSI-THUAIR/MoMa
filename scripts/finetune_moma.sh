#!/bin/bash
# system configs
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export PATH=/root/miniconda3/envs/fm2m/bin/python:$PATH
export WANDB_BASE_URL=https://api.bandw.top

# GPU configs
devices="0"
n_gpus=1
port=1288

# tasks configs
# few_shot_datapath='./data'
few_shot_datapath='/data/wangbt/MoMa_data/few_shot_lmdb'
moma_hub_path='./hub'
wandb_mode=dryrun

modules=('mp_eform' 'mp_bandgap' 'n_Seebeck' 'n_avg_eff_mass' 'n_th_cond' 'p_Seebeck' 'p_avg_eff_mass' 'p_e_cond' 'n_e_cond' 'p_th_cond' 'mp_gvrh' 'castelli_eform' 'jarvis_gvrh' 'jarvis_bandgap' 'jarvis_eform' 'jarvis_kvrh' 'mp_kvrh' 'jarvis_dielectric_opt')
datasets=('piezoelectric_tensor' 'expt_eform' 'mp_poly_total' 'mp_poisson_ratio' 'jarvis_2d_dielectric_opt' 'jarvis_2d_eform' 'jarvis_2d_exfoliation' 'jarvis_2d_gap_opt' 'mp_elastic_anisotropy' 'mp_eps_electronic' 'mp_eps_total' 'mp_phonons' 'mp_poly_electronic' 'expt_bandgap' 'jarvis_3d_eps_tbmbj' 'jarvis_3d_gap_tbmbj' 'mp_dielectric')

bsz=32
K=32

for data_name in $datasets; do
for split in {'0','1','2','3','4'}; do
        echo $data_name $split
        # Read the data from json file
        dict_mean=$(python3 -c "import json; import sys; data_name='$data_name'; print(json.load(open('json/stage2_mean.json'))[data_name])")
        dict_std=$(python3 -c "import json; import sys; data_name='$data_name'; print(json.load(open('json/stage2_std.json'))[data_name])")
        dict_select_modules=$(python3 -c "import json; import sys; data_name='$data_name'; print(json.load(open('./json/module_ids/split_${split}.json'))[data_name])")
        dict_num_experts=$(python3 -c "import json; import sys; data_name='$data_name'; print(json.load(open('./json/module_num/split_${split}.json'))[data_name])")
        weights=$(python3 -c "import json; data_name='$data_name'; data=json.load(open('./json/module_weights/split_${split}.json')); print(','.join(map(str, data[data_name])))")

        # print the data
        echo "mean of $data_name is $dict_mean"
        echo "std of $data_name is $dict_std"
        echo "selected experts for $data_name is $dict_select_modules"
        echo "num of experts for $data_name is $dict_num_experts"
        echo "weights for selected experts is $weights"

        IFS=',' read -r -a selected_modules_array <<< "$dict_select_modules"

        selected_module_names=()

        for selected_module in ${selected_modules_array[@]}; do
            echo "selected_module: ${modules[$selected_module]}"
            selected_module_names+="'${modules[$selected_module]}',"
        done

        echo [$selected_module_names]

        # stage 2 with pre-trained modules
        WANDB_MODE=$wandb_mode CUDA_VISIBLE_DEVICES=$devices python -m torch.distributed.run --nproc_per_node=$n_gpus --master_port=$port main.py \
            --num-gpus $n_gpus --distributed \
            --mode train \
            --task.data_name=$data_name \
            --config-yml configs/stage2_base.yml \
            --dataset.train.config.src="$few_shot_datapath/$data_name/$split/train" \
            --dataset.val[0].config.src="$few_shot_datapath/$data_name/$split/val" \
            --dataset.test[0].config.src="$few_shot_datapath/$data_name/$split/test" \
            --task.expert_select=False \
            --task.load_experts=False \
            --task.load_full_module=True \
            --task.selected_modules="[$selected_module_names]" \
            --task.selected_module_weights=$weights \
            --task.use_module_weights=True \
            --task.module_base_path=$moma_hub_path \
            --dataset.normalization.y.mean=$dict_mean \
            --dataset.normalization.y.std=$dict_std \
            --optim.batch_size=$bsz \
            --optim.eval_batch_size=$bsz \
            --model.edge_dropout=0.0 \
            --model.dropout=0.0 \
            --identifier moma+$data_name+split$split \
            --timestamp-id moma+$data_name+split$split \
            --logger.name="wandb" \
            --logger.project="MoMa" \
            --task.base_checkpoint.src="./jmp-l.pt"

done
done