export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export PATH=/root/miniconda3/envs/fm2m/bin/python:$PATH

# GPU configs
devices="3"
n_gpus=1

port=1236
few_shot_datapath='./data'
checkpoint_base_path='./hub'

modules=('mp_eform' 'mp_bandgap' 'n_Seebeck' 'n_avg_eff_mass' 'n_th_cond' 'p_Seebeck' 'p_avg_eff_mass' 'p_e_cond' 'n_e_cond' 'p_th_cond' 'mp_gvrh' 'castelli_eform' 'jarvis_gvrh' 'jarvis_bandgap' 'jarvis_eform' 'jarvis_kvrh' 'mp_kvrh' 'jarvis_dielectric_opt')
datasets=('piezoelectric_tensor' 'expt_eform' 'mp_poly_total' 'mp_poisson_ratio' 'jarvis_2d_dielectric_opt' 'jarvis_2d_eform' 'jarvis_2d_exfoliation' 'jarvis_2d_gap_opt' 'mp_elastic_anisotropy' 'mp_eps_electronic' 'mp_eps_total' 'mp_phonons' 'mp_poly_electronic' 'expt_bandgap' 'jarvis_3d_eps_tbmbj' 'jarvis_3d_gap_tbmbj' 'mp_dielectric')

for data_name in "${datasets[@]}"; do
for split in {'0','1','2','3','4'}; do
for module_name in "${modules[@]}"; do
        dict_mean=$(python3 -c "import json; import sys; data_name='$data_name'; print(json.load(open('json/stage2_mean.json'))[data_name])")
        dict_std=$(python3 -c "import json; import sys; data_name='$data_name'; print(json.load(open('json/stage2_std.json'))[data_name])")
        echo "Running $data_name split $split for module $module_name"
        WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$devices python -m torch.distributed.run --nproc_per_node=$n_gpus --master-port=$port main.py \
                --num-gpus $n_gpus --distributed \
                --mode train \
                --task.data_name="$data_name" \
                --task.split=$split \
                --task.flag_batchsize=16 \
                --config-yml configs/stage2_select_modules.yml \
                --dataset.train.config.src="$few_shot_datapath/$data_name/$split/train" \
                --dataset.val[0].config.src="$few_shot_datapath/$data_name/$split/val" \
                --dataset.test[0].config.src="$few_shot_datapath/$data_name/$split/test" \
                --task.load_module_name=$module_name \
                --dataset.normalization.y.mean=$dict_mean \
                --dataset.normalization.y.std=$dict_std \
                --identifier module-select+$data_name+split$split \
                --task.base_checkpoint.src="${checkpoint_base_path}/${module_name}.pt" \
                --task.module_embeddings_save_path="./outputs/embeddings" \
                --task.labels_save_path="./outputs/labels"

done
done
done

# Next steps:
# python scripts/run_knn.py  # knn-based prediction
# python scripts/weight_optimize.py  # module weight optimization