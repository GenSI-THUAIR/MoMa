export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO
# export PATH=/root/miniconda3/envs/fm2m/bin/python:$PATH

# GPU configs
devices=5
n_gpus=1

# tasks configs
datapath='./data'
# datapath='/data/wangbt/MoMa_data/npj_lmdb'
ckpt_path='./jmp-l.pt' # TODO: check jmp-l.pt

modules=('mp_eform' 'mp_bandgap' 'n_Seebeck' 'n_avg_eff_mass' 'n_th_cond' 'p_Seebeck' 'p_avg_eff_mass' 'p_e_cond' 'n_e_cond' 'p_th_cond' 'mp_gvrh' 'castelli_eform' 'jarvis_gvrh' 'jarvis_bandgap' 'jarvis_eform' 'jarvis_kvrh' 'mp_kvrh' 'jarvis_dielectric_opt')

# Initialize last_checkpoint variable with an initial model if there is any, otherwise set to an empty string or a default model path
bsz=16

for data_name in $modules; do

  dict_mean=$(python3 -c "import json; import sys; data_name='$data_name'; print(json.load(open('json/stage1_mean.json'))[data_name])")
  dict_std=$(python3 -c "import json; import sys; data_name='$data_name'; print(json.load(open('json/stage1_std.json'))[data_name])")  

  echo "mean of $data_name is $dict_mean"
  echo "std of $data_name is $dict_std"

  if [ -z "$ckpt_path" ]; then
    # If last_checkpoint is empty, use the initial model
    echo "error using initial model"
    break
  fi

  if [ -f $ckpt_path ]; then
    echo "checkpoint: $ckpt_path"
  fi

  case "$data_name" in
    mp_bandgap|mp_eform|n_Seebeck|n_avg_eff_mass|n_th_cond|p_Seebeck|p_avg_eff_mass|p_e_cond|n_e_cond|p_th_cond)
    bsz=32
    ;;
  esac    

  WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$devices python -m torch.distributed.run --nproc_per_node=$n_gpus --master-port=1236 main.py \
          --num-gpus $n_gpus --distributed \
          --mode train \
          --config-yml configs/stage1_base.yml \
          --dataset.train.config.src="$datapath/$data_name/train" \
          --dataset.val[0].config.src="$datapath/$data_name/val" \
          --dataset.test[0].config.src="$datapath/$data_name/test" \
          --dataset.normalization.y.mean=$dict_mean \
          --dataset.normalization.y.std=$dict_std \
          --identifier stage1_$data_name \
          --timestamp-id stage1_$data_name \
          --task.base_checkpoint.src="$ckpt_path" \
          --optim.batch_size=16 \
          --optim.eval_batch_size=16

done