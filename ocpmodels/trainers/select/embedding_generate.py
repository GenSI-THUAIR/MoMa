import os
from functools import cached_property
from logging import getLogger
import pickle
import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from typing_extensions import override

from ocpmodels.common import distutils, gp_utils
from ocpmodels.common.data_parallel import OCPDataParallel
from ocpmodels.common.registry import registry
from ocpmodels.common.typed_config import TypeAdapter
from ocpmodels.common.utils import scatter_det
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.trainers.mt.balanced_batch_sampler import BalancedBatchSampler
from ocpmodels.trainers.mt.collate import ParallelCollater
from ocpmodels.trainers.mt.config import ModelConfig
from ocpmodels.trainers.ocp_trainer import OCPTrainer

from ..ft.config import FinetuneConfig, OptimConfig
from ..ft.dataset import FTDatasetsConfig, create_ft_datasets

log = getLogger(__name__)


@registry.register_trainer("embed_gen")
class EmbedGenTrainer(OCPTrainer):
    @override
    def __init__(
        self,
        task,
        model,
        outputs,
        dataset,
        optimizer,
        loss_fns,
        eval_metrics,
        identifier,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
        name="ocp",
    ):
        super().__init__(
            task,
            model,
            outputs,
            FTDatasetsConfig.from_dict(
                dataset
            ),  
            optimizer,
            loss_fns,
            eval_metrics,
            identifier,
            timestamp_id,
            run_dir,
            is_debug,
            print_every,
            seed,
            logger,
            local_rank,
            amp,
            cpu,
            slurm,
            noddp,
            name,
        )

    @override
    def load(self) -> None:
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_task()
        print("=" * 20 + "Start loading model", "=" * 20)
        self.load_model()
        print("=" * 20 + "Finish loading model", "=" * 20)
        print("=" * 20 + "Start loading checkpoint", "=" * 20)
        self.load_checkpoint(self.config["task"]["base_checkpoint"]["src"])
        print("=" * 20 + "Finish loading checkpoint", "=" * 20)
        # print("=" * 20 + "Start loading loss", "=" * 20)
        self.load_loss()        
        
    @property
    def dataset_config(self):
        dataset_config = self.config["dataset"]
        assert isinstance(
            dataset_config, FTDatasetsConfig
        ), f"{dataset_config=} is not a FTDatasetsConfig"
        return dataset_config

    @override
    def load_datasets(self) -> None:
        log.info("Loading datasets")
        self.parallel_collater = ParallelCollater(
            0 if self.cpu else 1,
            otf_graph=self.config["model_attributes"].get("otf_graph", False),
        )

        (self.train_dataset, self.val_dataset, self.test_dataset) = create_ft_datasets(
            self.dataset_config, self.model_config
        )

        # loaders -------------------------------------------------------
        self.flag_loader = None
        if self.train_dataset is not None:
            flag_bs = self.config["task"].get("flag_batchsize", 32)
            sampler = self.get_sampler(self.train_dataset, flag_bs, shuffle=True)
            self.flag_loader = self.get_dataloader(self.train_dataset, sampler)
    
    def embed_gen(self):
        self.model.eval()
        # pass the whole training set to all modules and compute the consistency coefficient for each module
        print("=" * 20 + "Start module Masking" + "=" * 20)

        module_outputs = []
        all_labels = []

        # Create an iterator for the train loader
        flag_loader_iter = iter(self.flag_loader)

        data_name = self.config["task"]["data_name"]

        split = self.config["task"]["split"]

        for flag_batch in flag_loader_iter:
            with torch.no_grad():
                _, module_output = self._forward(flag_batch)
                # module_output = self._forward(flag_batch)
                # print(f"module_output.shape: {module_output.shape}")

                index = flag_batch[0].batch
                natoms = flag_batch[0].natoms.shape[0]
                target_name = list(self.output_targets.keys())[0]  
                labels = flag_batch[0][target_name].unsqueeze(1)
                all_labels.append(labels)

                module_output = module_output.permute(
                    1, 0, 2
                )  # N_modules * N_tokens * module_dim
                chunked_module_features = torch.chunk(
                    module_output, chunks=module_output.size(0), dim=0
                )  # a list with N features
                # print(f"len(chunked_module_features): {len(chunked_module_features)}")

                reduced_batch_feature = []

                for i in range(len(chunked_module_features)):
                    features = chunked_module_features[i]
                    features = features.squeeze(0)

                    reduced_features = scatter_det(
                        features,
                        index,
                        dim=0,
                        dim_size=natoms,
                        reduce="mean",
                    )
                    reduced_batch_feature.append(reduced_features)
                # print(f"reduced_batch_feature.shape: {reduced_features.shape}")

                module_outputs.append(reduced_batch_feature)

        module_outputs = [
            torch.cat(subtensors) for subtensors in zip(*module_outputs)
        ]
        print(
            "the shape of all modules outputs with all training data",
            len(module_outputs),
            module_outputs[0].shape,
        )
        labels = torch.cat(all_labels, dim=0)
        print("the shape of labels of all training data", labels.shape)

        module_name = self.config["task"].get("load_module_name")
        print(f"Loading full module: {module_name}")

        module_embeddings_save_base_path = self.config["task"].get("module_embeddings_save_path")
        if not os.path.exists(module_embeddings_save_base_path):
            os.makedirs(module_embeddings_save_base_path)
        module_embeddings_save_path = f'{module_embeddings_save_base_path}/{data_name}_split{split}_module_{module_name}.pkl'
        labels_save_base_path = self.config["task"].get("labels_save_path")
        if not os.path.exists(labels_save_base_path):
            os.makedirs(labels_save_base_path)
        labels_save_path = f'{labels_save_base_path}/{data_name}_split{split}_module_{module_name}.pkl'

        with open(module_embeddings_save_path, 'wb') as f:
            module_outputs_cpu = [e.cpu() for e in module_outputs]
            print(module_outputs_cpu[0])
            pickle.dump(module_outputs_cpu, f)

        with open(labels_save_path, 'wb') as f:
            labels_cpu = labels.cpu()
            pickle.dump(labels_cpu, f)

        print(f"Save module outputs to {module_embeddings_save_path}.")
        print(f"Save labels to {labels_save_path}.")

        return module_outputs

    @cached_property
    def model_config(self):
        model_config_dict: dict = self.config["model_attributes"].copy()
        model_config_dict["name"] = self.config["model"]

        return TypeAdapter(ModelConfig).validate_python(model_config_dict)

    @override
    def load_model(self):
        if distutils.is_master():
            log.info(f"Loading model: {self.config['model']}")
        self.model = registry.get_model_class(self.config["model"])(
            self.output_targets, self.model_config, **self.config["model_attributes"]
        ).to(self.device)
        self.model = OCPDataParallel(self.model, output_device=self.device, num_gpus=1 if not self.cpu else 0)
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(self.model, device_ids=[self.device], find_unused_parameters=True)

    @cached_property
    def finetune_config(self):
        return FinetuneConfig.from_dict(self.config["task"].get("ft", {}))

    @cached_property
    def optim_config(self):
        return TypeAdapter(OptimConfig).validate_python(self.config["optim"])

    @override
    def train(self, disable_eval_tqdm: bool = False) -> None:
        ensure_fitted(self._unwrapped_model, warn=True)

        self.embed_gen()

    @override
    def get_sampler(
        self, dataset, batch_size: int, shuffle: bool, seed=0
    ) -> BalancedBatchSampler:
        balancing_mode = self.config["optim"].get("load_balancing", None)
        on_error = self.config["optim"].get("load_balancing_on_error", None)
        if balancing_mode is not None:
            if on_error is None:
                on_error = "raise"
        else:
            balancing_mode = "atoms"

        if on_error is None:
            on_error = "warn_and_no_balance"

        if gp_utils.initialized():
            num_replicas = gp_utils.get_dp_world_size()
            rank = gp_utils.get_dp_rank()
            raise NotImplementedError("GP not implemented for MT/FT.")
        else:
            num_replicas = distutils.get_world_size()
            rank = distutils.get_rank()

        sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            on_error=on_error,
        )
        return sampler