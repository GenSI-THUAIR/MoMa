import errno
import os
from functools import cached_property
from logging import getLogger
import re

import numpy as np
import torch
import json
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
from ocpmodels.trainers.mt.normalizer import denormalize_context
from ocpmodels.trainers.mt.scaling.compat import load_scales_compat
from ocpmodels.trainers.ocp_trainer import OCPTrainer

from .config import FinetuneConfig, OptimConfig, OptimizerTrainerContext
from .dataset import FTDatasetsConfig, create_ft_datasets
from .optimizer import load_optimizer
from .util import (
    EarlyStopping,
    Label_prop_cosine,
    Label_prop_s2q,
    load_state_dict,
)

log = getLogger(__name__)


import json
import pickle
from collections import OrderedDict


@registry.register_trainer("ft")
class FTTrainer(OCPTrainer):
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

        batch_size = self.config["optim"]["batch_size"]
        assert isinstance(batch_size, int), f"{batch_size=} is not an integer"
        eval_batch_size = self.config["optim"].get(
            "eval_batch_size", batch_size
        )
        assert isinstance(
            eval_batch_size, int
        ), f"{eval_batch_size=} is not an integer"

        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = create_ft_datasets(self.dataset_config, self.model_config)

        self.flag_loader = None
        self.flag_val_loader = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if self.train_dataset is not None:
            self.train_sampler = self.get_sampler(
                self.train_dataset, batch_size, shuffle=True
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset, self.train_sampler
            )
            # construct flag_dataloader for modules choosing
            flag_batchsize = self.config["task"].get("flag_batchsize", 32)
            # self.flag_samplers = [self.get_sampler(self.train_dataset, flag_batchsize, shuffle=True, seed=i) for i in range(5)]
            self.flag_sampler = self.get_sampler(
                self.train_dataset, batch_size=flag_batchsize, shuffle=True
            )
            # self.flag_loaders = [self.get_dataloader(self.train_dataset, sampler) for sampler in self.flag_samplers]
            self.flag_loader = self.get_dataloader(
                self.train_dataset, self.flag_sampler
            )

        if self.val_dataset is not None:
            self.val_sampler = self.get_sampler(
                self.val_dataset, eval_batch_size, shuffle=False
            )
            self.val_loader = self.get_dataloader(
                self.val_dataset, self.val_sampler
            )

        if self.test_dataset is not None:
            self.test_sampler = self.get_sampler(
                self.test_dataset, eval_batch_size, shuffle=False
            )
            self.test_loader = self.get_dataloader(
                self.test_dataset, self.test_sampler
            )
        # load relaxation dataset
        if "relax_dataset" in self.config["task"]:
            raise NotImplementedError(
                "Relaxation dataset not implemented for MT."
            )

    @cached_property
    def model_config(self):
        model_config_dict: dict = self.config["model_attributes"].copy()
        model_config_dict["name"] = self.config["model"]

        return TypeAdapter(ModelConfig).validate_python(model_config_dict)

    @override
    def load_model(self) -> None:
        # Build model
        if distutils.is_master():
            log.info(f"Loading model: {self.config['model']}")

        self.model = registry.get_model_class(self.config["model"])(
            self.output_targets,
            self.model_config,
            **self.config["model_attributes"],
        ).to(self.device)

        if distutils.is_master():
            log.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{self.model.num_params} parameters."
            )

        if self.logger is not None:
            self.logger.watch(self.model)

        self.model = OCPDataParallel(
            self.model,
            output_device=self.device,
            num_gpus=1 if not self.cpu else 0,
        )

        unused_parameters = False
        if self.config['task']['freeze_backbone']:
            unused_parameters = True
        log.info(f"Unused_parameters:{unused_parameters}")

        # pass the keyword argument 'find_unused_parameters=True' to torch.nn.parallel.DistributedDataParallel
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.device],
                # find_unused_parameters=unused_parameters,
                find_unused_parameters=True,
            )
        print("=" * 20 + "check if model is properly loaded" + "=" * 20)
        # print(self.model.module.module.adapter_layers)
        # print(self.model.module.module.module_dict)

    @cached_property
    def finetune_config(self):
        return FinetuneConfig.from_dict(self.config["task"].get("ft", {}))

    @cached_property
    def optim_config(self):
        return TypeAdapter(OptimConfig).validate_python(self.config["optim"])

    def _load_ft_checkpoint(self):
        with self.finetune_config.base_checkpoint.src.open("rb") as fp:
            checkpoint = torch.load(fp, map_location="cpu")
        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module")
        mod_key_count = next(iter(self.model.state_dict())).count("module")
        key_count_diff = mod_key_count - ckpt_key_count

        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]

        _ = load_state_dict(
            self.model,
            new_dict,
            ignore_keys_patterns=self.finetune_config.base_checkpoint.ignore_keys_patterns,
            strict=self.config["task"].get("strict_load", True),
        )

    def load_modules(self, combined_ckpt):
        print("Loading Pre-trained modules to Stage-2 Model")

    @override
    def load_checkpoint(self, checkpoint_path: str):
        # First, we want to load the fine-tuning checkpoint.
        # self._load_ft_checkpoint()

        # Then, we continue as normal, except we need to use our own ScaleFactor implementation

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                errno.ENOENT, "Checkpoint file not found", checkpoint_path
            )

        log.info(f"Loading checkpoint from: {checkpoint_path}")
        map_location = torch.device("cpu") if self.cpu else self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.epoch = checkpoint.get("epoch", 0)
        self.step = checkpoint.get("step", 0)
        self.best_val_metric = checkpoint.get("best_val_metric", None)
        self.primary_metric = checkpoint.get("primary_metric", None)

        # Match the "module." count in the keys of model and checkpoint state_dict
        # DataParallel model has 1 "module.",  DistributedDataParallel has 2 "module."
        # Not using either of the above two would have no "module."

        ckpt_key_count = next(iter(checkpoint["state_dict"])).count("module.")
        mod_key_count = next(iter(self.model.state_dict())).count("module.")
        key_count_diff = mod_key_count - ckpt_key_count
        # print(next(iter(checkpoint["state_dict"])))
        # print("ckpt_key_count:", ckpt_key_count)
        # print("mod_key_count:", mod_key_count)
        # print("key_count_diff", key_count_diff)

        # align the number of module. in ckpt keys
        if key_count_diff > 0:
            new_dict = {
                key_count_diff * "module." + k: v
                for k, v in checkpoint["state_dict"].items()
            }
        elif key_count_diff < 0:
            new_dict = {
                k[len("module.") * abs(key_count_diff) :]: v
                for k, v in checkpoint["state_dict"].items()
            }
        else:
            new_dict = checkpoint["state_dict"]

        strict = self.config["task"].get("strict_load", True)  # no use

        # check whether to load pre-trained modules at stage 2
        load_full_module = self.config["task"].get("load_full_module", False)
        module_select = self.config["task"].get("module_select", False)        
        
        _ = load_state_dict(self.model, new_dict, strict=strict)
        
        if load_full_module and not module_select:
            print('---'*20)
            print('loading full modules...')
            selected_modules = self.config["task"]["selected_modules"]
            selected_module_weights = self.config["task"]["selected_module_weights"]
            use_module_weights = self.config["task"].get("use_module_weights", False)
            module_base_path = self.config["task"].get("module_base_path", None)

            if type(selected_module_weights) == int or type(selected_module_weights) == float:
                selected_module_weights = [selected_module_weights]

            print(type(selected_modules))
            print(selected_modules)
            print(type(selected_module_weights))
            print(selected_module_weights)

            base_path="/users/mdata/Multi-task-modular/checkpoints/"

            merged_state_dict = None
            
            for selected_module, module_weight in zip(selected_modules, selected_module_weights):
                print(f"Use module_weights: {use_module_weights}; Module weights: {module_weight}")
                if module_base_path:
                    module_path = f"{module_base_path}/{selected_module}.pt"
                else:
                    module_path = base_path+f'10-18-stage1-FullFT_10_epochs_{selected_module}/best_checkpoint.pt'
                
                if os.path.exists(module_path):
                    print(f"Path exists: {module_path}")
                    ckpt = torch.load(module_path, map_location="cpu")
                    state_dict = ckpt['state_dict']
                else:
                    print(f"Module path does not exist: {module_path}")
                    exit(0)

                if merged_state_dict is None:
                    # Initialize the averaged_state_dict with zeros
                    merged_state_dict = OrderedDict({k: torch.zeros_like(v) for k, v in state_dict.items()})                    

                for key in merged_state_dict:
                    if "module_dict.y" in key:
                        continue
                    if use_module_weights:
                        merged_state_dict[key] += state_dict[key] * module_weight
                    else:
                        merged_state_dict[key] += state_dict[key]

            if not use_module_weights:
                print(f"Averaging {len(selected_modules)} modules.")
                for key in merged_state_dict:
                    merged_state_dict[key] /= len(selected_modules)

            print("len(merged_state_dict):", len(merged_state_dict))

            _ = load_state_dict(self.model, merged_state_dict)
            # exit(0)

            
        # check if backbone need to be frozen
        freeze_backbone = self.config["task"].get("freeze_backbone", False)

        if freeze_backbone:
            print("Freeze backbone = ", freeze_backbone)
            # first freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # then unfreeze MoE(s) & Head(s)

            try:
                for param in self.model.module.module.moes.parameters():
                    param.requires_grad = True
            except:
                print("No moe params in the module.")

            try:
                for name, param in self.model.module.module.adapter_layers.named_parameters():
                    param.requires_grad = True
                    print(f"Unfreezing parameter: {name}")
            except:
                print("No adapter params in the module.")

            for param in self.model.module.module.module_dict.parameters():
                param.requires_grad = True
        else:
            print("Freeze backbone = ", freeze_backbone)
            for param in self.model.parameters():
                param.requires_grad = True

        # check if modules need to be frozen
        freeze_modules = self.config["task"].get("freeze_modules", False)
        if freeze_modules:
            for param in self.model.module.module.moes.y.modules.parameters():
                param.requires_grad = False
        else:
            pass

        # check the number of unfrozen parameters
        print(
            f"Total number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        print(
            f"Total number of unfrozen parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
        if "ema" in checkpoint and checkpoint["ema"] is not None:
            assert (
                self.ema is not None
            ), "EMA not initialized but found in checkpoint."
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None

        scale_dict = checkpoint.get("scale_dict", None)
        if scale_dict:
            log.info(
                "Overwriting scaling factors with those loaded from checkpoint. "
                "If you're generating predictions with a pretrained checkpoint, this is the correct behavior. "
                "To disable this, delete `scale_dict` from the checkpoint. "
            )
            load_scales_compat(self._unwrapped_model, scale_dict)

        for key in checkpoint["normalizers"]:
            if key in self.normalizers:
                self.normalizers[key].load_state_dict(
                    checkpoint["normalizers"][key]
                )
            if self.scaler and checkpoint["amp"]:
                self.scaler.load_state_dict(checkpoint["amp"])

    @override
    def load_optimizer(self) -> None:
        num_steps_per_epoch = len(self.train_loader)
        self.optimizer, self.scheduler, self.ema = load_optimizer(
            self.model,
            self.optim_config,
            OptimizerTrainerContext(num_steps_per_epoch=num_steps_per_epoch),
        )

    @override
    def load_extras(self) -> None:
        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm", False)

    @override
    def train(self, disable_eval_tqdm: bool = False) -> None:
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.evaluation_metrics.get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ):
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        start_epoch = self.step // len(self.train_loader)

        early_stop_patience = self.config["optim"].get(
            "early_stop_patience", 10
        )
        print("early_stop_patience", early_stop_patience)
        early_stopping = EarlyStopping(patience=early_stop_patience)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(
                    enabled=self.scaler is not None
                ):
                    out, _ = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale, self.metrics
                )

                # Log metrics.
                log_dict = {
                    k: self.metrics[k]["metric"] for k in self.metrics
                }
                log_dict.update(
                    {
                        # "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                        **self.scheduler.get_lr_dict(),
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v)
                        for k, v in log_dict.items()
                    ]
                    log.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                # if (
                #     checkpoint_every != -1
                #     and self.step % checkpoint_every == 0
                # ):
                #     self.save(
                #         checkpoint_file="checkpoint.pt",
                #         # checkpoint_file="checkpoint.pt".format(self.step),
                #         training_state=True,
                #     )

                self.scheduler.step()

            # Evaluate on val set every `eval_every` iterations.
            # if self.step % eval_every == 0:
            # Evaluate on every epoch
            if self.val_loader is not None:
                val_metrics = self.validate(
                    split="val",
                    disable_tqdm=disable_eval_tqdm,
                )
                self.update_best(
                    primary_metric,
                    val_metrics,
                    disable_eval_tqdm=disable_eval_tqdm,
                )

                early_stopping(
                    val_metrics[primary_metric]["metric"], self.model
                )
                if early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch_int}")                      
                    break
                else:
                    print(
                        f"Early stopping counter: {early_stopping.counter}"
                    )

            if self.config["task"].get("eval_relaxations", False):
                if "relax_dataset" not in self.config["task"]:
                    log.warning(
                        "Cannot evaluate relaxations, relax_dataset not specified"
                    )
                else:
                    self.run_relaxations()

            if self.step % eval_every == 0:
                self.scheduler.rlp_step(
                    val_metrics[primary_metric]["metric"]
                )

            torch.cuda.empty_cache()

            # if checkpoint_every == -1:
            #     self.save(
            #         checkpoint_file="checkpoint.pt", training_state=True
            #     )
            
            # cur_epoch = epoch_int +1
            # if (cur_epoch == 1 or cur_epoch == 5 or cur_epoch % 10 == 0):
            #     self.save(
            #         # checkpoint_file="checkpoint.pt",
            #         checkpoint_file=f"checkpoint_epoch_{cur_epoch}.pt",
            #         training_state=False,
            #     )
            #     print(f"Saved checkpoint_epoch_{cur_epoch}.pt")

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

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

    @override
    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        with denormalize_context(batch_list, [out], []) as (
            denormed_batch_list,
            [denormed_out],
            _,
        ):
            return super()._compute_metrics(
                denormed_out.copy(),
                denormed_batch_list,
                evaluator,
                metrics,
            )
        
    def _reinitialize_optimizer_and_scheduler(self):
        # Save the state of the current optimizer and scheduler
        optimizer_state_dict = self.optimizer.state_dict()
        if self.scheduler.scheduler is not None:
            scheduler_state_dict = self.scheduler.scheduler.state_dict()
        else:
            scheduler_state_dict = None

        # Reinitialize the optimizer and scheduler using the existing configuration
        optimizer_config = self.optim_config
        context = OptimizerTrainerContext(
            num_steps_per_epoch=len(self.train_loader),
        )

        self.optimizer, self.scheduler, self.ema = load_optimizer(
            self.model,
            optimizer_config,
            context,
        )

        # Load the optimizer state dict to retain optimizer states for existing parameters
        # self.optimizer.load_state_dict(optimizer_state_dict)

        # Load the scheduler state dict if applicable
        # if scheduler_state_dict is not None and self.scheduler.scheduler is not None:
        #     self.scheduler.load_state_dict(scheduler_state_dict)