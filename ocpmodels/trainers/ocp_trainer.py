"""
Open Catalyst Project (OCP) trainer implementation.

This module provides the OCPTrainer class that extends the base trainer for
Structure to Energy & Force (S2EF) and Initial State to Relaxed State (IS2RS) tasks.
"""

# Standard library imports
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Third-party imports
import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

# Local imports
from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import cg_decomp_mat, check_traj_files, irreps_sum
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

@registry.register_trainer("ocp")
@registry.register_trainer("energy")
@registry.register_trainer("forces")
class OCPTrainer(BaseTrainer):
    """Trainer class for Structure to Energy & Force (S2EF) and Initial State to Relaxed State (IS2RS) tasks.
    
    This trainer extends the base trainer with specific functionality for handling
    energy and force predictions, as well as structure relaxation tasks.
    
    Args:
        task (Dict[str, Any]): Task configuration dictionary
        model (Dict[str, Any]): Model configuration dictionary
        outputs (Dict[str, Any]): Output configuration dictionary
        dataset (Union[Dict[str, Any], List[Any]]): Dataset configuration
        optimizer (Dict[str, Any]): Optimizer configuration dictionary
        loss_fns (Dict[str, Any]): Loss function configuration dictionary
        eval_metrics (Dict[str, Any]): Evaluation metrics configuration dictionary
        identifier (str): Unique identifier for this training run
        timestamp_id (Optional[str]): Optional timestamp ID for this run
        run_dir (Optional[str]): Optional directory to save run artifacts
        is_debug (bool): Whether to run in debug mode
        print_every (int): How often to print training progress
        seed (Optional[int]): Random seed for reproducibility
        logger (Union[str, Dict[str, Any]]): Logger configuration
        local_rank (int): Local rank for distributed training
        amp (bool): Whether to use automatic mixed precision
        cpu (bool): Whether to force CPU training
        slurm (Dict[str, Any]): SLURM configuration dictionary
        noddp (bool): Whether to disable distributed data parallel
        name (str): Name of the trainer
    """

    def __init__(
        self,
        task: Dict[str, Any],
        model: Dict[str, Any],
        outputs: Dict[str, Any],
        dataset: Union[Dict[str, Any], List[Any]],
        optimizer: Dict[str, Any],
        loss_fns: Dict[str, Any],
        eval_metrics: Dict[str, Any],
        identifier: str,
        timestamp_id: Optional[str] = None,
        run_dir: Optional[str] = None,
        is_debug: bool = False,
        print_every: int = 100,
        seed: Optional[int] = None,
        logger: Union[str, Dict[str, Any]] = "tensorboard",
        local_rank: int = 0,
        amp: bool = False,
        cpu: bool = False,
        slurm: Dict[str, Any] = {},
        noddp: bool = False,
        name: str = "ocp",
    ) -> None:
        """Initialize the OCP trainer.
        
        Args:
            task: Task configuration dictionary
            model: Model configuration dictionary
            outputs: Output configuration dictionary
            dataset: Dataset configuration dictionary or list of datasets
            optimizer: Optimizer configuration dictionary
            loss_fns: Loss function configuration dictionary
            eval_metrics: Evaluation metrics configuration dictionary
            identifier: Unique identifier for this training run
            timestamp_id: Optional timestamp ID for this run
            run_dir: Optional directory to save run artifacts
            is_debug: Whether to run in debug mode
            print_every: How often to print training progress
            seed: Random seed for reproducibility
            logger: Logger configuration
            local_rank: Local rank for distributed training
            amp: Whether to use automatic mixed precision
            cpu: Whether to force CPU training
            slurm: SLURM configuration dictionary
            noddp: Whether to disable distributed data parallel
            name: Name of the trainer
        """
        super().__init__(
            task=task,
            model=model,
            outputs=outputs,
            dataset=dataset,
            optimizer=optimizer,
            loss_fns=loss_fns,
            eval_metrics=eval_metrics,
            identifier=identifier,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            slurm=slurm,
            noddp=noddp,
            name=name,
        )

    def run_relaxations(self, split: str = "val") -> None:
        """Run ML-based structure relaxations.
        
        This method performs structure relaxation using the trained model.
        It handles both validation and test splits, and computes relevant metrics.
        
        Args:
            split: Data split to run relaxations on ("val" or "test")
            
        Raises:
            ValueError: If split is not "val" or "test"
            RuntimeError: If relaxation fails
        """
        if split not in ["val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be 'val' or 'test'")
            
        try:
            ensure_fitted(self._unwrapped_model)

            # When set to true, uses deterministic CUDA scatter ops, if available.
            # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
            # Only implemented for GemNet-OC currently.
            registry.register(
                "set_deterministic_scatter",
                self.config["task"].get("set_deterministic_scatter", False),
            )

            logger.info("Running ML-relaxations")
            self.model.eval()
            if self.ema:
                self.ema.store()
                self.ema.copy_to()

            evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
            evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

            # Need both `pos_relaxed` and `y_relaxed` to compute val IS2R* metrics.
            # Else just generate predictions.
            if (
                hasattr(self.relax_dataset[0], "pos_relaxed")
                and self.relax_dataset[0].pos_relaxed is not None
            ) and (
                hasattr(self.relax_dataset[0], "y_relaxed")
                and self.relax_dataset[0].y_relaxed is not None
            ):
                split = "val"
            else:
                split = "test"

            ids = []
            relaxed_positions = []
            chunk_idx = []
            
            for i, batch in tqdm(
                enumerate(self.relax_loader), total=len(self.relax_loader)
            ):
                if i >= self.config["task"].get("num_relaxation_batches", 1e9):
                    break

                # If all traj files already exist, then skip this batch
                if check_traj_files(
                    batch, self.config["task"]["relax_opt"].get("traj_dir", None)
                ):
                    logger.info(f"Skipping batch: {batch[0].sid.tolist()}")
                    continue

                try:
                    relaxed_batch = ml_relax(
                        batch=batch,
                        model=self,
                        steps=self.config["task"].get("relaxation_steps", 200),
                        fmax=self.config["task"].get("relaxation_fmax", 0.0),
                        relax_opt=self.config["task"]["relax_opt"],
                        save_full_traj=self.config["task"].get("save_full_traj", True),
                        device=self.device,
                        transform=None,
                    )
                except Exception as e:
                    logger.error(f"Relaxation failed for batch {i}: {str(e)}")
                    continue

                if self.config["task"].get("write_pos", False):
                    systemids = [str(i) for i in relaxed_batch.sid.tolist()]
                    natoms = relaxed_batch.natoms.tolist()
                    positions = torch.split(relaxed_batch.pos, natoms)
                    batch_relaxed_positions = [pos.tolist() for pos in positions]

                    relaxed_positions += batch_relaxed_positions
                    chunk_idx += natoms
                    ids += systemids

                if split == "val":
                    mask = relaxed_batch.fixed == 0
                    s_idx = 0
                    natoms_free = []
                    for natoms in relaxed_batch.natoms:
                        natoms_free.append(
                            torch.sum(mask[s_idx : s_idx + natoms]).item()
                        )
                        s_idx += natoms

                    target = {
                        "energy": relaxed_batch.y_relaxed,
                        "positions": relaxed_batch.pos_relaxed[mask],
                        "cell": relaxed_batch.cell,
                        "pbc": torch.tensor([True, True, True]),
                        "natoms": torch.LongTensor(natoms_free),
                    }

                    prediction = {
                        "energy": relaxed_batch.y,
                        "positions": relaxed_batch.pos[mask],
                        "cell": relaxed_batch.cell,
                        "pbc": torch.tensor([True, True, True]),
                        "natoms": torch.LongTensor(natoms_free),
                    }

                    metrics_is2rs = evaluator_is2rs.eval(
                        prediction,
                        target,
                        metrics_is2rs,
                    )
                    metrics_is2re = evaluator_is2re.eval(
                        {"energy": prediction["energy"]},
                        {"energy": target["energy"]},
                        metrics_is2re,
                    )

            if self.config["task"].get("write_pos", False):
                self._save_relaxed_positions(ids, relaxed_positions, chunk_idx)

        except Exception as e:
            logger.error(f"Error during relaxation: {str(e)}")
            raise RuntimeError(f"Relaxation failed: {str(e)}")

    def _save_relaxed_positions(
        self, ids: List[str], relaxed_positions: List[Any], chunk_idx: List[int]
    ) -> None:
        """Save relaxed positions to disk.
        
        Args:
            ids: List of system IDs
            relaxed_positions: List of relaxed positions
            chunk_idx: List of chunk indices
        """
        rank = distutils.get_rank()
        pos_filename = os.path.join(
            self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.npz"
        )
        np.savez_compressed(
            pos_filename,
            ids=ids,
            pos=np.array(relaxed_positions, dtype=object),
            chunk_idx=chunk_idx,
        )

        distutils.synchronize()
        if distutils.is_master():
            gather_results = defaultdict(list)
            full_path = os.path.join(
                self.config["cmd"]["results_dir"],
                "relaxed_positions.npz",
            )

            for i in range(distutils.get_world_size()):
                rank_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    f"relaxed_pos_{i}.npz",
                )
                rank_results = np.load(rank_path, allow_pickle=True)
                gather_results["ids"].extend(rank_results["ids"])
                gather_results["pos"].extend(rank_results["pos"])
                gather_results["chunk_idx"].extend(rank_results["chunk_idx"])

            np.savez_compressed(
                full_path,
                ids=np.array(gather_results["ids"]),
                pos=np.array(gather_results["pos"], dtype=object),
                chunk_idx=np.array(gather_results["chunk_idx"]),
            )

            # Clean up individual rank files
            for i in range(distutils.get_world_size()):
                rank_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    f"relaxed_pos_{i}.npz",
                )
                try:
                    os.remove(rank_path)
                except OSError as e:
                    logger.warning(f"Could not remove {rank_path}: {e}")
