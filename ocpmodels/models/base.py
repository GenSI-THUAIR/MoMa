"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from e3nn import o3
from torch_geometric.nn import radius_graph
from ocpmodels.models.gemnet_oc.layers.base_layers import Dense

from ocpmodels.common.utils import (
    cg_decomp_mat,
    compute_neighbors,
    get_pbc_distances,
    irreps_sum,
    radius_graph_pbc,
    scatter_det,
)

logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    """Base model class that handles common model functionality.
    
    This class provides the foundation for all model implementations in the OCP project.
    It handles common tasks like output target management, module initialization, and forward passes.
    
    Attributes:
        output_targets (Dict[str, Any]): Dictionary mapping output names to their configurations
        num_targets (int): Number of output targets
        active_modules (int): Number of active modules for each target
        total_modules (int): Total number of modules for each target
        module_masking (bool): Whether to use module masking
        active_module_indices (List[int]): List of active module indices
        module_embedding_dim (int): Embedding dimension for modules
        sparse (bool): Whether to use sparse routing
        top_k (int): Number of top experts to select
        module_layers (int): Number of layers in each module
        router_layers (int): Number of layers in the router
        with_node_embedding (bool): Whether to include node embeddings
        routers (nn.ModuleDict): Dictionary of routers for each target
        moes (nn.ModuleDict): Dictionary of mixture of experts for each target
        module_dict (nn.ModuleDict): Dictionary of output modules for each target
    """

    def __init__(
        self,
        output_targets: Dict[str, Any] = {},
        node_embedding_dim: Optional[int] = None,
        edge_embedding_dim: Optional[int] = None,
    ) -> None:
        """Initialize the base model.
        
        Args:
            output_targets: Dictionary mapping output names to their configurations
            node_embedding_dim: Dimension of node embeddings
            edge_embedding_dim: Dimension of edge embeddings
        """
        super().__init__()

        self.output_targets = output_targets
        self.num_targets = len(output_targets)

        # Initialize target-specific parameters
        for target in output_targets:
            self.active_modules = self.output_targets[target].get("num_active_modules", 1)
            self.total_modules = self.output_targets[target].get("num_total_modules", 72)
            self.module_masking = self.output_targets[target].get("module_masking", False)
            self.active_module_indices = self.output_targets[target].get("active_module_indices", [0])

            logger.debug(f"type(self.active_module_indices): {type(self.active_module_indices)}")
            logger.debug(f"self.active_module_indices: {self.active_module_indices}")

            self.module_embedding_dim = self.output_targets[target].get(
                "module_dim", node_embedding_dim
            )
            self.sparse = self.output_targets[target].get("sparse", False)
            self.top_k = self.output_targets[target].get("top_k", 2)
            self.module_layers = self.output_targets[target].get("e_layers", 4)
            self.router_layers = self.output_targets[target].get("r_layers", 2)
            self.with_node_embedding = self.output_targets[target].get('with_node_embedding', False)

        # Initialize output modules
        self.module_dict = nn.ModuleDict({})
        for target in output_targets:
            if self.output_targets[target].get("custom_head", False):
                if "irrep_dim" in self.output_targets[target]:
                    if edge_embedding_dim is None:
                        raise NotImplementedError(
                            "Model does not support SO(3) equivariant prediction without edge embeddings."
                        )
                    embedding_dim = edge_embedding_dim
                    output_shape = 1
                else:
                    embedding_dim = node_embedding_dim
                    output_shape = self.output_targets[target].get("shape", 1)

                bias = self.output_targets[target].get("bias", True)

                layers = [
                    Dense(
                        embedding_dim,
                        embedding_dim,
                        activation="silu",
                        bias=bias,
                    )
                    for _ in range(
                        self.output_targets[target].get("num_layers", 2)
                    )
                ]

                hidden_dim = embedding_dim

                layers.append(
                    Dense(
                        hidden_dim,
                        output_shape,
                        activation=None,
                        bias=not self.output_targets[target].get(
                            "no_final_bias", not bias
                        ),
                    )
                )

                self.module_dict[target] = nn.Sequential(*layers)

    def update_module_num(self, new_num_modules: int) -> None:
        """Update the number of modules for all targets.
        
        Args:
            new_num_modules: New number of modules to use
            
        Raises:
            AssertionError: If mixture of experts is not enabled
        """
        for target in self.output_targets:
            self.moes[target].update_num_modules(new_num_modules)

    @abstractmethod
    def _forward(self, data: Any) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary mapping output names to their predictions
        """
        pass

    def forward(self, data: Any) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        This method handles the common forward pass logic and delegates
        the specific implementation to _forward.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary mapping output names to their predictions
        """
        batch = data.batch
        self.device = data.pos.device
        self.num_atoms = data.atomic_numbers.shape[0]
        self.num_systems = data.natoms.shape[0]

        # Call declared model forward pass
        out = self._forward(data)

        results = {}

        for target in self.output_targets:
            module_outputs = out["node_embedding"].unsqueeze(1)

            # For models that directly return desired property, add directly
            if target not in self.module_dict:
                pred = out[target]
                # Squeeze if necessary
                if len(pred.shape) > 1:
                    pred = pred.squeeze(dim=1)
                results[target] = pred
                continue

            # Equivariant prediction
            if "irrep_dim" in self.output_targets[target]:
                pred = self.forward_irrep(out, target)
            # Scalar prediction
            else:
                pred = self.module_dict[target](out["node_embedding"])

            # (batch, output_shape)
            if self.output_targets[target].get("level", "system") == "system":
                if (
                    self.output_targets[target].get("aggregate", "add")
                    == "mean"
                ):
                    pred = scatter_det(
                        pred,
                        batch,
                        dim=0,
                        dim_size=self.num_systems,
                        reduce=self.output_targets[target].get(
                            "reduce", "mean"
                        ),
                    )
                else:
                    pred = scatter_det(
                        pred,
                        batch,
                        dim=0,
                        dim_size=self.num_systems,
                        reduce=self.output_targets[target].get(
                            "reduce", "add"
                        ),
                    )

            results[target] = pred.squeeze(1)

        self.construct_parent_tensor(results)            

        return results, module_outputs

    def forward_irrep(self, out, target):
        """
        For equivariant properties, make use of spherical harmonics to ensure
        SO(3) equivariance.
        """
        irrep = self.output_targets[target]["irrep_dim"]

        ### leverage spherical harmonic embeddings directly
        if self.output_targets[target].get("use_sphere_s2", False):
            assert "sphere_values" in out
            assert "sphere_points" in out

            # (sphere_points, num_channels)
            sphere_values = out["sphere_values"]
            # (sphere_sample, 3)
            sphere_points = out["sphere_points"]
            num_sphere_samples = sphere_points.shape[0]

            # (sphere_sample, 2*l+1)
            sphharm = o3.spherical_harmonics(
                irrep, sphere_points, True
            ).detach()

            # (sphere_sample, 1)
            pred = self.module_dict[target](sphere_values)
            # (nnodes, num_sphere_samples, 1)
            pred = pred.view(-1, num_sphere_samples, 1)
            # (nnodes, num_sphere_samples, 2*l+1)
            pred = pred * sphharm
            pred = pred.sum(dim=1) / num_sphere_samples

        ### Compute spherical harmonics based on edge vectors
        else:
            assert "edge_vec" in out
            assert "edge_idx" in out
            assert "edge_embedding" in out

            edge_vec = out["edge_vec"]
            edge_idx = out["edge_idx"]

            # (nedges, (2*irrep_dim+1))
            if self.output_targets[target].get("use_raw_edge_vecs", False):
                # Because `edge_vec` contains normalized direction vectors,
                # `o3.spherical_harmonics(edge_vec)` will just return a
                # scaled up version of the edge_vecs. To avoid this, we
                # just use the raw edge_vecs.
                # This is used to reproduce the results of the original
                # FM model.
                assert (
                    self.output_targets[target]["irrep_dim"] == 1
                ), "Only irrep_dim=1 is supported when use_raw_edge_vecs=True"
                sphharm = edge_vec
            else:
                sphharm = o3.spherical_harmonics(
                    irrep, edge_vec, True
                ).detach()
            # (nedges, 1)
            pred = self.module_dict[target](out["edge_embedding"])
            # (nedges, 2*irrep-dim+1)
            pred = pred * sphharm

            # aggregate edges per node
            # (nnodes, 2*irrep-dim+1)
            pred = scatter_det(
                pred, edge_idx, dim=0, dim_size=self.num_atoms, reduce="add"
            )

        return pred

    def construct_parent_tensor(self, results):
        parent_construction = defaultdict(dict)

        # Identify target properties that are decomposition of parent property
        for target in self.output_targets:
            if "parent" in self.output_targets[target]:
                parent_target = self.output_targets[target]["parent"]
                irrep_dim = self.output_targets[target]["irrep_dim"]
                parent_construction[parent_target][irrep_dim] = target

        # Construct parent tensors from predicted irreps
        for parent_target in parent_construction:
            rank = max(parent_construction[parent_target].keys())
            cg_matrix = cg_decomp_mat(rank, self.device)

            # handle per-atom vs per-system properties
            prediction_irreps = torch.zeros(
                (self.num_systems, irreps_sum(rank)), device=self.device
            )

            # Rank 2 support
            for irrep in range(rank + 1):
                if irrep in parent_construction[parent_target]:
                    # (batch, 2*irrep+1)
                    prediction_irreps[
                        :, max(0, irreps_sum(irrep - 1)) : irreps_sum(irrep)
                    ] = results[
                        parent_construction[parent_target][irrep]
                    ].view(
                        -1, 2 * irrep + 1
                    )

            # AMP will return this as a float-16 tensor
            parent_prediction = torch.mm(prediction_irreps, cg_matrix)

            results[parent_target] = parent_prediction

    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
        enforce_max_neighbors_strictly=None,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

        if enforce_max_neighbors_strictly is not None:
            pass
        elif hasattr(self, "enforce_max_neighbors_strictly"):
            # Not all models will have this attribute
            enforce_max_neighbors_strictly = (
                self.enforce_max_neighbors_strictly
            )
        else:
            # Default to old behavior
            enforce_max_neighbors_strictly = True

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data,
                    cutoff,
                    max_neighbors,
                    enforce_max_neighbors_strictly,
                )

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=data.pos.device
            )
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_rep(self, data):
        batch = data.batch
        self.device = data.pos.device
        self.num_atoms = data.atomic_numbers.shape[0]
        self.num_systems = data.natoms.shape[0]

        # call declared model forward pass
        out = self._forward(data)

        backbone_feature = out["node_embedding"]
        print(backbone_feature.shape)

        return backbone_feature