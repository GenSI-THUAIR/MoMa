from torch_geometric.data import Data

from ocpmodels.models.gemnet_oc_mt.goc_graph import Cutoffs, MaxNeighbors
from ocpmodels.trainers.mt.config import TransformConfigs
from ocpmodels.trainers.mt.transform import (
    _common_transform,
    _common_transform_all,
    _generate_graphs,
)


def md17_transform(data: Data, *, config: TransformConfigs, training: bool):
    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(12.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
        pbc=False,
        training=training,
    )
    data = _common_transform(data)
    data = _common_transform_all(data)
    return data


def qm9_transform(data: Data, *, config: TransformConfigs, training: bool):
    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(12.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
        pbc=False,
        training=training,
    )
    data = _common_transform(data)
    data = _common_transform_all(data)
    return data


def md22_transform(data: Data, *, config: TransformConfigs, training: bool):
    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(12.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
        pbc=False,
        training=training,
    )
    data = _common_transform(data)
    data = _common_transform_all(data)
    return data


def spice_transform(data: Data, *, config: TransformConfigs, training: bool):
    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(12.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(30),
        pbc=False,
        training=training,
    )
    data = _common_transform(data)
    data = _common_transform_all(data)
    return data


def matbench_transform(
    data: Data, *, config: TransformConfigs, training: bool
):
    # Decrease max neighbors for very large systems to avoid memory issues.
    if data.natoms > 300:
        max_neighbors = 5
    elif data.natoms > 200:
        max_neighbors = 10
    else:
        max_neighbors = 30

    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(12.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors),
        pbc=True,
        training=training,
    )
    data = _common_transform(data)
    data = _common_transform_all(data)
    return data


def qmof_transform(data: Data, *, config: TransformConfigs, training: bool):
    # Decrease max neighbors for very large systems to avoid memory issues.
    if data.natoms > 300:
        max_neighbors = 5
    elif data.natoms > 200:
        max_neighbors = 10
    else:
        max_neighbors = 30

    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(12.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors),
        pbc=True,
        training=training,
    )
    data = _common_transform(data)
    data = _common_transform_all(data)
    return data
