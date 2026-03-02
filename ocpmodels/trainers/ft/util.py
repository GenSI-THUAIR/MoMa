import fnmatch
from logging import getLogger
from typing import TYPE_CHECKING, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

if TYPE_CHECKING:
    from torch.nn.modules.module import _IncompatibleKeys


log = getLogger(__name__)


def _resolve_scale_factor_submodule(model: nn.Module, name: str):
    from ocpmodels.trainers.mt.scaling.scale_factor import ScaleFactor

    try:
        scale = model.get_submodule(name)
        if not isinstance(scale, ScaleFactor):
            return None
        return scale
    except AttributeError:
        return None


def _report_incompat_keys(
    model: nn.Module,
    keys: "_IncompatibleKeys",
    ignore_keys_patterns: list[str],
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    # filter out the missing scale factor keys for the new scaling factor module
    missing_keys: list[str] = []
    # for full_key_name in keys.missing_keys:
    #     parent_module_name, _ = full_key_name.rsplit(".", 1)
    #     scale_factor = _resolve_scale_factor_submodule(
    #         model, parent_module_name
    #     )
    #     if scale_factor is not None:
    #         continue

    #     if (
    #         pattern := next(
    #             (
    #                 fnmatch.fnmatch(full_key_name, p)
    #                 for p in ignore_keys_patterns
    #             ),
    #             None,
    #         )
    #     ) is not None:
    #         log.info(f"Ignoring missing key {full_key_name} due to {pattern}")

    #     missing_keys.append(full_key_name)

    # revised by GPT-4
    for full_key_name in keys.missing_keys:
        parent_module_name, _ = full_key_name.rsplit(".", 1)
        scale_factor = _resolve_scale_factor_submodule(
            model, parent_module_name
        )
        if scale_factor is not None:
            continue

        matched_pattern = None
        for p in ignore_keys_patterns:
            if fnmatch.fnmatch(full_key_name, p):
                matched_pattern = p
                break

        if matched_pattern is not None:
            log.info(
                f"Ignoring missing key {full_key_name} due to pattern: {matched_pattern}"
            )
            continue

        missing_keys.append(full_key_name)

    # filter out unexpected scale factor keys that remain from the old scaling modules
    unexpected_keys: list[str] = []
    for full_key_name in keys.unexpected_keys:
        parent_module_name, _ = full_key_name.rsplit(".", 1)
        scale_factor = _resolve_scale_factor_submodule(
            model, parent_module_name
        )
        if scale_factor is not None:
            continue
        unexpected_keys.append(full_key_name)

    error_msgs = []
    if len(unexpected_keys) > 0:
        error_msgs.insert(
            0,
            "Unexpected key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in unexpected_keys)
            ),
        )
    if len(missing_keys) > 0:
        error_msgs.insert(
            0,
            "Missing key(s) in state_dict: {}. ".format(
                ", ".join('"{}"'.format(k) for k in missing_keys)
            ),
        )

    if len(error_msgs) > 0:
        error_msg = "Error(s) in loading state_dict for {}:\n\t{}".format(
            model.__class__.__name__, "\n\t".join(error_msgs)
        )
        if strict:
            raise RuntimeError(error_msg)
        else:
            # log.warning(error_msg)
            log.warning(error_msg[:50])

    return missing_keys, unexpected_keys


def load_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    ignore_keys_patterns: list[str] = [],
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    ignore_keys_patterns.append("*module_dict.y*")
    print("ignore_keys_patterns: ", ignore_keys_patterns)
    updated_state_dict: dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        matched_pattern = None
        for p in ignore_keys_patterns:
            if fnmatch.fnmatch(k, p):
                matched_pattern = p
                break

        if matched_pattern is not None:
            log.info(
                f"Ignoring existing key {k} due to matching pattern: {matched_pattern}"
            )
            continue  # ignore this key

        updated_state_dict[k] = v

    incompat_keys = module.load_state_dict(updated_state_dict, strict=False)
    return _report_incompat_keys(
        module,
        incompat_keys,
        ignore_keys_patterns,
        # strict=strict,
        strict=False,
    )


def consistency(
    features, labels
):  # features: bsz * embedding_dim; labels: bsz * 1
    # print('labels',labels)
    reshaped_features = features.unsqueeze(1)
    reshaped_labels = labels.unsqueeze(1)

    # Compute the pairwise L2 distances
    dis_matrix_1 = torch.sqrt(
        torch.sum(
            (reshaped_features - reshaped_features.transpose(0, 1)) ** 2, dim=2
        )
    )
    dis_matrix_2 = torch.sqrt(
        torch.sum(
            (reshaped_labels - reshaped_labels.transpose(0, 1)) ** 2, dim=2
        )
    )

    # normalization
    dis_matrix_1 = F.normalize(dis_matrix_1, p=1, dim=1)
    dis_matrix_2 = F.normalize(dis_matrix_2, p=1, dim=1)

    # calculate correlation coefficient
    pearson_coefficients, spearmanr_coefficients = [], []

    for row1, row2 in zip(dis_matrix_1.cpu(), dis_matrix_2.cpu()):
        # print(row1, row2)
        p, _ = pearsonr(row1, row2)
        s, _ = spearmanr(row1, row2)
        pearson_coefficients.append(p)
        spearmanr_coefficients.append(s)

    pearson_coeff = sum(pearson_coefficients) / len(pearson_coefficients)
    spearmanr_coeff = sum(spearmanr_coefficients) / len(spearmanr_coefficients)

    # dis_matrix_1 is a l2 distance function
    sim_matrix = torch.exp(-dis_matrix_1 / 0.1) - torch.eye(
        dis_matrix_1.size(-1)
    ).to(dis_matrix_1.device)

    # normalization
    normed_sim_matrix = F.normalize(sim_matrix, p=1, dim=1)

    # predict by label propagation
    prediction = normed_sim_matrix @ labels

    # compute MAE
    _abs = torch.abs(prediction - labels)
    mean_abs = torch.mean(_abs)

    return pearson_coeff, spearmanr_coeff, mean_abs.cpu()


def Label_prop_cosine(
    features, labels
):  # features: bsz * embedding_dim; labels: bsz * 1

    # normalization for cosine similarity
    norms = torch.norm(features, dim=1, keepdim=True)
    normalized_features = features / norms

    # cosine similarity matrix  bsz * bsz
    cos_matrix = normalized_features @ normalized_features.transpose(0, 1)
    # print("cos matrix", cos_matrix)

    # Gaussion and leave one out
    adjacent_matrix = torch.exp(cos_matrix / 0.2)
    adjacent_matrix = adjacent_matrix - torch.diag(torch.diag(adjacent_matrix))

    # select top_k neighbors for prediction
    topk_value, topk_idx = torch.topk(adjacent_matrix, k=10, dim=1)
    # print("topk idx", topk_idx)
    mask = torch.zeros_like(adjacent_matrix)
    mask = mask.scatter(1, topk_idx, 1)
    # mask = ((mask + torch.t(mask))>0).type(torch.float32)
    adjacent_matrix = adjacent_matrix * mask

    # row normalization
    D = torch.sum(adjacent_matrix, 1, True)
    # print("D", D)
    adjacent_matrix = adjacent_matrix / D
    # print("normalized adj matrix", adjacent_matrix)

    # label propagation
    prediction = adjacent_matrix @ labels
    # print("predictions of label prop", prediction)

    return prediction


def Label_prop_s2q(
    support, query, support_labels
):  # features: bsz * embedding_dim; labels: bsz * 1

    # normalization for cosine similarity
    norm_s = torch.norm(support, dim=1, keepdim=True)
    support = support / norm_s
    norm_q = torch.norm(query, dim=1, keepdim=True)
    query = query / norm_q

    # cosine similarity matrix  bsz * bsz
    cos_matrix = query @ support.transpose(0, 1)

    # Gaussion and leave one out
    adjacent_matrix = torch.exp(cos_matrix / 0.2)

    # select top_k neighbors for prediction
    _, topk_idx = torch.topk(adjacent_matrix, k=5, dim=1)
    mask = torch.zeros_like(adjacent_matrix)
    mask = mask.scatter(1, topk_idx, 1)
    # mask = ((mask + torch.t(mask))>0).type(torch.float32)
    adjacent_matrix = adjacent_matrix * mask

    # row normalization
    D = torch.sum(adjacent_matrix, 1, True)
    adjacent_matrix = adjacent_matrix / D
    # print("normalized adj matrix", adjacent_matrix)

    # label propagation
    prediction = adjacent_matrix @ support_labels
    # print("predictions of label prop", prediction)

    return prediction

import copy


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.best_model = copy.deepcopy(model.state_dict())

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None        