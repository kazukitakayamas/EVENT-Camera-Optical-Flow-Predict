import random
import numpy as np
import torch
from enum import Enum, auto
from time import time
import math

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

class EventRepresentation:
    def __init__(self):
        pass

    def convert(self, events):
        raise NotImplementedError

class VoxelGrid(EventRepresentation):
    def __init__(self, input_size: tuple, normalize: bool):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros(
            (input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]
        self.normalize = normalize

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events['p'].device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = events['t']
            t_norm = (C - 1) * (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])

            x0 = events['x'].int()
            y0 = events['y'].int()
            t0 = t_norm.int()

            value = 2 * events['p'] - 1
            for xlim in [x0, x0 + 1]:
                for ylim in [y0, y0 + 1]:
                    for tlim in [t0, t0 + 1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (
                            ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        interp_weights = value * (1 - (xlim - events['x']).abs()) * (
                            1 - (ylim - events['y']).abs()) * (1 - (tlim - t_norm).abs())
                        index = H * W * tlim.long() + \
                            W * ylim.long() + \
                            xlim.long()

                        voxel_grid.put_(
                            index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

class PolarityCount(EventRepresentation):
    def __init__(self, input_size: tuple):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros(
            (input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events['p'].device)
            voxel_grid = self.voxel_grid.clone()

            x0 = events['x'].int()
            y0 = events['y'].int()

            for xlim in [x0, x0 + 1]:
                for ylim in [y0, y0 + 1]:
                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (
                        ylim >= 0)
                    interp_weights = (1 - (xlim - events['x']).abs()) * (
                        1 - (ylim - events['y']).abs())
                    index = H * W * events['p'].long() + \
                        W * ylim.long() + \
                        xlim.long()

                    voxel_grid.put_(
                        index[mask], interp_weights[mask], accumulate=True)

        return voxel_grid

def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    flow_16bit = flow_16bit.astype('float')

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (
        flow_16bit[valid_map[0], valid_map[1], 0] - 2 ** 15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (
        flow_16bit[valid_map[0], valid_map[1], 1] - 2 ** 15) / 128
    return flow_map, valid2D

class ZCAWhitening:
    def __init__(self, epsilon=1e-4, device="cuda"):
        self.epsilon = epsilon
        self.device = device

    def fit(self, images, batch_size=1000):
        num_images = len(images)
        x = images[0][0].reshape(1, -1)
        self.mean = torch.zeros([1, x.size()[1]]).to(self.device)
        con_matrix = torch.zeros([x.size()[1], x.size()[1]]).to(self.device)

        num_batches = math.ceil(num_images / batch_size)
        for b in range(num_batches):
            batch = images[b * batch_size: (b + 1) * batch_size]
            x_batch = torch.cat([img[0].reshape(1, -1) for img in batch]).to(self.device)
            self.mean += x_batch.mean(dim=0, keepdim=True) / num_batches
            con_matrix += torch.mm(x_batch.t(), x_batch) / num_batches
            print(f"Processed batch {b + 1}/{num_batches}")

        self.mean /= num_batches
        self.E, self.V = torch.linalg.eigh(con_matrix)
        self.E = torch.max(self.E, torch.zeros_like(self.E))
        self.ZCA_matrix = torch.mm(torch.mm(self.V, torch.diag((self.E.squeeze() + self.epsilon) ** -0.5)), self.V.t())
        print("completed!")

    def __call__(self, x):
        size = x.size()
        x = x.reshape(1, -1).to(self.device)
        x -= self.mean
        x = torch.mm(x, self.ZCA_matrix.t())
        x = x.reshape(tuple(size))
        x = x.to("cpu")
        return x
