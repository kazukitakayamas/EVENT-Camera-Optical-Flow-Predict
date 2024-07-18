import math
from pathlib import Path, PurePath
from typing import Dict, Tuple
import cv2
import hdf5plugin
import h5py
import numpy as np
import os
import imageio
imageio.plugins.freeimage.download()
import imageio.v3 as iio
import torch
import torch.utils.data
from torch.utils.data import Dataset
from src.utils import RepresentationType, VoxelGrid, flow_16bit_to_float
import scipy
import scipy.optimize
from concurrent.futures import ThreadPoolExecutor

# FlowPatchクラスを追加
class FlowPatch:
    def __init__(self, x, y, shape, u=0.0, v=0.0):
        self.x = x
        self.y = y
        self.shape = shape
        self.u = u
        self.v = v

class ZCAWhitening():
    def __init__(self, epsilon=1e-4, device="cuda"):  # 計算が重いのでGPUを用いる
        self.epsilon = epsilon
        self.device = device

    def fit(self, images):  # 変換行列と平均をデータから計算
        x = images[0][0].reshape(1, -1)
        self.mean = torch.zeros([1, x.size()[1]]).to(self.device)
        con_matrix = torch.zeros([x.size()[1], x.size()[1]]).to(self.device)
        for i in range(len(images)):  # 各データについての平均を取る
            x = images[i][0].reshape(1, -1).to(self.device)
            self.mean += x / len(images)
            con_matrix += torch.mm(x.t(), x) / len(images)
            if i % 10000 == 0:
                print("{0}/{1}".format(i, len(images)))
        self.E, self.V = torch.linalg.eigh(con_matrix)  # 固有値分解
        self.E = torch.max(self.E, torch.zeros_like(self.E)) # 誤差の影響で負になるのを防ぐ
        self.ZCA_matrix = torch.mm(torch.mm(self.V, torch.diag((self.epsilon + self.E)**(-0.5))), self.V.t())
        print("completed!")

    def __call__(self, x):
        size = x.size()
        x = x.reshape(1, -1).to(self.device)
        x -= self.mean
        x = torch.mm(x, self.ZCA_matrix.t())
        x = x.reshape(size)
        return x.to("cpu")

class EventBasedNoiseFilter:
    def __init__(self, threshold=0.5, neighborhood_size=5, device="cuda"):
        self.threshold = threshold
        self.neighborhood_size = neighborhood_size
        self.device = device

    def __call__(self, x, y, t, p):
        valid_indices = torch.ones_like(x, dtype=torch.bool, device=self.device)
        for i in range(x.size(0)):
            neighborhood = (
                (x >= x[i] - self.neighborhood_size // 2) & (x <= x[i] + self.neighborhood_size // 2) &
                (y >= y[i] - self.neighborhood_size // 2) & (y <= y[i] + self.neighborhood_size // 2) &
                (t >= t[i] - 500) & (t <= t[i] + 500)
            )
            if torch.mean(p[neighborhood].float()) < self.threshold:
                valid_indices[i] = False
        return x[valid_indices], y[valid_indices], t[valid_indices], p[valid_indices]

class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f
        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f[f'events/{dset_str}']
        self.ms_to_idx = torch.tensor(np.asarray(self.h5f['ms_to_idx'], dtype='int64')).to('cuda')
        self.t_offset = int(h5f['t_offset'][()])
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, torch.Tensor]:
        assert t_start_us < t_end_us
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset
        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)
        if t_start_ms_idx is None or t_end_ms_idx is None:
            print('Error', 'start', t_start_us, 'end', t_end_us)
            return None

        events = dict()
        time_array_conservative = torch.tensor(np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx]), dtype=torch.int64).to('cuda')
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = torch.tensor(np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx]), dtype=torch.int64).to('cuda')
            assert events[dset_str].size(0) == events['t'].size(0)
        return events

    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us / 1000)
        window_end_ms = math.ceil(ts_end_us / 1000)
        return window_start_ms, window_end_ms

    @staticmethod
    def get_time_indices_offsets(time_array: torch.Tensor, time_start_us: int, time_end_us: int) -> Tuple[int, int]:
        assert time_array.ndim == 1
        idx_start = -1
        if time_array[-1] < time_start_us:
            return time_array.size(0), time_array.size(0)
        else:
            for idx_from_start in range(0, time_array.size(0), 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size(0)
        for idx_from_end in range(time_array.size(0) - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size(0):
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size(0):
            return None
        return self.ms_to_idx[time_ms].item()

class SparseFilter:
    def __init__(self, threshold=0.1, device="cuda"):
        self.threshold = threshold
        self.device = device

    def __call__(self, event_volume):
        return event_volume * (event_volume.abs() > self.threshold).float()

class MixedPatchContrastMaximization:
    def __init__(self, image_shape, calibration_parameter, solver_config={}, optimizer_config={}, output_config={}, visualize_module=None):
        self.image_shape = image_shape
        self.calibration_parameter = calibration_parameter
        self.solver_config = solver_config
        self.optimizer_config = optimizer_config
        self.output_config = output_config
        self.visualize_module = visualize_module
        self.n_patch = 0
        self.patch_image_size = (0, 0)
        self.patches = {}
        self.set_patch_size_and_sliding_window()

    def set_patch_size_and_sliding_window(self):
        if isinstance(self.solver_config["patch"]["size"], int):
            self.patch_size = (self.solver_config["patch"]["size"], self.solver_config["patch"]["size"])
        elif isinstance(self.solver_config["patch"]["size"], list):
            self.patch_size = tuple(self.solver_config["patch"]["size"])
        else:
            raise TypeError("Unsupported type for patch size.")

        if isinstance(self.solver_config["patch"]["sliding_window"], int):
            self.sliding_window = (self.solver_config["patch"]["sliding_window"], self.solver_config["patch"]["sliding_window"])
        elif isinstance(self.solver_config["patch"]["sliding_window"], list):
            self.sliding_window = tuple(self.solver_config["patch"]["sliding_window"])
        else:
            raise TypeError("Unsupported type for sliding window.")

    def prepare_patch(self, image_size: tuple, patch_size: tuple, sliding_window: tuple) -> Tuple[Dict[int, FlowPatch], tuple]:
        image_h, image_w = image_size
        patch_h, patch_w = patch_size
        slide_h, slide_w = sliding_window
        center_x = np.arange(0, image_h - patch_h + slide_h, slide_h) + patch_h / 2
        center_y = np.arange(0, image_w - patch_w + slide_w, slide_w) + patch_w / 2
        xx, yy = np.meshgrid(center_x, center_y)
        patch_shape = xx.T.shape
        xx = xx.T.reshape(-1)
        yy = yy.T.reshape(-1)
        patches = {i: FlowPatch(x=xx[i], y=yy[i], shape=patch_size, u=0.0, v=0.0) for i in range(0, len(xx))}
        return patches, patch_shape

    def optimize(self, events: np.ndarray) -> np.ndarray:
        logger.info("Start optimization.")
        logger.info(f"DoF is {self.motion_vector_size * self.n_patch}")

        if self.opt_method == "optuna":
            opt_result = self.run_optuna(events)
            logger.info("End optimization.")
            best_motion = self.get_motion_array_optuna(opt_result.best_params)
        elif self.opt_method in SCIPY_OPTIMIZERS:
            opt_result = self.run_scipy(events)
            logger.info(f"End optimization.\n Best parameters: {opt_result}")
            best_motion = opt_result.x.reshape((self.motion_vector_size,) + self.patch_image_size)

        logger.info("Profile file saved.")
        if self.visualize_module:
            shutil.copy("optimize.prof", self.visualize_module.save_dir)
            if self.opt_method in SCIPY_OPTIMIZERS:
                self.visualize_module.visualize_scipy_history(self.cost_func.get_history(), self.cost_weight)

        logger.info(f"{best_motion}")
        return best_motion

    # Optuna functions
    def objective(self, trial, events: np.ndarray):
        params = {k: self.sampling(trial, k) for k in self._patch_motion_model_keys}
        motion_array = self.get_motion_array_optuna(params)
        if self.normalize_t_in_batch:
            t_scale = np.max(events[:, 2]) - np.min(events[:, 2])
            motion_array *= t_scale
        dense_flow = self.motion_to_dense_flow(motion_array)
        loss = self.calculate_cost(events, dense_flow, self.motion_model_for_dense_warp)
        logger.info(f"{trial.number = } / {loss = }")
        return loss

    def sampling(self, trial, key: str):
        key_suffix = key[key.find("_") + 1:]
        return trial.suggest_uniform(key, self.opt_config["parameters"][key_suffix]["min"], self.opt_config["parameters"][key_suffix]["max"])

    def get_motion_array_optuna(self, params: dict) -> np.ndarray:
        motion_array = np.zeros((self.motion_vector_size, self.n_patch))
        for i in range(self.n_patch):
            param = {k: params[f"patch{i}_{k}"] for k in self.motion_model_keys}
            motion_array[:, i] = self.motion_model_to_motion(param)
        return motion_array.reshape((self.motion_vector_size,) + self.patch_image_size)

    # Scipy
    def run_scipy(self, events: np.ndarray) -> scipy.optimize.OptimizeResult:
        if self.previous_frame_best_estimation is not None:
            motion0 = np.copy(self.previous_frame_best_estimation)
        else:
            if self.solver_config["patch"]["initialize"] == "random":
                motion0 = self.initialize_random()
            elif self.solver_config["patch"]["initialize"] == "zero":
                motion0 = self.initialize_zeros()
            elif self.solver_config["patch"]["initialize"] == "global-best":
                logger.info("sampling initialization")
                best_guess = self.initialize_guess_from_whole_image(events)
                if isinstance(best_guess, torch.Tensor):
                    motion0 = torch.tile(best_guess[None], (self.n_patch, 1)).T.reshape(-1)
                elif isinstance(best_guess, np.ndarray):
                    motion0 = np.tile(best_guess[None], (self.n_patch, 1)).T.reshape(-1)
            elif self.solver_config["patch"]["initialize"] == "grid-best":
                logger.info("sampling initialization")
                best_guess = self.initialize_guess_from_patch(events, patch_index=self.n_patch // 2 - 1)
                if isinstance(best_guess, torch.Tensor):
                    motion0 = torch.tile(best_guess[None], (self.n_patch, 1)).T.reshape(-1)
                elif isinstance(best_guess, np.ndarray):
                    motion0 = np.tile(best_guess[None], (self.n_patch, 1)).T.reshape(-1)
            elif self.solver_config["patch"]["initialize"] == "optuna-sampling":
                logger.info("Optuna intelligent sampling initialization")
                motion0 = self.initialize_guess_from_optuna_sampling(events)
            self.cost_func.clear_history()

        self.events = torch.from_numpy(events).double().requires_grad_().to(self._device)
        result = scipy.optimize.minimize(self.objective_scipy, motion0, method=self.opt_method, options={"gtol": 1e-7, "disp": True, "maxiter": self.opt_config["max_iter"], "eps": 0.01}, precision="float64", torch_device=self._device)
        return result

    def objective_scipy(self, motion_array: np.ndarray, suppress_log: bool = False):
        if self.normalize_t_in_batch:
            t_scale = self.events[:, 2].max() - self.events[:, 2].min()
        else:
            t_scale = 1.0

        events = self.events.clone()
        dense_flow = self.motion_to_dense_flow(motion_array * t_scale)
        loss = self.calculate_cost(events, dense_flow, self.motion_model_for_dense_warp, motion_array.reshape((self.motion_vector_size,) + self.patch_image_size))
        if not suppress_log:
            logger.info(f"{loss = }")
        return loss

    def motion_to_dense_flow(self, motion_array):
        if isinstance(motion_array, np.ndarray):
            return self.interpolate_dense_flow_from_patch_numpy(motion_array)
        elif isinstance(motion_array, torch.Tensor):
            return self.interpolate_dense_flow_from_patch_tensor(motion_array)
        raise TypeError(f"Unsupported type: {type(motion_array)}")

    def interpolate_dense_flow_from_patch_numpy(self, motion_array: np.ndarray) -> np.ndarray:
        pad_h = int(self.patch_size[0] / 2 // self.sliding_window[0]) + self.patch_shift[0] // self.sliding_window[0] + 1
        pad_w = int(self.patch_size[1] / 2 // self.sliding_window[1]) + self.patch_shift[1] // self.sliding_window[1] + 1
        flow_array = np.pad(-motion_array.reshape((self.motion_vector_size,) + self.patch_image_size), ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="edge")

        interp = cv2.INTER_LINEAR if self.filter_type == "bilinear" else cv2.INTER_NEAREST
        upscaled_u = cv2.resize(flow_array[0], None, None, fx=self.sliding_window[1], fy=self.sliding_window[0], interpolation=interp)
        upscaled_v = cv2.resize(flow_array[1], None, None, fx=self.sliding_window[1], fy=self.sliding_window[0], interpolation=interp)
        dense_flow = np.concatenate([upscaled_u[None, ...], upscaled_v[None, ...]], axis=0)
        cx, cy = dense_flow.shape[1] // 2, dense_flow.shape[2] // 2
        h1 = cx - self.image_shape[0] // 2
        w1 = cy - self.image_shape[1] // 2
        h2 = h1 + self.image_shape[0]
        w2 = w1 + self.image_shape[1]
        return dense_flow[..., h1:h2, w1:w2]

    def interpolate_dense_flow_from_patch_tensor(self, motion_array: torch.Tensor) -> torch.Tensor:
        pad_h = int(self.patch_size[0] / 2 // self.sliding_window[0]) + self.patch_shift[0] // self.sliding_window[0] + 1
        pad_w = int(self.patch_size[1] / 2 // self.sliding_window[1]) + self.patch_shift[1] // self.sliding_window[1] + 1
        flow_array = torch.nn.functional.pad(-motion_array.reshape((1, self.motion_vector_size,) + self.patch_image_size), (pad_w, pad_w, pad_h, pad_h), mode="replicate")[0]

        interp = transforms.InterpolationMode.BILINEAR if self.filter_type == "bilinear" else transforms.InterpolationMode.NEAREST
        size = [flow_array.shape[1] * self.sliding_window[0], flow_array.shape[2] * self.sliding_window[1]]
        dense_flow = transforms.functional.resize(flow_array, size, interpolation=interp)
        cx, cy = dense_flow.shape[1] // 2, dense_flow.shape[2] // 2
        h1 = cx - self.image_shape[0] // 2
        w1 = cy - self.image_shape[1] // 2
        h2 = h1 + self.image_shape[0]
        w2 = w1 + self.image_shape[1]
        return dense_flow[..., h1:h2, w1:w2]

class Sequence(Dataset):
    def __init__(self, seq_path: Path, representation_type: RepresentationType, mode: str = 'test', delta_t_ms: int = 100,
                 num_bins: int = 4, transforms=[], name_idx=0, visualize=False, load_gt=False, zca=None, ebn_filter=None, sparse_filter=None):
        assert num_bins >= 1
        assert delta_t_ms == 100
        assert seq_path.is_dir()
        assert mode in {'train', 'test'}
        assert representation_type is not None
        self.seq_name = PurePath(seq_path).name
        self.mode = mode
        self.name_idx = name_idx
        self.visualize_samples = visualize
        self.load_gt = load_gt
        self.transforms = []
        self.zca = zca
        self.ebn_filter = ebn_filter
        self.sparse_filter = sparse_filter
        if self.mode == "test":
            assert not load_gt
            ev_dir_location = seq_path / 'events_left'
            timestamp_file = seq_path / 'forward_timestamps.txt'
            flow_path = seq_path / 'flow_forward'
            timestamps_flow = np.loadtxt(timestamp_file, delimiter=',', dtype='int64')
            self.indices = np.arange(len(timestamps_flow))
            self.timestamps_flow = timestamps_flow[:, 0]

        elif self.mode == "train":
            ev_dir_location = seq_path / 'events_left'
            flow_path = seq_path / 'flow_forward'
            timestamp_file = seq_path / 'forward_timestamps.txt'
            self.flow_png = [Path(os.path.join(flow_path, img)) for img in sorted(os.listdir(flow_path))]
            timestamps_flow = np.loadtxt(timestamp_file, delimiter=',', dtype='int64')
            self.indices = np.arange(len(timestamps_flow))
            self.timestamps_flow = timestamps_flow[:, 0]
        else:
            pass
        assert timestamp_file.is_file()

        file = np.genfromtxt(timestamp_file, delimiter=',')

        self.idx_to_visualize = file[:, 2] if file.shape[1] == 3 else []

        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        self.voxel_grid = VoxelGrid((self.num_bins, self.height, self.width), normalize=True)
        self.delta_t_us = delta_t_ms * 1000

        ev_data_file = ev_dir_location / 'events.h5'
        ev_rect_file = ev_dir_location / 'rectify_map.h5'

        h5f_location = h5py.File(str(ev_data_file), 'r')
        self.h5f = h5f_location
        self.event_slicer = EventSlicer(h5f_location)

        self.h5rect = h5py.File(str(ev_rect_file), 'r')
        self.rectify_ev_map = self.h5rect['rectify_map'][()]

    def events_to_voxel_grid(self, p, t, x, y, device: str = 'cpu'):
        t = (t - t[0]).float()
        t = (t / t[-1])
        x = x.float()
        y = y.float()
        pol = p.float()

        event_data_torch = {
            'p': pol.to(device),
            't': t.to(device),
            'x': x.to(device),
            'y': y.to(device),
        }

        voxel_grid = self.voxel_grid.convert(event_data_torch)

        if self.zca:
            voxel_grid = self.zca(voxel_grid)
        if self.sparse_filter:
            voxel_grid = self.sparse_filter(voxel_grid)

        if device == 'cpu':
            voxel_grid = voxel_grid.to('cpu')

        return voxel_grid

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32') / 256

    @staticmethod
    def load_flow(flowfile: Path):
        assert flowfile.exists()
        assert flowfile.suffix == '.png'
        flow_16bit = iio.imread(str(flowfile), plugin='PNG-FI')
        flow, valid2D = flow_16bit_to_float(flow_16bit)
        return flow, valid2D

    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def get_image_width_height(self):
        return self.height, self.width

    def __len__(self):
        return len(self.timestamps_flow)

    def rectify_events(self, x: torch.Tensor, y: torch.Tensor):
        rectify_map = torch.tensor(self.rectify_ev_map, device=x.device)
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y.long(), x.long()]

    def get_data(self, index) -> Dict[str, any]:
        ts_start = self.timestamps_flow[index] - self.delta_t_us
        ts_end = self.timestamps_flow[index]

        file_index = self.indices[index]

        output = {
            'file_index': file_index,
            'timestamp': self.timestamps_flow[index],
            'seq_name': self.seq_name
        }
        output['save_submission'] = file_index in self.idx_to_visualize
        output['visualize'] = self.visualize_samples
        event_data = self.event_slicer.get_events(ts_start, ts_end)

        # Event-based Noise Filteringの適用
        if self.ebn_filter:
            event_data['x'], event_data['y'], event_data['t'], event_data['p'] = self.ebn_filter(
                event_data['x'], event_data['y'], event_data['t'], event_data['p']
            )

        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']

        xy_rect = self.rectify_events(x, y)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        if self.voxel_grid is None:
            raise NotImplementedError
        else:
            event_representation = self.events_to_voxel_grid(p, t, x_rect, y_rect)
            if self.zca:
                event_representation = self.zca(event_representation)
            output['event_volume'] = event_representation
        output['name_map'] = self.name_idx

        if self.load_gt:
            output['flow_gt'] = [torch.tensor(x) for x in self.load_flow(self.flow_png[index])]
            output['flow_gt'][0] = torch.moveaxis(output['flow_gt'][0], -1, 0)
            output['flow_gt'][1] = torch.unsqueeze(output['flow_gt'][1], 0)
        return output

    def __getitem__(self, idx):
        sample = self.get_data(idx)
        return sample

    def get_voxel_grid(self, idx):
        if idx == 0:
            event_data = self.event_slicer.get_events(self.timestamps_flow[0] - self.delta_t_us, self.timestamps_flow[0])
        elif idx > 0 and idx <= self.__len__():
            event_data = self.event_slicer.get_events(self.timestamps_flow[idx-1], self.timestamps_flow[idx-1] + self.delta_t_us)
        else:
            raise IndexError

        p = event_data['p']
        t = event_data['t']
        x = event_data['x']
        y = event_data['y']

        xy_rect = self.rectify_events(x, y)
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]
        return self.events_to_voxel_grid(p, t, x_rect, y_rect)

    def get_event_count_image(self, ts_start, ts_end, num_bins, normalize=True):
        assert ts_end > ts_start
        delta_t_bin = (ts_end - ts_start) / num_bins
        ts_start_bin = torch.linspace(ts_start, ts_end, steps=num_bins, device='cuda')
        ts_end_bin = ts_start_bin + delta_t_bin
        assert abs(ts_end_bin[-1] - ts_end) < 10.
        ts_end_bin[-1] = ts_end

        event_count = torch.zeros((num_bins, self.height, self.width), dtype=torch.float, device='cuda', requires_grad=False)

        for i in range(num_bins):
            event_data = self.event_slicer.get_events(ts_start_bin[i].item(), ts_end_bin[i].item())
            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            t = (t - t[0]).float()
            t = (t / t[-1])
            x = x.float()
            y = y.float()
            pol = p.float()
            event_data_torch = {
                'p': pol,
                't': t,
                'x': x,
                'y': y,
            }
            x = event_data_torch['x']
            y = event_data_torch['y']
            xy_rect = self.rectify_events(x.int(), y.int())
            x_rect = xy_rect[:, 0].long()
            y_rect = xy_rect[:, 1].long()
            value = 2 * event_data_torch['p'] - 1
            index = self.width * y_rect + x_rect
            mask = (x_rect < self.width) & (y_rect < self.height)
            event_count[i].put_(index[mask], value[mask], accumulate=True)

        return event_count

    @staticmethod
    def normalize_tensor(event_count):
        mask = torch.nonzero(event_count, as_tuple=True)
        if mask[0].size(0) > 0:
            mean = event_count[mask].mean()
            std = event_count[mask].std()
            if std > 0:
                event_count[mask] = (event_count[mask] - mean) / std
            else:
                event_count[mask] = event_count[mask] - mean
        return event_count

class SequenceRecurrent(Sequence):
    def __init__(self, seq_path: Path, representation_type: RepresentationType, mode: str = 'test', delta_t_ms: int = 100,
                 num_bins: int = 15, transforms=None, sequence_length=1, name_idx=0, visualize=False, load_gt=False, zca=None, ebn_filter=None, sparse_filter=None):
        super(SequenceRecurrent, self).__init__(seq_path, representation_type, mode, delta_t_ms, transforms=transforms,
                                                name_idx=name_idx, visualize=visualize, load_gt=load_gt, zca=zca, ebn_filter=ebn_filter, sparse_filter=sparse_filter)
        self.crop_size = self.transforms['randomcrop'] if 'randomcrop' in self.transforms else None
        self.sequence_length = sequence_length
        self.valid_indices = self.get_continuous_sequences()

    def get_continuous_sequences(self):
        continuous_seq_idcs = []
        if self.sequence_length > 1:
            for i in range(len(self.timestamps_flow) - self.sequence_length + 1):
                diff = self.timestamps_flow[i + self.sequence_length - 1] - self.timestamps_flow[i]
                if diff < np.max([100000 * (self.sequence_length - 1) + 1000, 101000]):
                    continuous_seq_idcs.append(i)
        else:
            for i in range(len(self.timestamps_flow) - 1):
                diff = self.timestamps_flow[i + 1] - self.timestamps_flow[i]
                if diff < np.max([100000 * (self.sequence_length - 1) + 1000, 101000]):
                    continuous_seq_idcs.append(i)
        return continuous_seq_idcs

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        assert idx >= 0
        assert idx < len(self)

        valid_idx = self.valid_indices[idx]
        sequence = []
        j = valid_idx

        ts_cur = self.timestamps_flow[j]
        sample = self.get_data_sample(j)
        sequence.append(sample)

        crop_window = None
        flip = None
        if 'crop_window' in sample.keys():
            crop_window = sample['crop_window']
        if 'flipped' in sample.keys():
            flip = sample['flipped']

        for i in range(self.sequence_length - 1):
            j += 1
            ts_old = ts_cur
            ts_cur = self.timestamps_flow[j]
            assert (ts_cur - ts_old < 100000 + 1000)
            sample = self.get_data_sample(j, crop_window=crop_window, flip=flip)
            sequence.append(sample)

        if idx == 0 or self.valid_indices[idx] - self.valid_indices[idx - 1] != 1:
            sequence[0]['new_sequence'] = 1
            print(f"Timestamp {self.timestamps_flow[self.valid_indices[idx]]} is the first one of the next seq!")
        else:
            sequence[0]['new_sequence'] = 0

        if self.crop_size is not None:
            i, j, h, w = RandomCrop.get_params(sample["event_volume_old"], output_size=self.crop_size)
            keys_to_crop = ["event_volume_old", "event_volume_new", "flow_gt_event_volume_old", "flow_gt_event_volume_new", "flow_gt_next"]

            for sample in sequence:
                for key, value in sample.items():
                    if key in keys_to_crop:
                        if isinstance(value, torch.Tensor):
                            sample[key] = tf.functional.crop(value, i, j, h, w)
                        elif isinstance(value, list) or isinstance(value, tuple):
                            sample[key] = [tf.functional.crop(v, i, j, h, w) for v in value]
        return sequence

class DatasetProvider:
    def __init__(self, dataset_path: Path, representation_type: RepresentationType, delta_t_ms: int = 100, num_bins=4,
                 config=None, visualize=False, zca=None, ebn_filter=None, sparse_filter=None):
        test_path = Path(os.path.join(dataset_path, 'test'))
        train_path = Path(os.path.join(dataset_path, 'train'))
        assert dataset_path.is_dir(), str(dataset_path)
        assert test_path.is_dir(), str(test_path)
        assert delta_t_ms == 100
        self.config = config
        self.name_mapper_test = []
        self.zca = zca
        self.ebn_filter = ebn_filter
        self.sparse_filter = sparse_filter

        test_sequences = list()
        for child in test_path.iterdir():
            self.name_mapper_test.append(str(child).split("/")[-1])
            test_sequences.append(Sequence(child, representation_type, 'test', delta_t_ms, num_bins,
                                           transforms=[], name_idx=len(self.name_mapper_test) - 1, visualize=visualize, zca=self.zca, ebn_filter=self.ebn_filter, sparse_filter=self.sparse_filter))

        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)

        available_seqs = os.listdir(train_path)
        seqs = available_seqs

        train_sequences = []
        for seq in seqs:
            extra_arg = dict()
            train_sequences.append(Sequence(Path(train_path) / seq, representation_type=representation_type, mode="train",
                                            load_gt=True, zca=self.zca, ebn_filter=self.ebn_filter, sparse_filter=self.sparse_filter, **extra_arg))
        self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)

    def get_test_dataset(self):
        return self.test_dataset

    def get_train_dataset(self):
        return self.train_dataset

    def get_name_mapping_test(self):
        return self.name_mapper_test

    def summary(self, logger):
        logger.write_line("================================== Dataloader Summary ====================================", True)
        logger.write_line("Loader Type:\t\t" + self.__class__.__name__, True)
        logger.write_line("Number of Voxel Bins: {}".format(self.test_dataset.datasets[0].num_bins), True)
        logger.write_line("Number of Train Sequences: {}".format(len(self.train_dataset)), True)

def train_collate(sample_list):
    batch = dict()
    for field_name in sample_list[0]:
        if field_name == 'timestamp':
            batch['timestamp'] = [sample[field_name] for sample in sample_list]
        if field_name == 'seq_name':
            batch['seq_name'] = [sample[field_name] for sample in sample_list]
        if field_name == 'new_sequence':
            batch['new_sequence'] = [sample[field_name] for sample in sample_list]
        if field_name.startswith("event_volume"):
            batch[field_name] = torch.stack([sample[field_name] for sample in sample_list])
        if field_name.startswith("flow_gt"):
            if all(field_name in x for x in sample_list):
                batch[field_name] = torch.stack([sample[field_name][0] for sample in sample_list])
                batch[field_name + '_valid_mask'] = torch.stack([sample[field_name][1] for sample in sample_list])
    return batch

def rec_train_collate(sample_list):
    seq_length = len(sample_list[0])
    seq_of_batch = []
    for i in range(seq_length):
        seq_of_batch.append(train_collate([sample[i] for sample in sample_list]))
    return seq_of_batch

def main():

    dataset_path = Path('/path/to/dataset')
    representation_type = RepresentationType.SOMETYPE
    delta_t_ms = 100
    num_bins = 4
    visualize = False

    zca = ZCAWhitening()
    ebn_filter = EventBasedNoiseFilter()
    sparse_filter = SparseFilter()

    provider = DatasetProvider(dataset_path, representation_type, delta_t_ms, num_bins, visualize=visualize,
                               zca=zca, ebn_filter=ebn_filter, sparse_filter=sparse_filter)
    train_dataset = provider.get_train_dataset()
    test_dataset = provider.get_test_dataset()
