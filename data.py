from torch.utils.data import Dataset
import numpy as np

from utils import make_grid_np, rand_rotation_matrix, voxelize_gauss


class C2FDataSet(Dataset):
    def __init__(
        self,
        coords_aa,
        coords_cg,
        sigma,
        resolution,
        length,
        rand_rot: bool,
        n_frames: int,
    ):
        delta_s = length / resolution
        self.sigma = sigma
        self.coords_aa = coords_aa
        self.coords_cg = coords_cg
        self.grid = make_grid_np(delta_s, resolution)
        self.grid_shape = self.grid.shape[-3:]
        self.n_frames = n_frames
        self.rand_rot = rand_rot

    def __len__(self):
        return len(self.coords_aa) - self.n_frames

    @staticmethod
    def _mean_center(x):
        return x - x.mean(0)

    def __getitem__(self, item):
        if self.rand_rot:
            R = rand_rotation_matrix()
        else:
            R = np.eye(3)

        aa = self._mean_center(self.coords_aa[item] @ R.T)
        for i in range(1, self.n_frames):
            coords = self._mean_center(self.coords_aa[item + i] @ R.T)
            aa = np.concatenate((aa, coords))

        aa_vox = voxelize_gauss(aa, self.sigma, self.grid)

        cg = self._mean_center(self.coords_cg[item] @ R.T)
        for i in range(1, self.n_frames):
            coords = self._mean_center(self.coords_cg[item + i] @ R.T)
            cg = np.concatenate((cg, coords))

        cg_vox = voxelize_gauss(cg, self.sigma, self.grid)

        aa = aa.reshape(self.n_frames, -1, 3)
        aa_vox = aa_vox.reshape(self.n_frames, -1, *self.grid_shape)
        cg = cg.reshape(self.n_frames, -1, 3)
        cg_vox = cg_vox.reshape(self.n_frames, -1, *self.grid_shape)

        return (
            aa.astype(np.float32),
            aa_vox.astype(np.float32),
            cg.astype(np.float32),
            cg_vox.astype(np.float32),
        )


class C2FDataSetCLN(Dataset):
    def __init__(
        self,
        coords_aa,
        coords_cg,
        sigma,
        resolution,
        length,
        rand_rot: bool,
        n_frames: int,
    ):
        delta_s = length / resolution
        self.sigma = sigma
        self.coords_aa = coords_aa
        self.coords_cg = coords_cg
        self.grid = make_grid_np(delta_s, resolution)
        self.grid_shape = self.grid.shape[-3:]
        self.n_frames = n_frames
        self.rand_rot = rand_rot

        self.frames = np.array([a.shape[0] - self.n_frames for a in self.coords_aa])
        self.frames_cumsum = np.cumsum(self.frames)

    @staticmethod
    def _mean_center(x):
        return x - x.mean(0)

    def __len__(self):
        return self.frames_cumsum[-1]

    def __getitem__(self, item):
        if self.rand_rot:
            R = rand_rotation_matrix()
        else:
            R = np.eye(3)

        trj_idx = np.searchsorted(self.frames_cumsum, item, side="left")
        if trj_idx == 0:
            frame_idx = item
        else:
            frame_idx = item - self.frames_cumsum[trj_idx - 1]

        aa = self._mean_center(self.coords_aa[trj_idx][frame_idx] @ R.T)
        for i in range(1, self.n_frames):
            coords = self._mean_center(self.coords_aa[trj_idx][frame_idx + i] @ R.T)
            aa = np.concatenate((aa, coords))

        aa_vox = voxelize_gauss(aa, self.sigma, self.grid)

        cg = self._mean_center(self.coords_cg[trj_idx][frame_idx] @ R.T)
        for i in range(1, self.n_frames):
            coords = self._mean_center(self.coords_cg[trj_idx][frame_idx + i] @ R.T)
            cg = np.concatenate((cg, coords))

        cg_vox = voxelize_gauss(cg, self.sigma, self.grid)

        aa = aa.reshape(self.n_frames, -1, 3)
        aa_vox = aa_vox.reshape(self.n_frames, -1, *self.grid_shape)
        cg = cg.reshape(self.n_frames, -1, 3)
        cg_vox = cg_vox.reshape(self.n_frames, -1, *self.grid_shape)

        return (
            aa.astype(np.float32),
            aa_vox.astype(np.float32),
            cg.astype(np.float32),
            cg_vox.astype(np.float32),
        )
