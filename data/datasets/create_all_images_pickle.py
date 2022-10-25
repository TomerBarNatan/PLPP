import pickle

import numpy as np
from dpipe.dataset.segmentation import SegmentationFromCSV
from dpipe.dataset.wrappers import Proxy
from dpipe.dataset.wrappers import apply, cache_methods
from dpipe.im.box import get_centered_box
from dpipe.im.patch import sample_box_center_uniformly
from dpipe.im.shape_ops import crop_to_box
from dpipe.im.shape_ops import zoom
from dpipe.io import load
from tqdm import tqdm
from pathlib import Path

SPATIAL_DIMS = (-3, -2, -1)

def extract_patch(inputs, x_patch_size, y_patch_size, spatial_dims=SPATIAL_DIMS):
    x, y, center = inputs

    x_patch_size = np.array(x_patch_size)
    y_patch_size = np.array(y_patch_size)
    x_spatial_box = get_centered_box(center, x_patch_size)
    y_spatial_box = get_centered_box(center, y_patch_size)

    x_patch = crop_to_box(x, box=x_spatial_box, padding_values=np.min, axis=spatial_dims)
    y_patch = crop_to_box(y, box=y_spatial_box, padding_values=0, axis=spatial_dims)
    return x_patch, y_patch


def sample_center_uniformly(shape, patch_size, spatial_dims):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    if np.all(patch_size <= spatial_shape):
        return sample_box_center_uniformly(shape=spatial_shape, box_size=patch_size)
    else:
        return spatial_shape // 2


def get_random_patch_2d(image_slc, segm_slc, x_patch_size, y_patch_size):
    sp_dims_2d = (-2, -1)
    center = sample_center_uniformly(segm_slc.shape, y_patch_size, sp_dims_2d)
    x, y = extract_patch((image_slc, segm_slc, center), x_patch_size, y_patch_size, spatial_dims=sp_dims_2d)
    return x, y


class CC359(SegmentationFromCSV):
    def __init__(self, data_path, modalities=('MRI',), target='brain_mask', metadata_rpath='meta.csv'):
        super().__init__(data_path=data_path,
                         modalities=modalities,
                         target=target,
                         metadata_rpath=metadata_rpath)
        self.n_domains = len(self.df['fold'].unique())

    def load_image(self, i):
        return np.float32(super().load_image(i)[0])  # 4D -> 3D

    def load_segm(self, i):
        return np.float32(super().load_segm(i))  # already 3D

    def load_shape(self, i):
        return np.int32(np.shape(self.load_segm(i)))

    def load_spacing(self, i):
        voxel_spacing = np.array([self.df['x'].loc[i], self.df['y'].loc[i], self.df['z'].loc[i]])
        return voxel_spacing

    def load_domain_label(self, i):
        domain_id = self.df['fold'].loc[i]
        return np.eye(self.n_domains)[domain_id]  # one-hot-encoded domain

    def load_domain_label_number(self, i):
        return self.df['fold'].loc[i]

    @staticmethod
    def load_id(i):
        return int(i[2:])

    def load_domain_label_number_binary_setup(self, i, domains):
        """Assigns '1' to the domain of the largest index; '0' to another one
        Domains may be either (index1, index2) or (sample_scan1_id, sample_scan2_id) """

        if type(domains[0]) != int:
            # the fold numbers of the corresponding 2 samples
            doms = (self.load_domain_label_number(domains[0]), self.load_domain_label_number(domains[1]))
        else:
            doms = domains
        largest_domain = max(doms)
        domain_id = self.df['fold'].loc[i]
        if domain_id == largest_domain:
            return 1
        elif domain_id in doms:  # error otherwise
            return 0


class Change(Proxy):
    def _change(self, x, i):
        raise NotImplementedError

    def load_image(self, i):
        return self._change(self._shadowed.load_image(i), i)

    def load_segm(self, i):
        return np.float32(self._change(self._shadowed.load_segm(i), i) >= .5)


def scale_mri(image: np.ndarray, q_min: int = 1, q_max: int = 99) -> np.ndarray:
    image = np.clip(np.float32(image), *np.percentile(np.float32(image), [q_min, q_max]))
    image -= np.min(image)
    image /= np.max(image)
    return np.float32(image)


class Rescale3D(Change):
    def __init__(self, shadowed, new_voxel_spacing=1., order=3):
        super().__init__(shadowed)
        self.new_voxel_spacing = np.broadcast_to(new_voxel_spacing, 3).astype(float)
        self.order = order

    def _scale_factor(self, i):
        old_voxel_spacing = self._shadowed.load_spacing(i)
        scale_factor = old_voxel_spacing / self.new_voxel_spacing
        return np.nan_to_num(scale_factor, nan=1)

    def _change(self, x, i):
        return zoom(x, self._scale_factor(i), order=self.order)

    def load_spacing(self, i):
        old_spacing = self.load_orig_spacing(i)
        spacing = self.new_voxel_spacing.copy()
        spacing[np.isnan(spacing)] = old_spacing[np.isnan(spacing)]
        return spacing

    def load_orig_spacing(self, i):
        return self._shadowed.load_spacing(i)


def create_pickle(site):
    voxel_spacing = (1, 0.95, 0.95)
    preprocessed_dataset = apply(Rescale3D(CC359(cc359_data_path), voxel_spacing), load_image=scale_mri)
    ds = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)
    base_path = cc359_splits_dir + f'/site_{site}/'
    all_dict = {}
    ids = load(base_path + 'train_ids.json') + load(base_path + 'val_ids.json') + load(base_path + 'test_ids.json')
    for id1 in tqdm(ids, desc=f'calculating data_len site  {site}'):
        img = ds.load_image(id1)
        seg = ds.load_segm(id1)
        all_dict[id1] = (img, seg)

    pickle.dump(all_dict, open(base_path + '/all_img_segs.p', 'wb'))


for s in [0, 1, 2, 3, 4, 5]:
    create_pickle(s)
