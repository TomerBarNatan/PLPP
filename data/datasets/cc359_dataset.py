import pickle

import numpy as np
import torch
from dpipe.dataset.segmentation import SegmentationFromCSV
from dpipe.dataset.wrappers import Proxy
from dpipe.dataset.wrappers import apply, cache_methods
from dpipe.im.box import get_centered_box
from dpipe.im.patch import sample_box_center_uniformly
from dpipe.im.shape_ops import crop_to_box
from dpipe.im.shape_ops import zoom
from tqdm import tqdm

from paths import cc359_data_path, cc359_splits_dir

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


def get_center(shape, spatial_dims):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    return spatial_shape // 2


def sample_center_uniformly(shape, patch_size, spatial_dims):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    if np.all(patch_size <= spatial_shape):
        return sample_box_center_uniformly(shape=spatial_shape, box_size=patch_size)
    else:
        return spatial_shape // 2


def get_patch_2d(image_slc, segm_slc, x_patch_size, y_patch_size, random_patch=False):
    sp_dims_2d = (-2, -1)
    if random_patch:
        center = sample_center_uniformly(segm_slc.shape, y_patch_size, sp_dims_2d)
    else:
        center = get_center(segm_slc.shape, sp_dims_2d)
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
        return x

    def load_spacing(self, i):
        old_spacing = self.load_orig_spacing(i)
        spacing = self.new_voxel_spacing.copy()
        spacing[np.isnan(spacing)] = old_spacing[np.isnan(spacing)]
        return spacing

    def load_orig_spacing(self, i):
        return self._shadowed.load_spacing(i)


class CC359Ds(torch.utils.data.Dataset):
    def __init__(self, ids, site, yield_id=False, slicing_interval=1, random_patch=False,
                 patch_size=np.array([256, 256])):
        self.patch_size = patch_size
        self.random_patch = random_patch
        self.slicing_interval = slicing_interval
        voxel_spacing = (1, 0.95, 0.95)
        preprocessed_dataset = apply(Rescale3D(CC359(cc359_data_path), voxel_spacing), load_image=scale_mri)
        ds = apply(cache_methods(apply(preprocessed_dataset, load_image=np.float16)), load_image=np.float32)
        all_img_segs_dict_path = cc359_splits_dir / f'site_{site}' / 'all_img_segs.p'
        if all_img_segs_dict_path.exists():
            print('using all dict')
            self.all_img_segs_dict = pickle.load(open(all_img_segs_dict_path, 'rb'))
            temp_dict = {}
            for i, val in self.all_img_segs_dict.items():
                if i in ids:
                    temp_dict[i] = val
            self.all_img_segs_dict = temp_dict
            self.image_loader = lambda i: self.all_img_segs_dict[i][0]
            self.seg_loader = lambda i: self.all_img_segs_dict[i][1]
        else:
            print(
                'Warning: create all_images_pickle using python3 -m dataset create_all_images_pickle will make training faster')
            self.all_img_segs_dict = None
            self.image_loader = ds.load_image
            self.seg_loader = ds.load_segm
        self.spacing_loader = ds.load_spacing
        self.len_ds = 0
        self.yield_id = yield_id
        self.i_to_id = []
        for id1 in tqdm(ids, desc='calculating data_len'):
            self.i_to_id.append([self.len_ds, id1])
            num_of_slices = self.image_loader(id1).shape[-1]
            self.len_ds = self.len_ds + num_of_slices
        self.prev_img = None
        self.prev_seg = None
        self.prev_id = None

    def __getitem__(self, item):
        x_patch_size = y_patch_size = self.patch_size
        for (i, id1), (next_i, _) in zip(self.i_to_id, self.i_to_id[1:] + [(self.len_ds, None)]):
            if i <= item < next_i:
                if id1 != self.prev_id:
                    img = self.image_loader(id1)
                    seg = self.seg_loader(id1)
                    self.prev_img = img
                    self.prev_seg = seg
                    self.prev_id = id1
                else:
                    img = self.prev_img
                    seg = self.prev_seg
                slc_num = item - i
                slc_num -= slc_num % self.slicing_interval
                img_slc = img[..., slc_num]
                seg_slc = seg[..., slc_num]

                img_slc, seg_slc = get_patch_2d(img_slc, seg_slc, x_patch_size=x_patch_size, y_patch_size=y_patch_size,
                                                random_patch=self.random_patch)
                img_slc, seg_slc = np.expand_dims(img_slc, axis=0), np.expand_dims(seg_slc, axis=0)
                if self.yield_id:
                    return img_slc, seg_slc, self.load_id(id1), slc_num
                return img_slc, seg_slc

    def __len__(self):
        return self.len_ds

    def load_id(self, id1):
        return int(id1[2:])
