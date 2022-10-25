import os

import SimpleITK as sitk
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import paths
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from paths import multiSiteMri_site_to_int

cudnn.benchmark = True


class MultiSiteMri(torch.utils.data.Dataset):
    def __init__(self, ids, yield_id=False, test=False):
        self.yield_id = yield_id
        self.test = test
        self.random_patch = False
        self.patch_size = None

        self.patches_Allimages, self.patches_Allmasks = self.create_datalists(ids)

    def load_image(self, id1):
        return self.patches_Allimages[id1[0]]

    def load_segm(self, id1):
        return self.patches_Allmasks[id1[0]]

    def load_domain_label_number(self, id1):
        id1 = id1[0]
        sup = id1.split('/')[0]
        return multiSiteMri_site_to_int[sup]

    def load_id(self, id1):

        return int(str(self.load_domain_label_number(id1)) + id1[0][-6:-4])

    def __getitem__(self, idx):

        for (i, id1), (next_i, _) in zip(self.i_to_id, self.i_to_id[1:] + [(self.len_ds, None)]):
            if i <= idx < next_i:
                img, seg = self.patches_Allimages[id1[0]], self.patches_Allmasks[id1[0]]
                slc_num = idx - i
                img_slc = torch.tensor(img[slc_num])
                seg_slc = torch.tensor(seg[slc_num])
                if self.random_patch:
                    x1, x2, x3, x4 = transforms.RandomCrop.get_params(
                        img_slc, output_size=self.patch_size)
                    img_slc = TF.crop(img_slc, x1, x2, x3, x4)
                    seg_slc = TF.crop(seg_slc, x1, x2, x3, x4)
                if self.yield_id:
                    return img_slc, seg_slc, self.load_id(id1), slc_num
                return img_slc, seg_slc

    def __len__(self):
        return self.len_ds

    def create_datalists(self, ids):
        patches_Allimages = {}
        patches_Allmasks = {}
        self.i_to_id = []
        self.len_ds = 0
        for id1 in tqdm(ids, desc='creating dataset MSM'):
            self.i_to_id.append([self.len_ds, id1])
            patches = self.extract_patch(id1)
            patches_Allimages[id1[0]] = patches[0]
            patches_Allmasks[id1[0]] = patches[1]
            self.len_ds = self.len_ds + len(patches[0])
        return patches_Allimages, patches_Allmasks

    def extract_patch(self, id1):
        """Extracts a patch of given resolution and size at a specific location."""
        image, mask = self.parse_fn(id1)  # get the image and its mask
        image_patches = []
        mask_patches = []
        num_patches_now = 0
        limX, limY, limZ = np.where(mask > 0)

        z = []
        min1 = np.min(limZ)
        max1 = np.max(limZ)
        if self.test:
            for i in range(1, mask.shape[2] - 2):
                z.append(i)
        else:
            for i in range(1, mask.shape[2] - 2):
                if min1 <= i < max1:
                    z.append(i)
                elif np.random.random() < 0.1:
                    z.append(i)
        num_patches = len(z)

        while num_patches_now < num_patches:
            image_patch = image[:, :, z[num_patches_now] - 1:z[num_patches_now] + 2]
            mask_patch = mask[:, :, z[num_patches_now]]

            image_patches.append(image_patch)
            mask_patches.append(mask_patch)
            num_patches_now += 1
        image_patches = np.stack(
            image_patches)
        mask_patches = np.stack(mask_patches)
        mask_patches = np.expand_dims(mask_patches, -1)

        image_patches = image_patches.transpose([0, 3, 1, 2])
        mask_patches = mask_patches.transpose([0, 3, 1, 2])
        return image_patches, mask_patches

    def parse_fn(self, data_path):
        '''
        :param image_path: path to a folder of a patient
        :return: normalized entire image with its corresponding label
        In an image, the air region is 0, so we only calculate the mean and std within the brain area
        For any image-level normalization, do it here
        '''

        image_path = os.path.join(paths.msm_data_path, data_path[0])

        label_path = os.path.join(paths.msm_data_path, data_path[1])
        itk_image = sitk.ReadImage(image_path)  # os.path.join(image_path, 'T1_unbiased_brain_rigid_to_mni.nii.gz'))
        itk_mask = sitk.ReadImage(label_path)  # os.path.join(image_path, 'T1_brain_seg_rigid_to_mni.nii.gz'))

        image = sitk.GetArrayFromImage(itk_image)
        mask = sitk.GetArrayFromImage(itk_mask)
        binary_mask = np.ones(mask.shape)
        mean = np.sum(image * binary_mask) / np.sum(binary_mask)
        std = np.sqrt(np.sum(np.square(image - mean) * binary_mask) / np.sum(binary_mask))
        image = (image - mean) / std  # normalize per image, using statistics within the brain, but apply to whole image
        mask[mask == 2] = 1

        return image.transpose([1, 2, 0]), mask.transpose([1, 2, 0])  # transpose the orientation of the


class InfiniteLoader:
    def __init__(self, dl, batches_per_epoch):
        self.dl = dl
        self.batches_per_epoch = batches_per_epoch

    def __iter__(self):
        self.batch_idx = 0
        self.iter = iter(self.dl)
        return self

    def __call__(self):
        return self

    def __next__(self):
        self.batch_idx += 1
        if self.batch_idx > self.batches_per_epoch:
            raise StopIteration()
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dl)
            return next(self.iter)


class MultiSiteDl(torch.utils.data.DataLoader):
    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
