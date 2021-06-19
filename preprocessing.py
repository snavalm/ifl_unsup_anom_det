
import os
import pickle
import glob
import random
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom

from collections import namedtuple
from torch.utils.data import Dataset, DataLoader

img_extended = namedtuple( 'img_extended', ('img', 'seg', 'cid') )
train_batch = namedtuple('train_batch',('xyz','s','k'))

#%% Pytorch datasets and loaders
class xyzs_dataset(Dataset):
    def __init__(self,data_dir, samples = 1000,
                 dict_cid = None,):
        """
        Args:
        :param data_dir: Directory of images (str)
        :param dict_cid: If a dictionary of cid is passed, use this, if not initialize.
                        Dictionary 'cid'-->key for embedding
        """
        self.data_dir = data_dir

        # Set is defined as the filenames, which matches cid (case id)
        set = glob.glob(data_dir+'/*.nt')
        set = [os.path.basename(s).split(".")[0] for s in set]
        self.set = set

        if dict_cid is None:
            # Try to find if a dictionary exist in folder
            dict_cid_filename = os.path.join(data_dir,'dict_cid.k')
            if os.path.exists(dict_cid_filename):
                self.dict_cid = pickle.load(open(dict_cid_filename, 'rb'))
            else:
                #Generate dict_cid and dump
                self.dict_cid = {file:i for i,file in enumerate(self.set)}
                pickle.dump( self.dict_cid, open( dict_cid_filename, 'wb' ) )
        else:
            # If provided when class instatiated, just use this one
            self.dict_cid = dict_cid

        self.samples = samples

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):

        file_name = os.path.join(self.data_dir,self.set[item]+'.nt')
        img_batch = pickle.load(open(file_name, 'rb'))

        # Get samples. Assumes a single 3d image per img_batch
        XYZ = img_batch[0].img.shape
        xyz = [random.choices(range(size), k=self.samples) for size in XYZ]
        s = img_batch[0].img[tuple(xyz)]

        # Normalize in range [ -1, 1]
        xyz = np.array(xyz) / np.array(XYZ)[:, np.newaxis]
        xyz = (xyz * 2) - 1
        xyz = xyz.astype(np.float32)

        return train_batch(xyz,s,[self.dict_cid[img_batch[0].cid]])

class xyzs_loader(DataLoader):
    def __init__(self, data_dir, scenes_per_batch, samples_per_scene = 1000,
                 shuffle = True, num_workers = 2,
                 **dataset_kwargs):

         self.data = xyzs_dataset(data_dir = data_dir,
                             samples = samples_per_scene,
                             **dataset_kwargs)
         super( xyzs_loader, self ).__init__( self.data,
                                              batch_size = scenes_per_batch,
                                              shuffle = shuffle, num_workers = num_workers,
                                              collate_fn = collate_fn
                                              )

def collate_fn(batches):
    # Stack the np.arrays
    batch = train_batch(*map(np.stack, zip(*batches)))
    return batch

class vol_dataset(Dataset):
    def __init__(self,data_dir):
        """
        Args:
        :param data_dir: Directory of images (str)
        """
        self.data_dir = data_dir

        # Set is defined as the filenames, which matches cid (case id)
        set = glob.glob(data_dir+'/*.nt')
        set = [os.path.basename(s).split(".")[0] for s in set]
        self.set = set

    def __len__(self):
        return len(self.set)

    def __getitem__(self, item):

        file_name = os.path.join(self.data_dir,self.set[item]+'.nt')
        img_batch = pickle.load(open(file_name, 'rb'))

        img = img_batch[0].img
        seg = img_batch[0].seg

        if seg is not None:
            # If segmentation is NOT None (I.e. BRATS)
            seg = (seg > 0).astype(np.float32)
        else:
            # If it's HCP override segmentation with 0s
            seg = np.zeros_like(img).astype(np.float32)
        return img_extended( img, seg, img_batch[0].cid)

class vol_loader(DataLoader):
    def __init__(self, data_dir, samples, shuffle = True, num_workers = 2,
                 **dataset_kwargs):

         self.data = vol_dataset(data_dir = data_dir,
                             **dataset_kwargs)
         super( vol_loader, self ).__init__( self.data,
                                              batch_size = samples,
                                              shuffle = shuffle, num_workers = num_workers,
                                              collate_fn = collate_vol
                                              )

def collate_vol(batches):
    # Stack the np.arrays
    batch = img_extended(*map(np.stack, zip(*batches)))
    return batch


