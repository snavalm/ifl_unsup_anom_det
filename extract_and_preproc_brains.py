import nibabel as nib

from collections import defaultdict, namedtuple
import pickle
import numpy as np
from scipy.ndimage import zoom

import torch
import torch.nn.functional as F

import os
import argparse

# import matplotlib.pyplot as plt

img_extended = namedtuple( 'img_extended', ('img', 'seg', 'cid') )



def ClipPercentiles(vol, percentile=98):
    non_cero_mask = np.where( vol != 0 )
    perc_min, perc_max = np.percentile( vol[non_cero_mask], [100 - percentile, percentile] )
    vol[non_cero_mask] = np.clip( vol[non_cero_mask], perc_min, perc_max )
    return vol

def Normalize(vol):
    non_cero_mask = np.where( vol != 0 )
    mi, ma = vol[non_cero_mask].min(), vol[non_cero_mask].max()
    vol[non_cero_mask] = (vol[non_cero_mask] - mi) / (ma - mi)
    return vol

def encode_img_intensities(img, cluster_centers):
    """ Return an img with intensities encoded in the id of cluster of intensities"""
    if len( img.shape ) == 3:
        img = img[np.newaxis, :, :, :]
    elif len( img.shape ) == 2:
        img = img[np.newaxis, np.newaxis, :, :]
    cluster_centers = cluster_centers.reshape( -1, 1, 1, 1 )
    img = np.argmin( np.abs( cluster_centers - img ), axis = 0 )
    return img.squeeze()


class cluster_encoder():
    def __init__(self, cluster_centers):
        self.cluster_centers = cluster_centers

    def apply(self, img):
        return encode_img_intensities( img, self.cluster_centers )


def Unfold3d(tensor, kernel_size=3, stride=1, padding=1):
    ts = tensor.shape
    adj = -1 if kernel_size % 2 == 0 else 0
    tensor = F.pad( tensor, (padding, padding + adj, padding, padding + adj, padding, padding + adj) )

    return tensor.unfold( 1, size = kernel_size, step = stride ). \
        unfold( 2, size = kernel_size, step = stride ). \
        unfold( 3, size = kernel_size, step = stride ). \
        reshape( *ts, kernel_size ** 3 )


def ModePool3d(tensor, kernel_size=3, stride=1, padding=1):
    tensor = Unfold3d( tensor, kernel_size, stride, padding )
    #     tensor = F.one_hot(tensor,num_classes=10) Replaced with below
    oh_tmp = torch.arange( 10, dtype = torch.uint8 ).view( *[1] * 5, -1 )
    oh_tmp = oh_tmp.to( tensor.device )
    tensor = (tensor.unsqueeze( -1 ) == oh_tmp)
    tensor = tensor.sum( -2, dtype = torch.uint8 ).argmax( -1 )
    return tensor


def save_file(case_id, subj_dict, tgt_dir):

    target_file = os.path.join( tgt_dir, case_id + ".nt" )

    x_tmp = [img_extended( subj_dict["t2"], subj_dict["label"] if "label" in subj_dict else None, case_id )]
    pickle.dump( x_tmp, open( target_file, 'wb' ) )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument( '--dataset', type = str )
    parser.add_argument( '--source', type = str )
    parser.add_argument( '--target', type = str )
    parser.add_argument( '--res', type = str, default = '160', help = 'Either 160 or full' )
    parser.add_argument( '--cluster_centers', type = str, help = 'Location of the file with ' +
                                                                 'pixel intensity cluster centres' )
    parser.add_argument( '--mode_pool', type = int, default = None, help = 'By default not applied, ' +
                                                                           'otherwise kernel 2 or 3')

    args = parser.parse_args()

    src_dir = args.source
    tgt_dir = args.target

    cluster_centers = pickle.load(open(args.cluster_centers, 'rb' ) )
    cluster_encoder = cluster_encoder( cluster_centers )

    subj_dict = defaultdict( dict )

    if args.dataset == 'HCP':
        ###### HCP ########
        t2_templ = "T2w_acpc_dc_restore_brain.nii.gz"

        for cntr, (root, dirs, files) in enumerate( os.walk( src_dir ) ):

            case_id = root.split( "/" )[-2]

            for f_name in files:
                print( os.path.join( root, f_name ) )

                if f_name in [t2_templ]:
                    vol = nib.load( os.path.join( root, f_name ) )
                    vol = np.asarray( vol.get_fdata() )

                    vol = vol.transpose( (2, 1, 0) )
                    vol = vol[:, ::-1, :]
                    vol = np.pad(vol,[(20,20),(0,0),(20,20)], 'constant', constant_values = 0)
                    vol = vol[:, 5:-6, :]

                    vol = ClipPercentiles( vol )
                    vol = Normalize( vol )

                    # plt.imshow( vol[:, 150] )
                    # plt.show()

                    if args.res == '160':
                        vol = zoom( vol, (160 / vol.shape[1],) * 3, order = 1 )

                    # plt.imshow( vol[:, 80] )
                    # plt.show()

                    # Apply intensity cluster encoding
                    vol = cluster_encoder.apply( vol )

                    # plt.imshow( vol[:, 80] )
                    # plt.show()

                    # Apply Mode Pool
                    # TODO Implement this without torch
                    if args.mode_pool:
                        vol = ModePool3d( torch.from_numpy( vol ).unsqueeze( 0 ),
                                          kernel_size = args.mode_pool, padding = 1 )
                        vol = vol.squeeze().numpy()

                    # plt.imshow( vol[:, 80] )
                    # plt.show()

                    vol = vol.astype( np.uint8 )

                    if f_name == t2_templ:
                        subj_dict[case_id]["t2"] = vol

            if "t2" in subj_dict[case_id]:
                case_dict = subj_dict.pop( case_id )
                save_file( case_id, case_dict, tgt_dir )

        print( "Cases remaining: ", len( subj_dict ) )
        for k in subj_dict.keys():
            print( k )

    if args.dataset == 'BraTS':
        ###### BraTS ########
        for cntr, (root, dirs, files) in enumerate( os.walk( src_dir ) ):

            case_id = root.split( "/" )[-1]

            for f_name in files:
                if f_name.endswith( "seg.nii.gz" ) or f_name.endswith( "t2.nii.gz" ):
                    print( os.path.join( root, f_name ) )

                    vol = nib.load( os.path.join( root, f_name ) )
                    vol = np.asarray( vol.get_fdata() )
                    vol = vol.transpose( (2, 1, 0) )
                    vol = np.pad(vol,[(42,43),(0,0),(0,0)],'constant',constant_values = 0)

                    if f_name.endswith( "seg.nii.gz" ): # nii.gz with segmentation
                        vol = vol.astype( 'uint8' )

                        if args.res == '160':
                            vol = zoom( vol, (160 / vol.shape[1],) * 3, order = 1 )

                        subj_dict[case_id]["label"] = vol

                    else: #nii.gz with T2
                        vol = ClipPercentiles( vol )
                        vol = Normalize( vol )

                        if args.res == '160':
                            vol = zoom( vol, (160 / vol.shape[1],) * 3, order = 1 )

                        # Apply intensity cluster encoding
                        vol = cluster_encoder.apply(vol)

                        # Apply Mode Pool
                        # TODO Implement this without torch
                        if args.mode_pool:
                            vol = ModePool3d( torch.from_numpy( vol ).unsqueeze( 0 ),
                                              kernel_size = args.mode_pool, padding = 1 )
                            vol = vol.squeeze().numpy()

                        vol = vol.astype(np.uint8)

                        if f_name.endswith( "t2.nii.gz" ):
                            subj_dict[case_id]["t2"] = vol

            # If it finds both segmentation and T2, generate file
            if "t2" in subj_dict[case_id] and "label" in subj_dict[case_id]:
                case_dict = subj_dict.pop( case_id )
                save_file( case_id, case_dict, tgt_dir )

        print( "Cases remaining: ", len( subj_dict ) )
        for k in subj_dict.keys():
            print( k )