#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:16:11 2020

@author: ndahiya
"""

import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, transform_cbct_ct_pair3D
from data.image_folder import make_dataset
import numpy as np
import torch

class Cbct2ct3dDataset(BaseDataset):
  """
  A dataset class for 2D CBCT to CT pix2pix translation task.
  
  It assumes that the directory '/path/to/data/train' contains *.npz images which
  have two arrays named 'CBCT', 'CT'. Assuming data is already the desired size.
  
  During test time, you need to prepare a directory '/path/to/data/test'.
  """
  
  def __init__(self, opt):
    """Initialize this dataset class.

    Parameters:
    opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
    """
    BaseDataset.__init__(self, opt)
    self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
    self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
    assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
    self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
    self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
    self.phase = self.opt.phase
    
  def __getitem__(self, index):
    """Return a data point and its metadata information.

    Parameters:
      index - - a random integer for data indexing

    Returns a dictionary that contains A, B, A_paths and B_paths
      A (tensor) - - an image in the input domain
      B (tensor) - - its corresponding image in the target domain
      A_paths (str) - - image paths
      B_paths (str) - - image paths (same as A_paths)
      """
    # read a image given a random integer index
    AB_path = self.AB_paths[index]
    AB = np.load(AB_path)
    
    # Get AB image into A and B
    A = AB['CT']
    B = AB['CBCT']
    R = AB['RTSTRUCTS']

    # apply the same transform to both A and B
    if self.phase == 'train':
      A, B, R = transform_cbct_ct_pair3D(A, B, R, apply_transform=False)
      A = torch.unsqueeze(A, dim=0)
      B = torch.unsqueeze(B, dim=0)
      R = torch.unsqueeze(R, dim=0)
      A= torch.cat((A,R), axis=0) # Doing Ct/RTStructs segmentation
      print(B.dtype)
    else:
      A, B, R = transform_cbct_ct_pair3D(A, B, R, apply_transform=False)
      A = torch.unsqueeze(A, dim=0)
      B = torch.unsqueeze(B, dim=0)
      R = torch.unsqueeze(R, dim=0)
      A= torch.cat((A,R), axis=0) # Doing Ct/RTStructs segmentation
      
    return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

  def __len__(self):
    """Return the total number of images in the dataset."""
    return len(self.AB_paths)
