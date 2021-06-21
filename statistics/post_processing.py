#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:34:09 2020
Post processing for binary masks. Run dilation/erosion and/or conencted component
analysis. Used for multi-label RT structs.
@author: ndahiya
"""

from scipy import ndimage as ndi
from skimage import measure
#import SimpleITK as sitk
import nrrd
import argparse
import numpy as np
import os
from os import path

parser = argparse.ArgumentParser(description="This script applies 3D blob "
                                 "analysis on multi-class predicted masks "
                                 "to clean small random noise.")
parser.add_argument("--in_dir", help="Inference masks input directory.",
                    type=str)
parser.add_argument("--out_dir", help="Output directory to save cleaned masks.",
                    type=str)
parser.add_argument("--test_ids_file", help="Text file containing names/ids of "
                    "datasets to run.", type=str)
parser.add_argument("--num_dsets", help="Number of datasets to run.",
                    type=int)
parser.add_argument("--in_suffix", help="Suffix of the input inference masks.",
                    type=str)
parser.add_argument("--out_suffix", help="Suffix of cleaned output masks.",
                    type=str)
parser.add_argument("--num_classes", help="Number of classes present in masks.",
                    type=int)
parser.add_argument("--num_cc_to_keep", help=" List of number of blobs to keep "
                    " corresponding to each class label in output masks.",
                    type=int, nargs='*')
# Set dafualt values here
#in_dir='../results/pseudocbct2ct3d_ignore_cases/test_latest/npz_images',
parser.set_defaults(in_dir='/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/PseudoCBCT2CT-3D/results/week1_aug_valid_finetune_msk_aapm_noisefree/test_latest/npz_images',
                    out_dir='/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/PseudoCBCT2CT-3D/results/week1_aug_valid_finetune_msk_aapm_noisefree/test_latest/post_processed',
                    test_ids_file='./test_week1.txt', #test_ids_file='./test_pseudocbct.txt',
                    num_dsets=249,
                    in_suffix='_RTSTRUCTS.nrrd',
                    out_suffix='_RTSTRUCTS.nrrd',
                    num_classes=4,
                    num_cc_to_keep=[2,1,1,1])
args = parser.parse_args()

in_folder       = args.in_dir
out_folder      = args.out_dir
test_ids_file   = args.test_ids_file
num_dsets       = args.num_dsets
in_suffix       = args.in_suffix
out_suffix      = args.out_suffix
num_classes     = args.num_classes
num_cc_to_keep  = args.num_cc_to_keep 
classes_present = [lbl for lbl in range(1,num_classes+1)]

def get_caselist(case_file):
  # Get list of cases from case_file (train/test/valid)
  case_list = []
  with open(case_file, 'r') as f:
    for line in f:
      case_list.append(line.strip())
  return case_list

# ------------- Load name of all datasets to process ----------------------
if not os.path.exists(out_folder):
  os.makedirs(out_folder)
  
test_ids_file = path.abspath(test_ids_file)
test_list = get_caselist(test_ids_file)

nb_dsets_to_run = len(test_list)
apply_image_opening = False # Erosion followed by dilation (remove small isolated blobss)
apply_image_closing = False # Dilation followed by erosion (close small holes)
# 8 connectivity in 3D, any voxel with at least one background voxel as
# neighbor is part of object boundary
connectivity_struc = ndi.generate_binary_structure(3, 1)

# ------------- Process all predicted masks -----------------------------
for t_idx, test_id in enumerate(test_list):
  print('Processing {}: {} of {}'.format(test_id, t_idx+1, nb_dsets_to_run))
  pred_mask_name = in_folder + '/' + test_id + in_suffix
  pred_mask_name = path.abspath(pred_mask_name)
  print(pred_mask_name)
  #mask_itk = sitk.ReadImage(pred_mask_name)
  #mask_arr = sitk.GetArrayFromImage(mask_itk)
  mask_arr, _ = nrrd.read(pred_mask_name)
  
  out_mask = np.zeros_like(mask_arr)
  # Process each class present in the masks
  for idx, lbl in enumerate(classes_present):
    # Isolate the current class
    curr_class = np.zeros_like(mask_arr)
    curr_class[np.where(mask_arr == lbl)] = 1
    if apply_image_opening is True:
      #curr_class = ndi.binary_opening(curr_class,
                                      #structure=connectivity_struc).astype(np.uint8)
      curr_class = ndi.binary_dilation(curr_class, structure=ndi.generate_binary_structure(2, 1), iterations=3)
    # Find all connected components
    cc, num_cc = ndi.label(curr_class, structure=connectivity_struc)
    print(num_cc)
    # Find the largest connected component
    properties = measure.regionprops(cc)
    areas = [blobs.area for blobs in properties]
    largest_blob_idx = np.argmax(areas)
    largest_blob_lbl = properties[largest_blob_idx].label
    # Keep the largest blob
    
    out_mask[np.where(cc == largest_blob_lbl)] = lbl
    
    # If we need to keep more than 1 CC's, and have more than once CC, add the
    # other next largest CC's to the largest CC
    curr_num_cc_to_keep = num_cc_to_keep[idx]
    if (num_cc < curr_num_cc_to_keep):
      curr_num_cc_to_keep = num_cc
    if (curr_num_cc_to_keep > 1):
      sorted_areas_idx = list(reversed(np.argsort(areas))) # Descending order
      sorted_areas_idx = sorted_areas_idx[1:curr_num_cc_to_keep] # Largest already kept
      labels_to_keep = [properties[cc_idx].label for cc_idx in sorted_areas_idx]
      for cc_label in labels_to_keep:
        out_mask[np.where(cc == cc_label)] = lbl
  
  # Write processed mask
  processed_mask_name = out_folder + '/' + test_id + out_suffix
  processed_mask_name = path.abspath(processed_mask_name)
  
  #sitk.WriteImage(sitk.GetImageFromArray(out_mask), processed_mask_name)
  nrrd.write(processed_mask_name, out_mask)
  
print('Done')
































