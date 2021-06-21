#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  20 01:21:09 2020
Extract the 3D pseudo cbct dataset which has 5 RT Structs, 7 artifact based
CBCTs and 16 augmented CBCTs. Crop black border and resize to 128x128x128. Some CBCts/CT pairs have range > 1,
rescale those to [0 1], these are the cases causing high L1 loss which were previously ignored.
Also don't include GTV in combined RT structs. Not being included to prepare results for paper
Not cropping borders after all. Only testing images seem to have this problem. Only, rescaling images with range > 1 to have range [0 1] then.
@author: ndahiya
"""

import os
import nrrd
from skimage.transform import resize
import numpy as np


def get_caselist(case_file):
  # Get list of cases from case_file (train/test/valid)
  case_list = []
  with open(case_file, 'r') as f:
    for line in f:
      case_list.append(line.strip())
  return case_list


def fix_size(filename, is_mask=True, expected_shape=(128, 128, 128)):
  img, imginfo = nrrd.read(os.path.join(filename))

  order = 0
  if is_mask is False:
    order = 1

  if img.shape != expected_shape:
    img = resize(img, expected_shape, order=order, preserve_range=True, anti_aliasing=False)
  img = np.swapaxes(img, 0, -1)  # X,Y,Z to Z,Y,X

  return img

def get_border(ct):
  # The border region around the ct image's first slice
  # Actually return a grid that will exclude the borders
  img = ct[0]
  mask = img > 0
  x, y = np.any(mask, 0), np.any(mask, 1)
  grid = np.ix_(y, x)
  return grid

def crop_border_fix_size(img, valid_grid, crop_border=False, is_mask=True, expected_shape=(128, 128, 128)):
  # crop border and then resize to 128x128x128
  if crop_border is True:

    slice_2d = img[0]
    masked = slice_2d[valid_grid] # To get shape after removing border
    d = img.shape[0]
    out_img = np.zeros((d, masked.shape[0], masked.shape[1]), img.dtype)

    for i in range(d):
      out_img[i] = img[i][valid_grid]
  else:
    out_img = img

  order = 0
  if is_mask is False:
    order = 1

  if out_img.shape != expected_shape:
    out_img = resize(out_img, expected_shape, order=order, preserve_range=True, anti_aliasing=False)

  return out_img

def save_case_combined_rtstructs(heart, lungs, cord, eso, cbct, ct, dose, subcase, out_folder):
  # Get border from CT, crop and resize all using the border
  grid = get_border(ct)
  ct = crop_border_fix_size(ct, grid, is_mask=False)
  cbct = crop_border_fix_size(cbct, grid, is_mask=False)
  dose = crop_border_fix_size(dose, grid, is_mask=False)
  heart = crop_border_fix_size(heart, grid, is_mask=True)
  lungs = crop_border_fix_size(lungs, grid, is_mask=True)
  eso = crop_border_fix_size(eso, grid, is_mask=True)
  cord = crop_border_fix_size(cord, grid, is_mask=True)

  # Combine all rtstructs into one image:
  # BG = 0, Eso = 4, Cord = 3, Heart = 2, Lungs = 1
  RT_Structs = np.zeros((128, 128, 128), dtype=np.uint8)
  RT_Structs[np.where(eso == 1)] = 4
  RT_Structs[np.where(cord == 1)] = 3
  RT_Structs[np.where(heart == 1)] = 2
  RT_Structs[np.where(lungs == 1)] = 1

  # RT_Structs[np.where(eso == 1)] = 1
  # RT_Structs[np.where(cord == 1)] = 2
  # RT_Structs[np.where(heart == 1)] = 3
  # RT_Structs[np.where(lungs == 1)] = 4

  maxm = dose.max()
  minm = dose.min()
  if maxm > 0:
    dose /= (maxm - minm)

  # Some CBCT/CT pairs have range > 1. Rescale (clip?) these to have range [0 1]
  maxm = cbct.max()
  if maxm > 1:
    cbct /= maxm
    print('rescaling ...')
  maxm = ct.max()
  if maxm > 1:
    ct /= maxm
    print('rescaling ...')

  out_path = os.path.join(out_folder, '{}'.format(subcase))
  np.savez(out_path, CT=ct, CBCT=cbct, DOSE=dose, RTSTRUCTS=RT_Structs)

def process_case(base_dir, case, out_dir):
  # Process case with case_name; Has 23 sub-cases
  # All sub-cases have 5 strucsts + dose names same
  # CBCT and CT names need to be identified correctly
  # Remove cases with sigmoid contrast and sharpening and cases with alpha, beta = 0,0.5/0,1

  print("Case: ", case)
  subcases = os.listdir(os.path.join(base_dir, case))
  num_subcases = 0
  for idx, subcase in enumerate(subcases):

    if '_pCT_OSSART_' in subcase:
      if subcase.endswith('_1') or subcase.endswith('_2'):
        continue
    else:
      if subcase.endswith('_3') or subcase.endswith('_4'):
        continue
    num_subcases += 1
    print("\tSubcase: ", num_subcases, " ", subcase)

    files = os.listdir(os.path.join(base_dir, case, subcase))
    for file in files:
      # print("\t\t File: ", file)
      filename = os.path.join(base_dir, case, subcase, file)
      if 'Heart' in file:
        heart,_ = nrrd.read(filename)
        heart = np.swapaxes(heart, 0, -1) # X,Y,Z to Z,Y,X
      elif 'Lungs' in file:
        lungs,_ = nrrd.read(filename)
        lungs = np.swapaxes(lungs, 0, -1)
      elif 'Cord' in file:
        cord,_ = nrrd.read(filename)
        cord = np.swapaxes(cord, 0, -1)
      elif 'Eso' in file:
        eso,_ = nrrd.read(filename)
        eso = np.swapaxes(eso, 0, -1)
      elif 'GTV' in file:
        gtv,_ = nrrd.read(filename)
        gtv = np.swapaxes(gtv, 0, -1)
      elif 'dose' in file:
        dose,_ = nrrd.read(filename)
        dose = np.swapaxes(dose, 0, -1)
      elif 'CT_plan_50' in file:
        ct,_ = nrrd.read(filename)
        ct = np.swapaxes(ct, 0, -1)
      elif '_pCT_aug' in file:
        ct,_ = nrrd.read(filename)
        ct = np.swapaxes(ct, 0, -1)
      elif '_img_aug' in file:
        cbct,_ = nrrd.read(filename)
        cbct = np.swapaxes(cbct, 0, -1)
      else:
        cbct,_ = nrrd.read(filename)
        cbct = np.swapaxes(cbct, 0, -1)

    save_case_combined_rtstructs(heart, lungs, cord, eso, cbct, ct, dose, subcase, out_dir)


def save_case_combined_rtstructs_aapm(heart, lungs, cord, eso, cbct, ct, subcase, out_folder):
  # Get border from CT, crop and resize all using the border
  grid = get_border(ct)
  ct = crop_border_fix_size(ct, grid, is_mask=False)
  cbct = crop_border_fix_size(cbct, grid, is_mask=False)
  heart = crop_border_fix_size(heart, grid, is_mask=True)
  lungs = crop_border_fix_size(lungs, grid, is_mask=True)
  eso = crop_border_fix_size(eso, grid, is_mask=True)
  cord = crop_border_fix_size(cord, grid, is_mask=True)

  # Combine all rtstructs into one image:
  # BG = 0, Eso = 4, Cord = 3, Heart = 2, Lungs = 1
  RT_Structs = np.zeros((128, 128, 128), dtype=np.uint8)
  RT_Structs[np.where(eso == 1)] = 4
  RT_Structs[np.where(cord == 1)] = 3
  RT_Structs[np.where(heart == 1)] = 2
  RT_Structs[np.where(lungs == 1)] = 1

  # RT_Structs[np.where(eso == 1)] = 1
  # RT_Structs[np.where(cord == 1)] = 2
  # RT_Structs[np.where(heart == 1)] = 3
  # RT_Structs[np.where(lungs == 1)] = 4

  # Some CBCT/CT pairs have range > 1. Rescale (clip?) these to have range [0 1]
  maxm = cbct.max()
  if maxm > 1:
    cbct /= maxm
    print('rescaling ...')
  maxm = ct.max()
  if maxm > 1:
    ct /= maxm
    print('rescaling ...')

  out_path = os.path.join(out_folder, '{}'.format(subcase))
  np.savez(out_path, CT=ct, CBCT=cbct, RTSTRUCTS=RT_Structs)


def process_case_aapm(base_dir, case, out_dir):
  # For AAPM data, subcases are extracted out into single parent folder and there is no GTV. Rest of the processing is same
  print("Case: ", case)
  subcase = case
  if '_pCT_OSSART_' in subcase:
    if subcase.endswith('_1') or subcase.endswith('_2'):
      return
  else:
    if subcase.endswith('_3') or subcase.endswith('_4'):
      return

  files = os.listdir(os.path.join(base_dir, case))
  for file in files:
    # print("\t\t File: ", file)
    filename = os.path.join(base_dir, case, file)
    if 'Heart' in file:
      heart,_ = nrrd.read(filename)
      heart = np.swapaxes(heart, 0, -1) # X,Y,Z to Z,Y,X
    elif 'Lungs' in file:
      lungs,_ = nrrd.read(filename)
      lungs = np.swapaxes(lungs, 0, -1)
    elif 'Cord' in file:
      cord,_ = nrrd.read(filename)
      cord = np.swapaxes(cord, 0, -1)
    elif 'Eso' in file:
      eso,_ = nrrd.read(filename)
      eso = np.swapaxes(eso, 0, -1)
    elif 'CT_plan_50' in file:
      ct,_ = nrrd.read(filename)
      ct = np.swapaxes(ct, 0, -1)
    elif '_pCT_aug' in file:
      ct,_ = nrrd.read(filename)
      ct = np.swapaxes(ct, 0, -1)
    elif '_img_aug' in file:
      cbct,_ = nrrd.read(filename)
      cbct = np.swapaxes(cbct, 0, -1)
    else:
      cbct,_ = nrrd.read(filename)
      cbct = np.swapaxes(cbct, 0, -1)

  save_case_combined_rtstructs_aapm(heart, lungs, cord, eso, cbct, ct, subcase, out_dir)

# MSK Train data
base_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT-to-CT-Translation/PseudoCBCTs/Summer_Interns/train_psCBCT_MSK'
case_file = './train_pseudocbct.txt'
out_dir = '../datasets/msk_aapm_combined/train'

if not os.path.exists(out_dir):
  os.makedirs(out_dir)

cases = get_caselist(case_file)
for case in cases:
  process_case(base_dir, case, out_dir)

# AAPM Train/Test data
# base_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/AAPM-Data/train/train_psCBCT_AAPM'
# #base_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/AAPM-Data/test/test_psCBCT_AAPM'
# out_dir = '../datasets/msk_aapm_combined/train'
#
# if not os.path.exists(out_dir):
#   os.makedirs(out_dir)
#
# cases = os.listdir(base_dir)
# for case in cases:
#   process_case_aapm(base_dir, case, out_dir)