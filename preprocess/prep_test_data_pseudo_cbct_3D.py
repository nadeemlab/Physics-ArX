#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:19:09 2020
Prepare Test Pseudo CBCT data for inference. Resize all to 128x128x128 and 
combine RT structs into single file.
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

def fix_size(filename, is_mask=True, expected_shape=(128,128,128)):
  
  img, imginfo = nrrd.read(os.path.join(filename))
  
  order = 0
  if is_mask is False:
    order = 1
    
  if img.shape != expected_shape:
    img = resize(img, expected_shape, order=order, preserve_range=True, anti_aliasing=False)
  img = np.swapaxes(img, 0, -1) # X,Y,Z to Z,Y,X
  
  return img
      
def save_case_combined_rtstructs(heart, lungs, cord, eso,
              gtv, cbct, ct, dose, subcase, out_folder):
  # Combine all rtstructs into one image:
    # BG = 0, Eso = 1, GTV = 2, Cord = 3, Heart = 4, Lungs = 5
  RT_Structs = np.zeros((128,128,128), dtype=np.uint8)
  RT_Structs[np.where(eso==1)]   = 4
  #RT_Structs[np.where(gtv==1)]   = 2
  RT_Structs[np.where(cord==1)]  = 3
  RT_Structs[np.where(heart==1)] = 2
  RT_Structs[np.where(lungs==1)] = 1

  maxm = dose.max()
  minm = dose.min()
  if maxm > 0:
    dose /= (maxm - minm)

  # Some CBCT/CT pairs have range > 1. Rescale (clip?) these to have range [0 1]
  maxm = cbct.max()
  if maxm > 1:
    cbct /= maxm
  maxm = ct.max()
  if maxm > 1:
    ct /= maxm
  
  out_path = os.path.join(out_folder, '{}'.format(subcase))
  np.savez(out_path, CT=ct, CBCT=cbct, DOSE=dose, RTSTRUCTS=RT_Structs)
    
def process_case(base_dir, case, out_dir):
  # Process case with case_name; All cases have 5 structs + dose + Plan CT and Pseudo CBCT
  print("Case: ", case)
  files = os.listdir(os.path.join(base_dir, case))
  for file in files:
    #print("\t\t File: ", file)
    filename = os.path.join(base_dir, case, file)
    #print(filename)
    if 'Heart' in file:
      heart = fix_size(filename, is_mask=True)
    elif 'Lungs' in file:
      lungs = fix_size(filename, is_mask=True)
    elif 'Cord' in file:
      cord = fix_size(filename, is_mask=True)
    elif 'Eso' in file:
      eso = fix_size(filename, is_mask=True)
    elif 'GTV' in file:
      gtv = fix_size(filename, is_mask=True)
    elif 'dose' in file:
      dose = fix_size(filename, is_mask=False)
    elif 'CT_plan_50' in file:
      ct = fix_size(filename, is_mask=False)
    elif '_pCT_OSSART_' in file:
      cbct = fix_size(filename, is_mask=False)
    else:
      print('\tUnknown file type ...')
        
  save_case_combined_rtstructs(heart, lungs, cord, eso, gtv, cbct, ct, dose, case, out_dir)
        
base_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT-to-CT-Translation/PseudoCBCTs/Summer_Interns/test_psCBCT_MSK'
case_file = './test_pseudocbct.txt'
#case_file = './test_pscbct_validation_added.txt'
out_dir = '../datasets/pseudo_cbct3D/test'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

cases = get_caselist(case_file)
for case in cases:
  process_case(base_dir, case, out_dir)