#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 19:44:39 2021

@author: ndahiya
"""

import os
import tigre
import numpy as np
#from tigre import Ax
import tigre.algorithms as algs
from tigre.utilities import CTnoise
import tigre.utilities.gpu as gpu
import SimpleITK as sitk

def pCT_CBCT_reconstruction(in_dir):
    """
    Expects input directory to have folders, one folder for each patient. Each
    patient has different artifact added images which need to be reconstructed.
    """
    
    listGpuNames = gpu.getGpuNames()
    if len(listGpuNames) == 0:
      print("Error: No gpu found")
    else:
      for id in range(len(listGpuNames)):
        print("{}: {}".format(id, listGpuNames[id]))

    gpuids = gpu.getGpuIds(listGpuNames[0])
    print(gpuids)

    files = os.listdir(in_dir)
    patients = []
    for file in files:
        if os.path.isdir(os.path.join(in_dir, file)):
            patients.append(file)
            
    for idx, patient in enumerate(patients):
        pth = os.path.join(in_dir, patient, 'Artifacts')
        files = os.listdir(pth)
        artifacts = []
        fnames = ['0.5_0.5_rsc.nrrd', '0.5_0_rsc.nrrd', '0.5_1_rsc.nrrd', '0_0.5_rsc.nrrd', 
                  '0_1_rsc.nrrd', '1_0.5_rsc.nrrd', '1_0_rsc.nrrd']
        fnames = sorted(fnames)
        for file in files:
            for suffix in fnames:
                if suffix in file:
                    artifacts.append(file)
        artifacts = sorted(artifacts)
        
        # Now process 7 artifact added files
        for file, artifact in zip(fnames,artifacts):
            caseID = file[:-9]
            print(file, artifact, caseID)
            infile = os.path.join(in_dir, patient, 'Artifacts', artifact)
            
            #img_Artct, info1 = nrrd.read(infile)
            img_Artct_itk = sitk.ReadImage(infile)
            img_Artct = sitk.GetArrayFromImage(img_Artct_itk).astype(np.float32)
            sz = img_Artct_itk.GetSize()
            spc = img_Artct_itk.GetSpacing()
            #print(img_Artct.GetSize(), img_Artct.GetSpacing())
            #print(img_Artct.shape, sz)
            
            geo = tigre.geometry_default(high_resolution=False)
            angles=np.linspace(0,2*np.pi,500)
            
            geo.DSD = 1500;
            geo.nDetector = np.array((400, 400));
            geo.sDetector = np.array((400, 400));
            geo.dDetector = geo.sDetector / geo.nDetector
     
            geo.nVoxel = np.array((sz[2],sz[1],sz[0]))
     
            geo.sVoxel = np.array((sz[2]*float(spc[2]),sz[1]*float(spc[1]),sz[0]*float(spc[0])))
            geo.dVoxel = geo.sVoxel / geo.nVoxel
            #geo.dVoxel = np.array((spc[2],spc[1],spc[0]))
     
            #%geo.offOrigin=[info1.spaceorigin(1);info1.spaceorigin(2);info1.spaceorigin(3)];
     
            #geo.offDetector = np.array((0, -160))
            
            #print(geo)
            projections = tigre.Ax(img_Artct, geo, angles, 'interpolated', gpuids=gpuids)
            
            noise_projections = CTnoise.add(projections, Poisson=1e8, Gaussian=np.array([0, 4]))
            print(noise_projections.shape, noise_projections.min(), noise_projections.max())
            # Save projections data
            filename = os.path.join(in_dir, patient, 'Artifacts', patient + '_projections_' + caseID + '.npz')
            np.savez(filename, projection=noise_projections)
            
            # Reconstruct OSSART image, compute quality measures, plot and save all
            out_dir = os.path.join(in_dir, patient, 'Recons', 'QualMeasure')
            if not os.path.exists(out_dir):
              os.makedirs(out_dir)
              
            qual_measures = ['RMSE', 'CC', 'MSSIM', 'UQI']
            #qual_measures = ['RMSE', 'MSSIM', 'UQI']
            blcks = 20;
            niter = 30;
            imgOSSART = algs.ossart(noise_projections, geo, angles, niter, blocksize=blcks, gpuids=gpuids)  #, qualityOSSART Quameasopts=qual_measures, computel2=True,
            #print(qualityOSSART.shape)
            filename = os.path.join(in_dir, patient, 'Recons', patient + '_pCT_OSSART_' + caseID + '.nrrd')
            imgOSSART_itk = sitk.GetImageFromArray(imgOSSART)
            imgOSSART_itk.SetDirection(img_Artct_itk.GetDirection())
            imgOSSART_itk.SetSpacing(img_Artct_itk.GetSpacing())
            imgOSSART_itk.SetOrigin(img_Artct_itk.GetOrigin())
            sitk.WriteImage(imgOSSART_itk, filename)
        
in_dir = '../Sample-Patients'
pCT_CBCT_reconstruction(in_dir)


























