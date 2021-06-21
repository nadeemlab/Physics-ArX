# Compute MAE between uncropped CT and week 1 CBCTs for the same case.
# Using old 113 case data, *_rsc_w1 and *_rsc_pCT have same dimensions.

import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import similarity_metrics as measures


def get_caselist(case_file, ignore_cases):
	# Get list of cases from case_file (train/test/valid)
	case_list = []
	with open(case_file, 'r') as f:
		for line in f:
			line = line.strip()
			if line.split('_')[0] in ignore_cases:
				continue
			case_list.append(line.strip())
	return case_list


def fix_size(itk_img, ref_itk, is_mask=False):
	# Not all datasets are 128x128x128
	# Also get correct axes order
	img = sitk.GetArrayFromImage(itk_img)
	tar_size = ref_itk.GetSize()
	expected_shape = (tar_size[2], tar_size[1], tar_size[0])

	order = 0
	if is_mask is False:
		order = 1

	img = resize(img, expected_shape, order=order, preserve_range=True, anti_aliasing=False)
	img = sitk.GetImageFromArray(img)

	img.SetOrigin(ref_itk.GetOrigin())
	img.SetDirection(ref_itk.GetDirection())
	img.SetSpacing(ref_itk.GetSpacing())

	return img


def add_sCT_to_pCT(pCT, sCT):
	# Add the translated synthetic CT (or CBCT) to the correct cropped place in the original full resolution pCT
	start = pCT.TransformPhysicalPointToIndex(sCT.GetOrigin())  # Start point indexes in pCT (w, h, d)
	end = (start[0] + sct.GetSize()[0], start[1] + sCT.GetSize()[1], start[2] +sCT.GetSize()[2])

	pCT_arr = sitk.GetArrayFromImage(pCT)
	sCT_arr = sitk.GetArrayFromImage(sCT)

	pCT_arr[start[2]:end[2], start[1]:end[1], start[0]:end[0]] = sCT_arr

	pCT_sCT = sitk.GetImageFromArray(pCT_arr)
	pCT_sCT.SetOrigin(pCT.GetOrigin())
	pCT_sCT.SetDirection(pCT.GetDirection())
	pCT_sCT.SetSpacing(pCT.GetSpacing())

	return pCT_sCT

def create_border(ct, cbct):
	# Use the dark region around the CT to create a 0 border around both CT/CBCTs

	img = ct[0]
	mask = img > 0.1
	x, y = np.any(mask, 0), np.any(mask, 1)
	grid = np.ix_(y, x)

	ct_border = np.zeros_like(ct)
	cbct_border = np.zeros_like(ct)

	for i in range(ct.shape[0]):
		ct_border[i][grid] = ct[i][grid]
		cbct_border[i][grid] = cbct[i][grid]

	return ct_border, cbct_border

pct_in_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT-to-CT-Translation/data/MSKCC-Data'
scbct_in_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/PseudoCBCT2CT-3D/results/aapm_msk_stable_transp_conv_noisefree_psCBCT/test_latest/npz_images'  # Contains results of applciation of aapm_msk model on week 1 testing data
pscbct_in_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT-to-CT-Translation/PseudoCBCTs/Summer_Interns/test_psCBCT_MSK'

ignore_cases = ['1217', '6228', '6407', '35186959', '38003994']  # Contrast enhanced cases with brighter histogram
case_ids_file = './test_pseudocbct.txt'
cases = get_caselist(case_ids_file, ignore_cases)

mae_metrics = np.zeros((len(cases), 4), dtype=np.float64)

for idx, case in enumerate(cases):
	dset_name = case.split('_')[0]
	filename = os.path.join(pct_in_dir, dset_name, dset_name + '_rsc_pCT.nrrd')
	ct = sitk.ReadImage(filename)

	files = os.listdir(os.path.join(pscbct_in_dir, case))
	for file in files:
		if '_pCT_OSSART' in file:
			filename = file

	filename = os.path.join(pscbct_in_dir, case, filename)
	cbct = sitk.ReadImage(filename)

	filename = os.path.join(scbct_in_dir, case + '_CBCT2CT.nrrd')
	sct = sitk.ReadImage(filename)

	sct = fix_size(sct, cbct, is_mask=False)

	#pCT_sCT = add_sCT_to_pCT(ct, sct)
	pCT_sCT = add_sCT_to_pCT(ct, cbct)  # CBCT vs pCT

	ct = sitk.GetArrayFromImage(ct)
	pCT_sCT = sitk.GetArrayFromImage(pCT_sCT)

	ssim = measures.ssim(ct, pCT_sCT)

	ct = ct * 4095 - 1000
	pCT_sCT = pCT_sCT * 4095 - 1000
	mae = np.mean(np.abs(ct - pCT_sCT))
	psnr = measures.psnr(ct, pCT_sCT, data_range=4095)
	rmse = measures.rmse(ct, pCT_sCT)

	mae_metrics[idx, 0] = ssim
	mae_metrics[idx, 1] = mae
	mae_metrics[idx, 2] = psnr
	mae_metrics[idx, 3] = rmse

# Print metrics to file
metrics_file = open('./stats/psCBCT_CBCT_HU_stats_aapm_msk_noisefree.txt', 'w')

frmt_string = '{:<12s} ' + '{:<8s}'*4
print(frmt_string.format('Case', 'SSIM', 'MAE', 'PSNR', 'RMSE'), file=metrics_file)
print('-'*45, file=metrics_file)

print_str = '{:<12s}'
print_str += ' {:<.2f} '*4

for idx,case in enumerate(cases):
	print(print_str.format(case, *mae_metrics[idx]), file=metrics_file)

print('-'*45, file=metrics_file)

avg  = np.mean(mae_metrics, axis=0)
minm = np.min(mae_metrics, axis=0)
maxm = np.max(mae_metrics, axis=0)
stddev  = np.std(mae_metrics, axis=0)
print(print_str.format('average', *avg), file=metrics_file)
print(print_str.format('Std:', *stddev), file=metrics_file)
print(print_str.format('minimum', *minm), file=metrics_file)
print(print_str.format('maximum', *maxm), file=metrics_file)

metrics_file.close()
