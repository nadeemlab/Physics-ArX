# Compute diff images with HU range to pinpoint bad MAE cases

import os
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize


def fix_size(img, is_mask=False, expected_shape=(128, 128, 128)):
	# Not all datasets are 128x128x128
	# Also get correct axes order

	order = 0
	if is_mask is False:
		order = 1

	if img.shape != expected_shape:
		img = resize(img, expected_shape, order=order, preserve_range=True, anti_aliasing=False)

	return img

def crop_border(gt_ct, pred_ct):
	# Some ground truth CT's have been cropped to match the field of view of CBCT's
	# Hence they have a black border in every slice. The CBCT doesn't have that
	# and since CBCT is the input the ouput translated CT doesn't have a border which
	# leads to lower SSIM values.

	# To overcome that, detect border in Ground truth CT and crop that from both
	# pred CT and Ground Truth CT to make a fair comparison
	img = gt_ct[0]
	mask = img > 0
	x, y = np.any(mask, 0), np.any(mask, 1)
	grid = np.ix_(y, x)
	masked = img[grid]

	out_pred_ct = np.zeros((gt_ct.shape[0], masked.shape[0], masked.shape[1]), gt_ct.dtype)

	for i in range(gt_ct.shape[0]):
		out_pred_ct[i] = pred_ct[i][grid]

	return out_pred_ct


pct_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT-to-CT-Translation/PseudoCBCTs/Summer_Interns/test_psCBCT_MSK'
w1_dir  = '/data/MSKCC-Intern-Summer-2020/codes/CBCT-to-CT-Translation/PseudoCBCTs/Summer_Interns/test_w1CBCT_MSK'
pred_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/PseudoCBCT2CT-3D/results/aapm_msk_stable_transp_conv_noisefree_week1/test_latest/npz_images'
out_dir = './diff_images'
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

cases = []
ignore_cases = ['0721', '3714', '7423', '00270510', '00696458']  # Cases that were in training set for earlier experiment
with open('./test_week1.txt', 'r') as f:
	for line in f:
		line = line.strip()
		if line in ignore_cases:
			continue
		cases.append(line)

#cases = ['35574899']
for case in cases:
	print("Processing case: ", case)
	filename = os.path.join(pct_dir, case, case + '_CT_plan_50.nrrd')
	ct = sitk.ReadImage(filename)
	ct = sitk.GetArrayFromImage(ct)
	ct = fix_size(ct)
	print(ct.shape)
	filename = os.path.join(w1_dir, case + '_CBCT_w1.nrrd')
	cbct = sitk.ReadImage(filename)
	cbct = sitk.GetArrayFromImage(cbct)
	cbct = fix_size(cbct)

	filename = os.path.join(pred_dir, case + '_CBCT2CT.nrrd')
	cbct2ct = sitk.ReadImage(filename)
	cbct2ct = sitk.GetArrayFromImage(cbct2ct)

	cbct = crop_border(ct, cbct)
	cbct2ct = crop_border(ct, cbct2ct)
	ct = crop_border(ct, ct)

	cbct = cbct*4095 - 1000
	ct = ct*4095 - 1000
	cbct2ct = cbct2ct*4095 - 1000

	filename = os.path.join(out_dir, case + '_CT.nrrd')
	sitk.WriteImage(sitk.GetImageFromArray(ct), filename)

	filename = os.path.join(out_dir, case + '_CBCT.nrrd')
	sitk.WriteImage(sitk.GetImageFromArray(cbct), filename)

	filename = os.path.join(out_dir, case + '_CBCT2CT.nrrd')
	sitk.WriteImage(sitk.GetImageFromArray(cbct2ct), filename)

	diff_cbct = cbct - ct
	diff_cbct2ct = cbct2ct - ct

	filename = os.path.join(out_dir, case + '_diff_cbct.nrrd')
	sitk.WriteImage(sitk.GetImageFromArray(diff_cbct), filename)

	filename = os.path.join(out_dir, case + '_diff_cbct2ct.nrrd')
	sitk.WriteImage(sitk.GetImageFromArray(diff_cbct2ct), filename)

