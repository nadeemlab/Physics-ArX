# Compute DICE scores for the synthetic CTs

import os
import nrrd
import numpy as np
from skimage.transform import resize
import similarity_metrics as measures


def get_caselist(case_file):
	# Get list of cases from case_file (train/test/valid)
	case_list = []
	with open(case_file, 'r') as f:
		for line in f:
			case_list.append(line.strip())
	return case_list


def fix_size(filename, is_mask=True, expected_shape=(128, 128, 128)):
	# Not all datasets are 128x128x128
	# Also get correct axes order
	img, imginfo = nrrd.read(filename)

	order = 0
	if is_mask is False:
		order = 1

	if img.shape != expected_shape:
		img = resize(img, expected_shape, order=order, preserve_range=True, anti_aliasing=False)
	img = np.swapaxes(img, 0, -1)  # X,Y,Z to Z,Y,X

	return img


def get_combined_rtstructs(heart, lungs, cord, eso):
	# Combine all rtstructs into one image:
	# BG = 0, Eso = 1, Cord = 2, Heart = 3, Lungs = 4
	RT_Structs = np.zeros((128, 128, 128), dtype=np.uint8)
	RT_Structs[np.where(eso == 1)] = 4
	RT_Structs[np.where(cord == 1)] = 3
	RT_Structs[np.where(heart == 1)] = 2
	RT_Structs[np.where(lungs == 1)] = 1

	return RT_Structs


def get_gt_case(base_dir, case):
	# Process case with case_name; All cases have 5 structs + dose + Plan CT and Pseudo CBCT
	# CT needed to crop borders
	files = os.listdir(os.path.join(base_dir, case))
	for file in files:
		filename = os.path.join(base_dir, case, file)
		if 'Heart' in file:
			heart = fix_size(filename, is_mask=True)
		elif 'Lungs' in file:
			lungs = fix_size(filename, is_mask=True)
		elif 'Cord' in file:
			cord = fix_size(filename, is_mask=True)
		elif 'Eso' in file:
			eso = fix_size(filename, is_mask=True)
		elif 'CT_plan_50' in file:
			ct = fix_size(filename, is_mask=False)
		else:
			pass

	# Get combined RTSTRUCTS
	RTSTRUCTS = get_combined_rtstructs(heart, lungs, cord, eso)

	return ct, RTSTRUCTS


def get_pred_case(pred_dir, case):
	pred_ct, info = nrrd.read(os.path.join(pred_dir, case + '_RTSTRUCTS.nrrd'))
	pred_ct = np.swapaxes(pred_ct, 0, -1)  # X,Y,Z to Z,Y,X

	return pred_ct


def crop_border(gt_ct, segmentation):
	# Some ground truth CT's have been cropped to match the field of view of CBCT's
	# Hence they have a black border in every slice. The CBCT doesn't have that
	# and since CBCT is the input the ouput translated CT doesn't have a border which
	# leads to lower SSIM values. Same applied to manual segmentations.

	# To overcome that, detect border in Ground truth CT and crop that from both
	# pred CT and Ground Truth CT to make a fair comparison
	img = gt_ct[0]
	mask = img > 0
	x, y = np.any(mask, 0), np.any(mask, 1)
	grid = np.ix_(y, x)
	masked = img[grid]

	out_seg = np.zeros((128, masked.shape[0], masked.shape[1]), segmentation.dtype)

	for i in range(128):
		out_seg[i] = segmentation[i][grid]

	return out_seg


def get_all_dice(gt, pred):
	# Get dice scores for individual RTSTRUCTS
	dice_scores = []
	for i in range(1, 5):
		gt_mask = np.zeros_like(gt)
		pred_mask = np.zeros_like(pred)

		gt_mask[np.where(gt == i)] = 1
		pred_mask[np.where(pred == i)] = 1

		dice = measures.compute_dice(pred_mask, gt_mask)
		dice_scores.append(dice)

	return dice_scores

########################################################
############## Main script #############################
########################################################

# MSK Data
#case_ids_file = './test_pseudocbct.txt'
case_ids_file = './test_week1.txt'
metrics_filename = './stats/week1_aapm_msk_no_stabilization_dice_scores.txt'

gt_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT-to-CT-Translation/PseudoCBCTs/Summer_Interns/test_psCBCT_MSK'
#pred_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/PseudoCBCT2CT-3D/results/pseudocbct2ct3d_fixed_cases/test_latest/npz_images'
#pred_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/PseudoCBCT2CT-3D/results/pseudocbct2ct3d_fixed_cases/test_latest/post_processed' # Post processed; keep largest blob(s)
pred_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/PseudoCBCT2CT-3D/results/msk_aapm_combined_no_stablization_week1/test_latest/post_processed' # Post processed; keep largest blob(s)

# AAPM Data
# case_ids_file = './test_pseudocbct_aapm.txt'
# metrics_filename = 'AAPM_dice_scores_post_processed.txt'
#
# gt_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/AAPM-Data/test/test_psCBCT_AAPM'
# pred_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/PseudoCBCT2CT-3D/results/aapm_cbct2ct3d_fixed_cases/test_latest/npz_images/post_processed' # Post processed; keep largest blob(s)


cases1 = get_caselist(case_ids_file)
#cases = cases[:2]
#ignore_cases = ['1217', '6228', '6407', '35186959', '38003994']  # Contrast enhanced cases with brighter histogram
ignore_cases = ['0721', '3714', '7423', '00270510', '00696458']  # Cases that were in training set for earlier experiment
cases = []
for case in cases1:
	case_prefix = case.split('_')[0]
	if case_prefix in ignore_cases:
		continue
	cases.append(case)

metrics = np.zeros((len(cases), 4), dtype=np.float32)
metrics_file = open(metrics_filename, 'w')

for idx, case in enumerate(cases):
	# Get ground truth data (128x128x128)
	gt_ct, gt_rtstructs = get_gt_case(gt_dir, case)
	pred_rtstructs = get_pred_case(pred_dir, case)

	gt_rtstructs_cropped = crop_border(gt_ct, gt_rtstructs)
	pred_rtstructs_cropped = crop_border(gt_ct, pred_rtstructs)  # To compare cbct with ct

	#dice_scores = get_all_dice(gt_rtstructs, pred_rtstructs)
	dice_scores = get_all_dice(gt_rtstructs_cropped, pred_rtstructs_cropped)
	metrics[idx] = np.asarray(dice_scores, dtype=np.float32)

# Print metrics to file
metric_headings = ['Case', 'Lungs', 'Heart', 'Cord', 'Eso']
#heading = '{:<12s}'
#print_str = '{:<12s}'

heading = '{:<25s}'
print_str = '{:<25s}'
for i in range(4):
	heading += ' {:<5s} '
	print_str += ' {:<.2f}  '

print(heading.format(*metric_headings), file=metrics_file)
print("-"*40, file=metrics_file)

for i, case in enumerate(cases):
	print(print_str.format(case, *metrics[i]), file=metrics_file)

print("-"*40, file=metrics_file)
avg  = np.mean(metrics, axis=0)
minm = np.min(metrics, axis=0)
maxm = np.max(metrics, axis=0)
stddev  = np.std(metrics, axis=0)
print(print_str.format('average', *avg), file=metrics_file)
print(print_str.format('Std:', *stddev), file=metrics_file)
print(print_str.format('minimum', *minm), file=metrics_file)
print(print_str.format('maximum', *maxm), file=metrics_file)

metrics_file.close()

















































