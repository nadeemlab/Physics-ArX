# Compute different hausdorff metrics

import os
import numpy as np
from skimage.transform import resize
from medpy.io import load, save
from medpy.metric import hd, hd95, asd, assd


def get_caselist(case_file):
	# Get list of cases from case_file (train/test/valid)
	case_list = []
	with open(case_file, 'r') as f:
		for line in f:
			case_list.append(line.strip())
	return case_list


def get_combined_rtstructs(heart, lungs, cord, eso):
	# Combine all rtstructs into one image:
	# BG = 0, Eso = 4, Cord = 3, Heart = 2, Lungs = 1
	RT_Structs = np.zeros(heart.shape, dtype=np.uint8)
	RT_Structs[np.where(eso == 1)] = 4
	RT_Structs[np.where(cord == 1)] = 3
	RT_Structs[np.where(heart == 1)] = 2
	RT_Structs[np.where(lungs == 1)] = 1
	# RT_Structs[np.where(eso == 1)] = 1
	# RT_Structs[np.where(cord == 1)] = 2
	# RT_Structs[np.where(heart == 1)] = 3
	# RT_Structs[np.where(lungs == 1)] = 4

	return RT_Structs


def get_case_data(gt_dir, pred_dir, case):
	# Process GT case with case_name; All cases have 5 structs + dose + Plan CT and Pseudo CBCT
	# CT needed to crop borders
	files = os.listdir(os.path.join(gt_dir, case))
	for file in files:
		filename = os.path.join(gt_dir, case, file)
		if 'Heart' in file:
			heart, hdr = load(filename)
		elif 'Lungs' in file:
			lungs, hdr = load(filename)
		elif 'Cord' in file:
			cord, hdr = load(filename)
		elif 'Eso' in file:
			eso, hdr = load(filename)
		else:
			pass

	# Get combined RTSTRUCTS
	gt_rtstructs = get_combined_rtstructs(heart, lungs, cord, eso)

	# Get pred case
	pred_rtstructs, pred_hdr = load(os.path.join(pred_dir, case + '_RTSTRUCTS.nrrd'))

	# Resize prediction to match ground truth
	pred_rtstructs = resize(pred_rtstructs, gt_rtstructs.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
	#save(gt_rtstructs, '1217_gt.nrrd', hdr)
	#save(pred_rtstructs, '1217_pred.nrrd', hdr)

	return gt_rtstructs, pred_rtstructs, hdr


########################################################
############## Main script #############################
########################################################

case_ids_file = './test_week1.txt'
gt_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT-to-CT-Translation/PseudoCBCTs/Summer_Interns/test_psCBCT_MSK'
#pred_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/PseudoCBCT2CT-3D/results/pseudocbct2ct3d_fixed_cases/test_latest/npz_images'
#pred_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/PseudoCBCT2CT-3D/results/msk_aapm_combined_eso_label_1_week1/test_latest/post_processed' # Post processed; keep largest blob(s)
pred_dir = '/data/MSKCC-Intern-Summer-2020/codes/CBCT2CT-Fall-2020/PseudoCBCT2CT-3D/results/msk_aapm_combined_no_stablization_week1/test_latest/post_processed' # Post processed; keep largest blob(s)

cases1 = get_caselist(case_ids_file)
#ignore_cases = ['1217', '6228', '6407', '35186959', '38003994']  # Contrast enhanced cases with brighter histogram
ignore_cases = ['0721', '3714', '7423', '00270510', '00696458']  # Cases that were in training set for earlier experiment
cases = []
for case in cases1:
	case_prefix = case.split('_')[0]
	if case_prefix in ignore_cases:
		continue
	cases.append(case)

metrics = np.zeros((3, len(cases), 4), dtype=np.float32) # 3 metrics for all cases for all 4 anatomies
for idx, case in enumerate(cases):
	print('Processing case: ', case)
	gt_rtstructs, pred_rtstructs, hdr = get_case_data(gt_dir, pred_dir, case)
	hd95_metrics = []
	hd_metrics = []
	#asd_metrics = []
	assd_metrics = []

	for i in range(1, 5):
		gt = gt_rtstructs==i
		pred = pred_rtstructs==i

		hd95_metrics.append(hd95(pred, gt, hdr.get_voxel_spacing()))
		hd_metrics.append(hd(pred, gt, hdr.get_voxel_spacing()))
		#asd_metrics.append(asd(pred, gt, hdr.get_voxel_spacing()))
		assd_metrics.append(assd(pred, gt, hdr.get_voxel_spacing()))

	metrics[0, idx] = np.asarray(hd95_metrics, dtype=np.float32)
	metrics[1, idx] = np.asarray(hd_metrics, dtype=np.float32)
	metrics[2, idx] = np.asarray(assd_metrics, dtype=np.float32)

hd95_metrics_file = open('./stats/week1_aapm_msk_no_stabilization_noisefree_HD95.txt', 'w')
hd_metrics_file = open('./stats/week1_aapm_msk_no_stabilization_noisefree_HD.txt', 'w')
assd_metrics_file = open('./stats/week1_aapm_msk_no_stabilization_noisefree_ASSD.txt', 'w')

# Print metrics to file
metric_headings = ['Case', 'Lungs', 'Heart', 'Cord', 'Eso']
#metric_headings = ['Case', 'Eso', 'Cord', 'Heart', 'Lungs']
heading = '{:<12s}'
print_str = '{:<12s}'

for i in range(4):
	heading += ' {:<5s} '
	print_str += ' {:<.2f}  '

print(heading.format(*metric_headings), file=hd95_metrics_file)
print("-"*40, file=hd95_metrics_file)

print(heading.format(*metric_headings), file=hd_metrics_file)
print("-"*40, file=hd_metrics_file)

print(heading.format(*metric_headings), file=assd_metrics_file)
print("-"*40, file=assd_metrics_file)

for idx, case in enumerate(cases):
	print(print_str.format(case, *metrics[0, idx]), file=hd95_metrics_file)
	print(print_str.format(case, *metrics[1, idx]), file=hd_metrics_file)
	print(print_str.format(case, *metrics[2, idx]), file=assd_metrics_file)

# Print averages
print("-"*40, file=hd95_metrics_file)
print("-"*40, file=hd_metrics_file)
print("-"*40, file=assd_metrics_file)

avg  = np.mean(metrics[0], axis=0)
minm = np.min(metrics[0], axis=0)
maxm = np.max(metrics[0], axis=0)
stddev  = np.std(metrics[0], axis=0)

print(print_str.format('average', *avg), file=hd95_metrics_file)
print(print_str.format('Std:', *stddev), file=hd95_metrics_file)
print(print_str.format('minimum', *minm), file=hd95_metrics_file)
print(print_str.format('maximum', *maxm), file=hd95_metrics_file)

avg  = np.mean(metrics[1], axis=0)
minm = np.min(metrics[1], axis=0)
maxm = np.max(metrics[1], axis=0)
stddev  = np.std(metrics[1], axis=0)

print(print_str.format('average', *avg), file=hd_metrics_file)
print(print_str.format('Std:', *stddev), file=hd_metrics_file)
print(print_str.format('minimum', *minm), file=hd_metrics_file)
print(print_str.format('maximum', *maxm), file=hd_metrics_file)

avg = np.mean(metrics[2], axis=0)
minm = np.min(metrics[2], axis=0)
maxm = np.max(metrics[2], axis=0)
stddev = np.std(metrics[2], axis=0)

print(print_str.format('average', *avg), file=assd_metrics_file)
print(print_str.format('Std:', *stddev), file=assd_metrics_file)
print(print_str.format('minimum', *minm), file=assd_metrics_file)
print(print_str.format('maximum', *maxm), file=assd_metrics_file)

hd95_metrics_file.close()
hd_metrics_file.close()
assd_metrics_file.close()























