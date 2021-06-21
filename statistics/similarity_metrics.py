#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 20:10:23 2020
Implement 3D image similarity metrics. Use pre existing wherever possible.
@author: ndahiya
"""

from skimage.metrics import mean_squared_error, normalized_root_mse, structural_similarity, peak_signal_noise_ratio
import numpy as np


def mse(gt, pred):
	# Mean squared error, image order does'nt matter
	return mean_squared_error(gt, pred)


def rmse(gt, pred):
	# Root mean squared error, image order does'nt matter
	return np.sqrt(mse(gt, pred))


def nrmse(gt, pred):
	# Normalized root mean squared error, image order matters
	# NRMSE = RMSE * sqrt(N) / || gt || where || || is Frobenius norm and N = gt.size
	return normalized_root_mse(gt, pred)


def mae(gt, pred):
	# Mean absolute error
	return np.mean(np.abs(gt - pred))


def ssim(gt, pred, win_len=7):
	# Structural similarity index, image order does'nt matter
	return structural_similarity(gt, pred, win_size=win_len)


def psnr(gt, pred, data_range=None):
	# Peak signal to noise ratio
	return peak_signal_noise_ratio(gt, pred, data_range=data_range)

def compute_dice(pred, mask):
	intersection = np.sum(pred * mask)
	area_sum = np.sum(pred) + np.sum(mask)

	dice = (2 * intersection) / area_sum

	return dice


def bhattacharya_dist(gt, pred):
	# Bhattacharya distance between two images
	# Values expected in range [0 1]
	gt_hist, gt_bin_edges = np.histogram(gt, bins=256, range=(0, 1))
	pred_hist, pred_bin_edges = np.histogram(pred, bins=256, range=(0, 1))

	gt_probs = gt_hist / float(np.sum(gt_hist))
	pred_probs = pred_hist / float(np.sum(pred_hist))

	sq = 0
	for i in range(len(gt_hist)):
		sq += np.sqrt(gt_probs[i] * pred_probs[i])

	return -np.log(sq)


def zero_normalized_cross_correlation(gt, pred):
	# Zero Normalized Cross-Correlation

	num = np.sum((gt - np.mean(gt)) * (pred - np.mean(pred)))
	den = (gt.size - 1) * np.std(gt) * np.std(pred)

	dist_zncc = num / den

	return dist_zncc
