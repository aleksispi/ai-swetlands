import os
import sys
import numpy as np
import tifffile
import cv2
import matplotlib
import matplotlib.pyplot as plt

"""
This script is used merge all individual-location wetland segmentations performed
using the segment_all_of_sweden.py script into one coherent tiff file.

Note: See the script write_geotiff.py for translating the tiff so that it reads correctly in GIS.
"""


BASE_PATH_LOG = '../log'
SAVE_BINARY_TIFF = True  # Takes significantly less disk space with a bool rather than float tiff
LOAD_NAME_EXISTING_TIFF = None  # Can be used to load another tiff, in order to average predictions made at slightly 'shifted' locations, to get rid of edge effects in predicted locations
IM_H = 146258
IM_W = 64034
CROP_SIZE = 100
SHORTCUT_FLOAT_TO_BOOL = True  # If saving a boolean tiff, this uses less complex code

# Set below to True in order to produce a png version of a tiff
if False:
	big_tiff = np.array(tifffile.imread('<tiff-name>.tiff')).astype(float)
	resized_image = cv2.resize(big_tiff, (5*640, 5*1463), interpolation = cv2.INTER_NEAREST)
	matplotlib.image.imsave('resized_tiff.png', (resized_image >= 0.5).astype(float) * 255)
	print("DONE")
	sys.exit(0)

tiff_folder = os.listdir(os.path.join(BASE_PATH_LOG, '2023-11-09_06-29-00', 'pred_maps'))
if SAVE_BINARY_TIFF:
	if LOAD_NAME_EXISTING_TIFF is None:
		big_tiff = np.zeros((IM_H, IM_W), dtype=bool)
	else:
		big_tiff = np.array(tifffile.imread(LOAD_NAME_EXISTING_TIFF))
		if SHORTCUT_FLOAT_TO_BOOL:
			tifffile.imwrite('big_tiff_binary.tif', 2*big_tiff >= 0.5)
			print(np.max(2*big_tiff))
			print("DONE SAVING BOOL")
			sys.exit(0)
else:
	big_tiff = np.zeros((IM_H, IM_W), dtype=np.float32)
max_tiff_h = 0
max_tiff_w = 0
for i, tiff_file in enumerate(tiff_folder):
	if i % 1000 == 0:
		print(i, len(tiff_folder))
	tiff_h = int(tiff_file.split('_')[-2])
	tiff_w = int(tiff_file.split('_')[-1].replace('.tif', ''))
	max_tiff_h = max(max_tiff_h, tiff_h)
	max_tiff_w = max(max_tiff_w, tiff_w)
	curr_tiff = np.array(tifffile.imread(os.path.join(BASE_PATH_LOG, '2023-11-09_06-29-00', 'pred_maps', tiff_file)))
	if SAVE_BINARY_TIFF and LOAD_NAME_EXISTING_TIFF is None:
		big_tiff[tiff_h : tiff_h + CROP_SIZE, tiff_w : tiff_w + CROP_SIZE] = curr_tiff >= 0.5
	elif LOAD_NAME_EXISTING_TIFF is not None:
		curr_big = big_tiff[tiff_h : tiff_h + CROP_SIZE, tiff_w : tiff_w + CROP_SIZE]
		# OBS: THE CODE IS ASSUMED TO BE RUN IN SUCH A WAY THAT:
		# 1. FIRST ONE CREATS THE (0,0)-BASED MAP ("if 1" below)
		# 2. THEN ONE CREATS THE (50,0)-BASED MAP (the first "elif 1" below), averaging it with the previous (0,0)-based map
		# 3. THEN ONE CREATS THE (0,50)-BASED MAP (the last "elif 1" below), averaging it with the previous (0,0)+(50,0)-averaged map
		if 1:
			# (0,0)
			big_tiff[tiff_h : tiff_h + CROP_SIZE, tiff_w : tiff_w + CROP_SIZE] = curr_tiff
		elif 1:
			# (50,0)
			assert tiff_h >= 50 and tiff_h <= 146150
			if tiff_h == 50:  # In the start (50,0)-setup, h \in [0, 50) need to be treated differently (keep only the previous preds, no changes for that part)
				# big_tiff[0 : 50, tiff_w : tiff_w + CROP_SIZE] = big_tiff[0 : 50, tiff_w : tiff_w + CROP_SIZE] -- redundant, commented it out!
				big_tiff[50 : CROP_SIZE + 50, tiff_w : tiff_w + CROP_SIZE] = 0.5 * big_tiff[50 : CROP_SIZE + 50, tiff_w : tiff_w + CROP_SIZE] + 0.5 * curr_tiff
			elif tiff_h == 146150:  # In the start (50,0)-setup, h in (outer realm) need to be treated differently (use only new preds here)
				big_tiff[146250-100 : 146250-50, tiff_w : tiff_w + CROP_SIZE] = 0.5 * big_tiff[146250-100 : 146250-50, tiff_w : tiff_w + CROP_SIZE] + 0.5 * curr_tiff[:50, :]
				big_tiff[146250-50 : 146250, tiff_w : tiff_w + CROP_SIZE] = curr_tiff[50:, :]
			else:
				big_tiff[tiff_h : tiff_h + CROP_SIZE, tiff_w : tiff_w + CROP_SIZE] = 0.5 * curr_big + 0.5 * curr_tiff
		elif 1:
			# TODO: CURRENTLY RESULTS IN MAX VALUE 0.5 INSTEAD OF 1
			# (0,50)
			assert tiff_w >= 50 and tiff_w <= 63850
			if tiff_h == 0:
				if tiff_w == 50:
					#big_tiff[0 : 50, 0 : 50] = big_tiff[0 : 50, 0 : 50]  # redundant, commented it out!
					big_tiff[0 : 50, 50 : CROP_SIZE + 50] = 0.5 * big_tiff[0 : 50, 50 : CROP_SIZE + 50] + 0.5 * curr_tiff[0 : 50, :]
					big_tiff[50 : CROP_SIZE, 50 : CROP_SIZE + 50] = 2/3 * big_tiff[50 : CROP_SIZE, 50 : CROP_SIZE + 50] + 1/3 * curr_tiff[50 : CROP_SIZE, :]
					#big_tiff[50 : CROP_SIZE, 0 : 50] = big_tiff[50 : CROP_SIZE, 0 : 50]  # redundant, commented it out!
				else:
					big_tiff[0 : 50, tiff_w : tiff_w + CROP_SIZE] = 0.5 * big_tiff[0 : 50, tiff_w : tiff_w + CROP_SIZE] + 0.5 * curr_tiff[0 : 50, :]
					big_tiff[50 : CROP_SIZE, tiff_w : tiff_w + CROP_SIZE] = 2/3 * big_tiff[50 : CROP_SIZE, tiff_w : tiff_w + CROP_SIZE] + 1/3 * curr_tiff[50 : CROP_SIZE, :]
			elif tiff_w == 50:  # In the start (0,50)-setup, w \in [0, 50) need to be treated differently (keep only the previous preds, no changes for that part)
				# big_tiff[tiff_h : tiff_h + CROP_SIZE, 0 : 50] = big_tiff[tiff_h : tiff_h + CROP_SIZE, 0 : 50]   # redundant, commented it out!
				big_tiff[tiff_h : tiff_h + CROP_SIZE, 50 : CROP_SIZE + 50] = 2/3 * big_tiff[tiff_h : tiff_h + CROP_SIZE, 50 : CROP_SIZE + 50] + 1/3 * curr_tiff
			else:
				big_tiff[tiff_h : tiff_h + CROP_SIZE, tiff_w : tiff_w + CROP_SIZE] = 2/3 * curr_big + 1/3 * curr_tiff
			# NOTE: THERE IS NO "elif tiff_w == max_w case", because actually the max_w "belongs" to the (0,0) and (50,0) setups, i.e. only 100% of the old stuff in there
	else:
		big_tiff[tiff_h : tiff_h + CROP_SIZE, tiff_w : tiff_w + CROP_SIZE] = curr_tiff
print("SAVING BIG TIFF")
if SAVE_BINARY_TIFF:
	tifffile.imwrite('big_tiff_binary.tif', big_tiff >= 0.5)
else:
	tifffile.imwrite('big_tiff.tif', big_tiff)
print("DONE MERGING TIFFS")
