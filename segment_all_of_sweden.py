import os
import sys
import random
import argparse
import datetime
import numpy as np
import tifffile
import time
from shutil import copyfile
from sklearn.metrics import jaccard_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from fcnpytorch.torchfcn.models import FCN8s as FCN8s
from utils import StatCollector
import imageio

"""
This script is used to evaluate a wetland model on every single 100x100-patch spanning Sweden,
i.e. to perform an all-of-Sweden-covering wetland mapping.

NOTE: The model is evaluated on disjoint CROP_SIZE x CROP_SIZE=100x100 rectangles spanning all of Sweden, so
there may be "border" effects between individual crops in the final result. To alleviate this, see the
flags START_H and START_W below (defaults to 0 each). By running this script with a few different settings
for START_H, START_W, one can compute on overlapping rectangles, that can later be averaged to reduce
border effects
"""


# Global vars
BASE_PATH_DATA = '../wetlands-data'
BASE_PATH_LOG = '../log'
SEED = 0
USE_GPU = True
CROP_SIZE = 100  # This size is assumed in this project
BATCH_SIZE = 64
PRED_ONLY_FG_BG = False  # True --> all wetland types are grouped into a single 'super category' of wetland types
WETLANDS_TO_NOT_PREDICT = ['oppna_mosse', 'aapamyr', 'rikkarr', 'kallor']  # Add elements from ['oppna_mosse', 'aapamyr', 'rikkarr', 'hogmosse', 'kallor'] to suppress it / these from the set of classes to predict
BG_WEIGHT = 1.0  # 1.0 --> weight bg as rest (i.e. 1 / NBR_CLASSES), < 1.0 --> lower weight on bg
NUM_ITER = 9999999
#MODEL_LOAD_PATH = os.path.join(BASE_PATH_LOG, '2022-11-01_08-55-00', 'model_it_100000')
MODEL_LOAD_PATH = os.path.join(BASE_PATH_LOG, '2022-12-20_11-14-31', 'model_it_250000')
WETLAND_TO_GT_IDX = {'background': 0, 'hogmosse': 1, 'rikkarr': 2, 'oppna_mosse': 3, 'aapamyr': 4, 'kallor': 5}
WETLAND_CODES_TO_NAMES = {7110: 'hogmosse', 7111: 'hogmosse',
                          7230: 'rikkarr', 7231: 'rikkarr', 7232: 'rikkarr', 7233: 'rikkarr',
                          7140: 'oppna_mosse', 7141: 'oppna_mosse', 7142: 'oppna_mosse', 7143: 'oppna_mosse',
                          7310: 'aapamyr',
                          7160: 'kallor'}
START_H = 0
START_W = 0

parser = argparse.ArgumentParser(description='Segmenting Swedish wetlands')
parser.add_argument('--use_latest_folder', default=False, type=bool, 
                    help='create new folder or save in latest folder')
parser.add_argument('--start_idx', default=0, type=int, 
                    help='what indices (coords) to segment')
parser.add_argument('--nbr_outer', default=20, type=int, 
                    help='---')                
args = parser.parse_args()

# Set all tiff paths
tif_paths_input = [os.path.join(BASE_PATH_DATA, 'VatmarkerBasklass.tif'),
				   os.path.join(BASE_PATH_DATA, 'VMIobjekttyp.tif'),
				   os.path.join(BASE_PATH_DATA, 'NhSverige.tif'),
				   os.path.join(BASE_PATH_DATA, 'MarfuktighetsindexNMD.tif'),
				   os.path.join(BASE_PATH_DATA, 'SLUmarkfuktighet.tif'),
				   os.path.join(BASE_PATH_DATA, 'NMDbas.tif'),
				   os.path.join(BASE_PATH_DATA, 'Objekthojd05_5.tif'),
				   os.path.join(BASE_PATH_DATA, 'Objekttackning05_5.tif'),
				   os.path.join(BASE_PATH_DATA, 'Objekthojd5_45.tif'),
				   os.path.join(BASE_PATH_DATA, 'Objekttackning5_45.tif')]
tif_path_ntypes = os.path.join(BASE_PATH_DATA, 'NNK_YTAvatmark.tif')
key_path = os.path.join(BASE_PATH_DATA, 'NNKvatmarkID.txt')

# Create directory in which to save current run
if args.use_latest_folder:
	log_dir = sorted(os.listdir(BASE_PATH_LOG))[-1]
	log_dir = os.path.join(BASE_PATH_LOG, log_dir)
else:
	timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
	log_dir = os.path.join(BASE_PATH_LOG, timestamp)
stat_train_dir = os.path.join(log_dir, "train_stats")
save_pred_path = os.path.join(log_dir, 'pred_maps')
if not args.use_latest_folder:
	os.makedirs(stat_train_dir, exist_ok=False)
	copyfile("wetland_training.py", os.path.join(log_dir, "wetland_training.py"))
	os.makedirs(save_pred_path)

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set max number of threads to use to 10 (dgx1 limit)
torch.set_num_threads(10)

# Extract and set certain dimensionalities
dim_input = 10
if PRED_ONLY_FG_BG:
    NBR_CLASSES = 2
else:
    NBR_CLASSES = 6 - len(WETLANDS_TO_NOT_PREDICT)

# Setup loss
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss(reduce=False)

# Set up range over which to iterate
H = 146258
W = 64034
if True:
	ijs_all = []
	for i in range(START_H, H - CROP_SIZE, CROP_SIZE):
		for j in range(START_W, W - CROP_SIZE, CROP_SIZE):
			ijs_all.append([i, j])
else:
	# List all gt maps, to speed up (and save disk space for!) the input map save process
	list_gt_maps = np.sort(os.listdir(os.path.join(BASE_PATH_DATA, 'annot_maps_100x100')))
	ijs_all = [[int(vv.split('.')[0].split('_')[-2]), int(vv.split('.')[0].split('_')[-1])] for vv in list_gt_maps]
nbr_ijs = len(ijs_all)
data_per_round = nbr_ijs // args.nbr_outer

for outer_loop in range(1):  # range(args.nbr_outer):

	# Current indices
	ijs = ijs_all[args.start_idx * data_per_round : min((args.start_idx + 1) * data_per_round, nbr_ijs)]

	print("Reading nature/wetland type data")

	# Read keys-values
	print("READING KEYS")
	key_to_nature_code = {}
	with open(key_path) as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			if i == 0:
				continue
			key_to_nature_code[int(line.split(',')[1])] = int(line.split(',')[-1])
			
	# Read tif image and convert keys to nature values
	print("READING IMAGES")
	im = np.array(tifffile.imread(tif_path_ntypes))

	# Read current set of wetland and naturetype data
	tot_ctr = 0
	data_names = []
	ims_wetland = []
	ims_ntype = []
	for ij in ijs:
		i = ij[0]
		j = ij[1]
		tot_ctr += 1
		sub_im_w = im[i : i + CROP_SIZE, j : j + CROP_SIZE].astype(int)  # wetlands
		sub_im_n = im[i : i + CROP_SIZE, j : j + CROP_SIZE].astype(int)  # the other nature types
		for key in range(256):
			if key in key_to_nature_code:
				value = key_to_nature_code[key]
				if value in WETLAND_CODES_TO_NAMES:
					sub_im_w[sub_im_w == key] = value
				else:
					sub_im_w[sub_im_w == key] = 0
				if value not in WETLAND_CODES_TO_NAMES:
					sub_im_n[sub_im_n == key] = value
				else:
					sub_im_n[sub_im_n == key] = 0
			else:
				# Background (not a specific nature type)
				sub_im_w[sub_im_w == key] = 0
				sub_im_n[sub_im_n == key] = 0
				
		# Map from nature code to the {1, ..., N} range in ground truth
		# 'hogmosse' -- category 1
		sub_im_w[sub_im_w == 7110] = 1
		sub_im_w[sub_im_w == 7111] = 1
		# 'rikkarr' -- category 2
		sub_im_w[sub_im_w == 7230] = 2
		sub_im_w[sub_im_w == 7231] = 2
		sub_im_w[sub_im_w == 7232] = 2
		sub_im_w[sub_im_w == 7233] = 2
		# 'oppna_mosse' -- category 3
		sub_im_w[sub_im_w == 7140] = 3
		sub_im_w[sub_im_w == 7141] = 3
		sub_im_w[sub_im_w == 7142] = 3
		sub_im_w[sub_im_w == 7143] = 3
		# 'aapamyr' -- category 4
		sub_im_w[sub_im_w == 7310] = 4
		# 'kallor' -- category 5
		sub_im_w[sub_im_w == 7160] = 5
		
		# Append to respective lists
		ims_wetland.append(sub_im_w[:, :, np.newaxis])
		ims_ntype.append(sub_im_n[:, :, np.newaxis])
		
		# Keep track of names
		data_names.append('pred_map_%d_%d' % (i, j))
		
		# Display progress
		if tot_ctr % 100 == 0:
			print(len(ims_wetland), len(ims_ntype), len(ijs))

	# Clear up memory
	del im

	# Read current set of model inputs
	print("Reading and normalizing input data")
			
	# Read tif image and convert keys to nature values
	ims_input = []
	for tif_ctr, tif_path in enumerate(tif_paths_input):
		print("READING IMAGES")
		im = np.array(tifffile.imread(tif_path))
		H, W = im.shape

		if 'NhSverige' in tif_path:
			max_input = np.load(os.path.join(BASE_PATH_DATA, 'height_max.npy'))
		if 'VatmarkerBasklass' in tif_path:
			uniques = np.load(os.path.join(BASE_PATH_DATA, 'baseclass_uniques.npy'))
		elif 'VMIobjekttyp' in tif_path:
			uniques = np.load(os.path.join(BASE_PATH_DATA, 'vmi_uniques.npy'))
		elif 'MarfuktighetsindexNMD' in tif_path:
			uniques = np.load(os.path.join(BASE_PATH_DATA, 'soilmoistindex_uniques.npy'))
		elif 'SLUmarkfuktighet' in tif_path:
			uniques = np.load(os.path.join(BASE_PATH_DATA, 'soilmoist_uniques.npy'))
		elif 'NMDbas' in tif_path:
			uniques = np.load(os.path.join(BASE_PATH_DATA, 'nmd_uniques.npy'))
		elif 'Objekthojd05_5' in tif_path:
			uniques = np.load(os.path.join(BASE_PATH_DATA, 'bush_uniques.npy'))
		elif 'Objekttackning05_5' in tif_path:
			uniques = np.load(os.path.join(BASE_PATH_DATA, 'bush_cover_uniques.npy'))
		elif 'Objekthojd5_45' in tif_path:
			uniques = np.load(os.path.join(BASE_PATH_DATA, 'tree_uniques.npy'))
		elif 'Objekttackning5_45' in tif_path:
			uniques = np.load(os.path.join(BASE_PATH_DATA, 'tree_cover_uniques.npy'))

		tot_ctr = 0
		ims = []
		for ij in ijs:
			i = ij[0]
			j = ij[1]
			tot_ctr += 1
			sub_im = im[i : i + CROP_SIZE, j : j + CROP_SIZE]

			# Re-write data as appropriate
			if 'NhSverige' in tif_path:
				sub_im[sub_im < 0] = 0
			if 'Objekthojd' in tif_path:
				sub_im[sub_im == 255] = 0
			elif 'Objekttackning' in tif_path:
				sub_im[sub_im == 255] = 0
			elif 'VMI' in tif_path:
				sub_im[sub_im == 128] = 0
			elif 'SLUmarkfuktighet' in tif_path:
				sub_im[sub_im != 255] += 1  # instead of 0, ..., N-1 --> 1, ..., N
				sub_im[sub_im == 255] = 0
			elif 'MarfuktighetsindexNMD' in tif_path:
				sub_im[sub_im < 0] = 0
				sub_im = np.round(sub_im).astype(int)

			# Setup input map values to correct range
			if 'NhSverige' not in tif_path:
				max_input = len(uniques)
				for j, unq in enumerate(uniques):
					sub_im[sub_im == unq] = j
			sub_im = sub_im[:, :, np.newaxis] / max_input

			# Append to lists
			ims.append(sub_im)
			
			# Display progress
			if tot_ctr % 100 == 0:
				print(len(ims), len(ijs))

		# Append to list of all inputs and clear up memory
		ims_input.append(ims)
		del im
		del ims

	# Bring it all together
	combined_maps = [None for _ in range(len(ijs))]
	for i in range(len(ijs)):
		combined_map = np.concatenate([ims_input[0][i], ims_input[1][i], ims_input[3][i], ims_input[4][i],
									   ims_input[5][i], ims_input[6][i], ims_input[7][i], ims_input[8][i],
									   ims_input[9][i], ims_input[2][i], ims_wetland[i], ims_ntype[i]], axis=2)
		combined_maps[i] = combined_map
	del ims_input
	del ims_wetland
	del ims_ntype

	# Split into train-val
	nbr_examples = len(data_names)
	assert nbr_examples == len(combined_maps)
	frame_ctr = 0

	# Setup model
	model = FCN8s(n_class=NBR_CLASSES, dim_input=dim_input, weight_init='normal')
	model.load_state_dict(torch.load(MODEL_LOAD_PATH))
	model.to(device)

	# Setup StatCollector
	sc = StatCollector(stat_train_dir, NUM_ITER, 10)
	sc.register('CE_loss', {'type': 'avg', 'freq': 'step'})
	sc.register('mIoU', {'type': 'avg', 'freq': 'step'})
	sc.register('Recall', {'type': 'avg', 'freq': 'step'})
	sc.register('Precision', {'type': 'avg', 'freq': 'step'})
	if True:  # Too much printing to show over all classes..?
		for i in range(NBR_CLASSES):
			sc.register('IoU_' + str(i), {'type': 'avg', 'freq': 'step'})
			sc.register('Recall_' + str(i), {'type': 'avg', 'freq': 'step'})
			sc.register('Precision_' + str(i), {'type': 'avg', 'freq': 'step'})
	sc.register('Class-dist gt', {'type': 'avg', 'freq': 'step'})
	sc.register('Entropy', {'type': 'avg', 'freq': 'step'})

	# Setup re-mapping of GT if omitting some wetland types
	wetland_idxs_to_not_predict = [WETLAND_TO_GT_IDX[key] for key in WETLANDS_TO_NOT_PREDICT]
	wetland_idxs_remaining = [wetland_idx for wetland_idx in [0, 1, 2, 3, 4, 5] if wetland_idx not in wetland_idxs_to_not_predict]

	def _forward_compute(sc, will_visualize=False):

		global frame_ctr

		# Format current batch (including ground truth)
		data_batch = np.zeros((BATCH_SIZE, dim_input, CROP_SIZE, CROP_SIZE), dtype=np.float32)
		gt_batch = np.zeros((BATCH_SIZE, CROP_SIZE, CROP_SIZE), dtype=np.float32)
		support_batch = np.zeros((BATCH_SIZE, CROP_SIZE, CROP_SIZE), dtype=bool)
		b = 0
		data_names_batch = []
		while b < BATCH_SIZE:

			# Sample image-gt pair
			if frame_ctr >= len(data_names):
				return None, None
			data_name = data_names[frame_ctr]
			data_names_batch.append(data_name)
			data_map = combined_maps[frame_ctr]
			frame_ctr += 1
			input_map = data_map[:, :, :dim_input]
			gt_map = data_map[:, :, dim_input]
			naturetype_map = data_map[:, :, dim_input+1]
			naturetype_map += gt_map  # Naturetype map represents the union of wetland + other naturetypes!
			naturetype_map = naturetype_map > 0
			if PRED_ONLY_FG_BG:
				# In this case we just want to classify 'wetland' vs 'not wetland'
				gt_map[gt_map > 0] = 1
			else:
				# Cancel out all wetland types we don't want to predict into a
				# background category
				for wetland_type in WETLANDS_TO_NOT_PREDICT:
					gt_map[gt_map == WETLAND_TO_GT_IDX[wetland_type]] = 0
				# Remap gt
				for wetland_idx in wetland_idxs_remaining:
					gt_map[gt_map == wetland_idx] = wetland_idxs_remaining.index(wetland_idx)

			# Track GT class stats
			curr_class_stat = np.zeros(NBR_CLASSES)
			curr_class_stat = np.array([np.count_nonzero(gt_map == x) / CROP_SIZE / CROP_SIZE for x in range(NBR_CLASSES)])
			sc.s('Class-dist gt').collect(curr_class_stat)

			# Extract crop and the measurement points that are inside the crop
			img_crop = input_map
			naturetype_crop = naturetype_map[:, :, np.newaxis]
			curr_gt_batch = gt_map

			# Perform data augmentation
			curr_data_batch = np.transpose(img_crop, [2, 0, 1])
			curr_naturetype_batch = np.transpose(naturetype_crop, [2, 0, 1])

			if b == BATCH_SIZE - 1:
				out_naturetype = curr_naturetype_batch

			# Track support for proper loss computation
			naturetype_non_bg = np.squeeze(curr_naturetype_batch > 0)
			support_h, support_w = np.nonzero(naturetype_non_bg)
			support_batch[b, support_h, support_w] = 1

			# Insert batch elements
			data_batch[b, :, :, :] = curr_data_batch
			gt_batch[b, :, :] = curr_gt_batch
			b += 1

		# Send to device (typically GPU)
		data_batch = torch.tensor(data_batch).to(device)
		# data_batch[:, 5:9, :, :] = 0  # zero out bush-tree-map
		gt_batch = torch.tensor(gt_batch, dtype=torch.long).to(device)
		support_batch = torch.tensor(support_batch, dtype=torch.long).to(device)

		# Forward the batch through the model, then compute the loss
		map_pred = model(data_batch)
		map_probs = nn.Softmax(dim=1)(map_pred)
		map_probs_np = map_probs.cpu().detach().numpy()
		err_pred = criterion(map_pred, gt_batch)
		err_pred *= support_batch
		err_pred = torch.sum(err_pred)
		if torch.count_nonzero(support_batch) > 0:
			err_pred /= torch.count_nonzero(support_batch)

		# After computing loss, ensure map_pred is argmaxed into class predictions
		map_pred = torch.argmax(map_pred, dim=1)

		# Save all predictions to log folder
		for i, data_name in enumerate(data_names_batch):
			imageio.imwrite(os.path.join(save_pred_path, data_name) +'.tif', map_probs_np[i, 1, :, :])  # Save only the prob of the predicted fg class

		# Return
		out_pred = map_pred.cpu().detach().numpy()[-1, :, :]
		out_gt = gt_batch.cpu().detach().numpy()[-1, :, :]
		out_support = support_batch.cpu().detach().numpy()[-1, :, :]
		out_in = np.squeeze(data_batch.cpu().detach().numpy()[-1, :, :, :])
		out_naturetype = np.squeeze(out_naturetype)

		# Compute separate stats per prediction type (background, wetland #1, ..., wetland #N)
		# For each type, compute iou, precision and recall
		support_flat = support_batch.cpu().contiguous().view(-1).numpy()  # make it 1D
		if np.any(support_flat):
			
			# Compute mIoU
			map_pred_flat = map_pred.cpu().contiguous().view(-1).numpy()  # make it 1D
			gt_flat = gt_batch.cpu().contiguous().view(-1).numpy()  # make it 1D
			# Only compute mIoU in support area
			miou = jaccard_score(gt_flat, map_pred_flat, average='macro', sample_weight=support_flat.astype(float))

			map_pred_flat = map_pred_flat[support_flat > 0]
			gt_flat = gt_flat[support_flat > 0]
			iou_prec_recs = np.nan * np.ones((3, NBR_CLASSES))
			for i in range(NBR_CLASSES):
				gt_flat_i = gt_flat == i
				nnz_gt_flat_i = np.count_nonzero(gt_flat_i)
				if nnz_gt_flat_i == 0:
					continue
				map_pred_flat_i = map_pred_flat == i
				intersec_i = np.logical_and(gt_flat_i, map_pred_flat_i)
				union_i = np.logical_or(gt_flat_i, map_pred_flat_i)
				nnz_intersec_i = np.count_nonzero(intersec_i)
				iou_i = nnz_intersec_i / np.count_nonzero(union_i)
				iou_prec_recs[0, i] = iou_i
				recall_i = nnz_intersec_i / nnz_gt_flat_i
				iou_prec_recs[2, i] = recall_i
				nnz_map_pred_flat_i = np.count_nonzero(map_pred_flat_i)
				if nnz_map_pred_flat_i > 0:
					prec_i = nnz_intersec_i / nnz_map_pred_flat_i
					iou_prec_recs[1, i] = prec_i

			# Track stats
			sc.s('CE_loss').collect(err_pred.item())
			sc.s('mIoU').collect(miou)
			if True:
				for i in range(NBR_CLASSES):
					if not np.isnan(iou_prec_recs[0, i]):
						sc.s('IoU_' + str(i)).collect(iou_prec_recs[0, i])
					if not np.isnan(iou_prec_recs[1, i]):
						sc.s('Precision_' + str(i)).collect(iou_prec_recs[1, i])
					if not np.isnan(iou_prec_recs[2, i]):
						sc.s('Recall_' + str(i)).collect(iou_prec_recs[2, i])
			mean_prec = np.nanmean(iou_prec_recs[1, :])
			if not np.isnan(mean_prec):
				sc.s('Precision').collect(mean_prec)
			mean_rec = np.nanmean(iou_prec_recs[2, :])
			if not np.isnan(mean_rec):
				sc.s('Recall').collect(mean_rec)
			mean_ent = np.nanmean(-np.sum(map_probs_np * np.log2(map_probs_np), axis=1))
			sc.s('Entropy').collect(mean_ent)

		return err_pred, None  #[out_in, out_pred, out_gt, out_support, out_naturetype, miou, iou_prec_recs]

	# Main training loop
	print("Starting training loop...")
	model.eval()  # Indicate that the model is to be eval'd
	for it in range(NUM_ITER):

		# Forward computation
		err_pred, _ = _forward_compute(sc)

		# Check for termination
		if err_pred is None:
			break

		# Track training and validation statistics
		if it % 25 == 0:
			sc.print()
			sc.save()
			print("Iter: %d / %d, %d" % (it * BATCH_SIZE, nbr_examples, outer_loop))
