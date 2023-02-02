import os
import sys
import random
import datetime
import numpy as np
import time
from shutil import copyfile
from sklearn.metrics import jaccard_score
from skimage import measure
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from fcnpytorch.torchfcn.models import FCN8s as FCN8s
from utils import StatCollector


# Global vars
BASE_PATH_DATA = '../wetlands-data'
BASE_PATH_LOG = '../log'
SEED = 0
USE_GPU = True
CROP_SIZE = 100  # This size is assumed in this project
BATCH_SIZE = 64
PRED_ONLY_FG_BG = False  # True --> all wetland types are grouped into a single 'super category' of wetland types, so the model is wetland-agnostic, only predicting "wetland" or "not wetland"
WETLANDS_TO_NOT_PREDICT = []  # Add elements from ['oppna_mosse', 'aapamyr', 'rikkarr', 'hogmosse', 'kallor'] to suppress it / these from the set of classes to predict. For example, to train a högmosse-only model, set WETLANDS_TO_NOT_PREDICT = ['oppna_mosse', 'aapamyr', 'rikkarr', 'kallor']
BG_WEIGHT = 1.0  # 1.0 --> weight bg as rest (i.e. 1 / NBR_CLASSES), < 1.0 --> lower weight on bg
REQUIRE_AT_LEAST_ONE_PER_BATCH = False  # True --> include more of under-represented classes in training
NUM_TRAIN_ITER = 250000
OPTIMIZER = 'adam'
LR = 0.0002
WEIGHT_DECAY = 0
MOMENTUM = 0.9
BETA1 = 0.5  # for ADAM
DATA_AUGMENTATIONS = ['left-right', 'up-down']
SAVE_MODEL_EVERY_KTH = 50000
SAVE_PREDICTION_VISUALIZATAIONS = True
WETLAND_TO_GT_IDX = {'background': 0, 'hogmosse': 1, 'rikkarr': 2, 'oppna_mosse': 3, 'aapamyr': 4, 'kallor': 5}
MODEL_LOAD_PATH = None  # Set to a string to load a model with that path. None --> Load no  model
EVAL_ONLY = False  # Set to true to run model in eval-only mode, e.g. to eval a model specified in MODEL_LOAD_PATH

# Create directory in which to save current run
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
log_dir = os.path.join(BASE_PATH_LOG, timestamp)
stat_train_dir = os.path.join(log_dir, "train_stats")
os.makedirs(stat_train_dir, exist_ok=False)
copyfile("wetland_training.py", os.path.join(log_dir, "wetland_training.py"))

# Set seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set max number of threads to use to 10 (dgx1 limit)
torch.set_num_threads(10)

# Read data
data_path = os.path.join(BASE_PATH_DATA, 'aligned_maps')
data_names = np.sort(os.listdir(data_path))

# Split into train-val
nbr_examples = len(data_names)
nbr_examples_train = nbr_examples * 4 // 5  # 80% train
nbr_examples_val = nbr_examples - nbr_examples_train  # 20% train
data_names_train = []
data_names_val = []
nbr_rows = 146258 // 50
nbr_cols = 64034 // 50
check_matrix = np.zeros((nbr_rows, nbr_cols), dtype=int)
for outer_ctr in range(5):
    for i, data_name in enumerate(data_names[outer_ctr:]):
        H_start = int(data_name.split('_')[-2])
        H_start_idx = H_start // 50
        W_start = int(data_name.split('_')[-1].replace('.npy', ''))
        W_start_idx = W_start // 50
        # VAL
        if check_matrix[H_start_idx, W_start_idx] == -1 or check_matrix[H_start_idx, W_start_idx] == 0:
            # Below tries to ensure that we get 20% val, 80% train
            skip_val = False
            if len(data_names_train) > 0:
                if len(data_names_val) / len(data_names_train) > 0.202:
                    skip_val = True
            if len(data_names_train) == 0 and len(data_names_val) > 0:
                skip_val = True
            if not skip_val:
                check_matrix[H_start_idx, W_start_idx] = -2
                if H_start % 100 == 0 and H_start_idx < nbr_rows and check_matrix[H_start_idx + 1, W_start_idx] == 0:
                    check_matrix[H_start_idx + 1, W_start_idx] = -1
                elif H_start_idx > 0 and check_matrix[H_start_idx - 1, W_start_idx] == 0:
                    check_matrix[H_start_idx - 1, W_start_idx] = -1
                if W_start % 100 == 0 and W_start_idx < nbr_cols and check_matrix[H_start_idx, W_start_idx + 1] == 0:
                    check_matrix[H_start_idx, W_start_idx + 1] = -1
                elif W_start_idx > 0 and check_matrix[H_start_idx, W_start_idx - 1] == 0:
                    check_matrix[H_start_idx, W_start_idx - 1] = -1
                data_names_val.append(data_name)
        # TRAIN
        if check_matrix[H_start_idx, W_start_idx] == 1 or check_matrix[H_start_idx, W_start_idx] == 0:
            # Below tries to ensure that we get 20% val, 80% train
            if len(data_names_train) > 0:
                if len(data_names_val) / len(data_names_train) < 0.198:
                    continue
            check_matrix[H_start_idx, W_start_idx] = 2
            if H_start_idx % 100 == 0 and H_start_idx < nbr_rows and check_matrix[H_start_idx + 1, W_start_idx] == 0:
                check_matrix[H_start_idx + 1, W_start_idx] = 1
            elif H_start_idx > 0 and check_matrix[H_start_idx - 1, W_start_idx] == 0:
                check_matrix[H_start_idx - 1, W_start_idx] = 1
            if W_start_idx % 100 == 0 and W_start_idx < nbr_cols and check_matrix[H_start_idx, W_start_idx + 1] == 0:
                check_matrix[H_start_idx, W_start_idx + 1] = 1
            elif W_start_idx > 0 and check_matrix[H_start_idx, W_start_idx - 1] == 0:
                check_matrix[H_start_idx, W_start_idx - 1] = 1
            data_names_train.append(data_name)
data_names_sub_train = []  # Track data which satisfies certain constraints
data_names_sub_val = []
frame_ctr_train = 0
frame_ctr_val = 0

# Load these separately, in case we want to ensure these under-represented inputs get
# more representation
data_names_aapamyr = list(np.load('data_names_aapamyr.npy'))
data_names_hogmosse = list(np.load('data_names_hogmosse.npy'))
data_names_kallor = list(np.load('data_names_kallor.npy'))

# Extract and set certain dimensionalities
dim_input = 10
if PRED_ONLY_FG_BG:
    NBR_CLASSES = 2
else:
    NBR_CLASSES = 6 - len(WETLANDS_TO_NOT_PREDICT)

# Setup model
model = FCN8s(n_class=NBR_CLASSES, dim_input=dim_input, weight_init='normal')
if MODEL_LOAD_PATH is not None:
    model.load_state_dict(torch.load(MODEL_LOAD_PATH))
device = torch.device("cuda:0" if USE_GPU and torch.cuda.is_available() else "cpu")
model.to(device)

# Setup loss
bg_weight = BG_WEIGHT * 1 / NBR_CLASSES
fg_weight = (1 - bg_weight) / (NBR_CLASSES - 1)
ce_weights = [bg_weight] + [fg_weight for _ in range(NBR_CLASSES - 1)]
criterion = nn.CrossEntropyLoss(weight=torch.Tensor(ce_weights).to(device), reduce=False)

# Setup optimizer.
if OPTIMIZER == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, 0.999))
elif OPTIMIZER == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)

# Setup StatCollector
sc = StatCollector(stat_train_dir, NUM_TRAIN_ITER, 10)
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
sc.register('CE_loss_val', {'type': 'avg', 'freq': 'step'})
sc.register('mIoU_val', {'type': 'avg', 'freq': 'step'})
sc.register('Recall_val', {'type': 'avg', 'freq': 'step'})
sc.register('Precision_val', {'type': 'avg', 'freq': 'step'})
if True:
    for i in range(NBR_CLASSES):
        sc.register('IoU_' + str(i) + '_val', {'type': 'avg', 'freq': 'step'})
        sc.register('Recall_' + str(i) + '_val', {'type': 'avg', 'freq': 'step'})
        sc.register('Precision_' + str(i) + '_val', {'type': 'avg', 'freq': 'step'})

# Setup re-mapping of GT if omitting some wetland types
wetland_idxs_to_not_predict = [WETLAND_TO_GT_IDX[key] for key in WETLANDS_TO_NOT_PREDICT]
wetland_idxs_remaining = [wetland_idx for wetland_idx in [0, 1, 2, 3, 4, 5] if wetland_idx not in wetland_idxs_to_not_predict]

def _forward_compute(sc, mode='train', will_visualize=False):
    global data_names_train
    global data_names_val
    global data_names_sub_train
    global data_names_sub_val
    global frame_ctr_train
    global frame_ctr_val

    # Format current batch (including ground truth)
    data_batch = np.zeros((BATCH_SIZE, dim_input, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    gt_batch = np.zeros((BATCH_SIZE, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    support_batch = np.zeros((BATCH_SIZE, CROP_SIZE, CROP_SIZE), dtype=bool)
    b = 0
    while b < BATCH_SIZE:

        # Sample data augmentation during training, and image-gt pair
        aug_lr_flip = False
        aug_ud_flip = False
        if mode == 'train':
            # Sample data augmentation
            for aug in DATA_AUGMENTATIONS:
                if aug == 'left-right':
                    # Flips image left-right with probability 50%
                    aug_lr_flip = random.choice([False, True])
                if aug == 'up-down':
                    # Flips image up-down with probability 50%
                    aug_ud_flip = random.choice([False, True])
            if frame_ctr_train >= len(data_names_train):
                if data_names_sub_train is not None:
                    data_names_train = data_names_sub_train
                    data_names_sub_train = None
                random.shuffle(data_names_train)
                frame_ctr_train = 0
            if REQUIRE_AT_LEAST_ONE_PER_BATCH and b == 0:
                # Always throw in an aapamyr example
                data_name = random.sample(data_names_aapamyr, 1)[0]
            elif REQUIRE_AT_LEAST_ONE_PER_BATCH and b == 1:
                # Always throw in a hogmosse example
                data_name = random.sample(data_names_hogmosse, 1)[0]
            elif REQUIRE_AT_LEAST_ONE_PER_BATCH and b == 2:
                # Always throw in a kallor example
                data_name = random.sample(data_names_kallor, 1)[0]
            else:
                data_name = data_names_train[frame_ctr_train]
            frame_ctr_train += 1
        else:
            if frame_ctr_val >= len(data_names_val):
                if data_names_sub_val is not None:
                    data_names_val = data_names_sub_val
                    data_names_sub_val = None
                random.shuffle(data_names_val)
                frame_ctr_val = 0
            data_name = data_names_val[frame_ctr_val]
            frame_ctr_val += 1
        data_map = np.load(os.path.join(data_path, data_name))
        input_map = data_map[:, :, :dim_input]
        gt_map = data_map[:, :, dim_input]
        assert np.any(gt_map > 0)
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
            #if True:
            for wetland_idx in wetland_idxs_remaining:
                gt_map[gt_map == wetland_idx] = wetland_idxs_remaining.index(wetland_idx)

        # Skip all-background examples
        if 1:
            if random.random() <= 0.80 and np.all(gt_map == 0):
                continue
            elif mode == 'train' and data_names_sub_train is not None:
                data_names_sub_train.append(data_name)
            elif mode == 'val' and data_names_sub_val is not None:
                data_names_sub_val.append(data_name)
        else:
            # This branch ensures that some background-only examples
            # are used. Can be useful, for example in the case when
            # training a högmosse-only model, because in that case there
            # is no data in the North of Sweden. But the model has to
            # learn that there should not be any wetland predictions of
            # type högmosse in the North.
            if mode == 'train':
                if data_names_sub_train is not None:
                    if random.random() <= 0.80 and np.all(gt_map == 0):
                        continue
                    data_names_sub_train.append(data_name)
                else:
                    if random.random() <= 0.90 and np.all(gt_map == 0):
                        continue
            if mode == 'val':
                if data_names_sub_val is not None:
                    if random.random() <= 0.80 and np.all(gt_map == 0):
                        continue
                    data_names_sub_val.append(data_name)
                else:
                    if random.random() <= 0.90 and np.all(gt_map == 0):
                        continue

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
        if aug_lr_flip:
            curr_data_batch = np.flip(curr_data_batch, axis=2)
            curr_naturetype_batch = np.flip(curr_naturetype_batch, axis=2)
            curr_gt_batch = np.flip(curr_gt_batch, axis=1)
        if aug_ud_flip:
            curr_data_batch = np.flip(curr_data_batch, axis=1)
            curr_naturetype_batch = np.flip(curr_naturetype_batch, axis=1)
            curr_gt_batch = np.flip(curr_gt_batch, axis=0)

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
    #data_batch[:, 4, :, :] = 0  # zero out NMD
    #data_batch[:, 2:4, :, :] = 0  # zero out soil moisture maps
    #data_batch[:, 1, :, :] = 0  # zero out VMI
    #data_batch[:, 0, :, :] = 0  # zero out base class (?)
    gt_batch = torch.tensor(gt_batch, dtype=torch.long).to(device)
    support_batch = torch.tensor(support_batch, dtype=torch.long).to(device)

    # Forward the batch through the model, then compute the loss
    map_pred = model(data_batch)
    err_pred = criterion(map_pred, gt_batch)
    err_pred *= support_batch
    err_pred = torch.sum(err_pred) / torch.count_nonzero(support_batch)

    # After computing loss, ensure map_pred is argmaxed into class predictions
    map_pred = torch.argmax(map_pred, dim=1)

    # Compute mIoU
    map_pred_flat = map_pred.cpu().contiguous().view(-1).numpy()  # make it 1D
    gt_flat = gt_batch.cpu().contiguous().view(-1).numpy()  # make it 1D
    # Only compute mIoU in support area
    support_flat = support_batch.cpu().contiguous().view(-1).numpy()  # make it 1D
    miou = jaccard_score(gt_flat, map_pred_flat, average='macro', sample_weight=support_flat.astype(float))

	# Compute separate stats per prediction type (background, wetland #1, ..., wetland #N)
    # For each type, compute iou, precision and recall
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
    if mode == 'train':
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
    else:
        sc.s('CE_loss_val').collect(err_pred.item())
        sc.s('mIoU_val').collect(miou)
        if True:
            for i in range(NBR_CLASSES):
                if not np.isnan(iou_prec_recs[0, i]):
                    sc.s('IoU_' + str(i) + '_val').collect(iou_prec_recs[0, i])
                if not np.isnan(iou_prec_recs[1, i]):
                    sc.s('Precision_' + str(i) + '_val').collect(iou_prec_recs[1, i])
                if not np.isnan(iou_prec_recs[2, i]):
                    sc.s('Recall_' + str(i) + '_val').collect(iou_prec_recs[2, i])
        mean_prec = np.nanmean(iou_prec_recs[1, :])
        if not np.isnan(mean_prec):
            sc.s('Precision_val').collect(mean_prec)
        mean_rec = np.nanmean(iou_prec_recs[2, :])
        if not np.isnan(mean_rec):
            sc.s('Recall_val').collect(mean_rec)

    out_pred = map_pred.cpu().detach().numpy()[-1, :, :]
    out_gt = gt_batch.cpu().detach().numpy()[-1, :, :]
    out_support = support_batch.cpu().detach().numpy()[-1, :, :]
    if will_visualize:

        # Compute mIoU
        map_pred_flat = out_pred.flatten()
        gt_flat = out_gt.flatten()
        # Only compute mIoU in support area
        support_flat = out_support.flatten()
        miou = jaccard_score(gt_flat, map_pred_flat, average='macro', sample_weight=support_flat.astype(float))

        # Compute separate stats per prediction type (background, wetland #1, ..., wetland #N)
        # For each type, compute iou, precision and recall
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

    # Return
    out_in = np.squeeze(data_batch.cpu().detach().numpy()[-1, :, :, :])
    out_naturetype = np.squeeze(out_naturetype)
    return err_pred, [out_in, out_pred, out_gt, out_support, out_naturetype, miou, iou_prec_recs]

# Main training loop
print("Starting training loop...")
if EVAL_ONLY:
    model.eval()
else:
    model.train()  # Indicate that the model is to be trained
for it in range(NUM_TRAIN_ITER):

    # Forward computation
    err_pred, _ = _forward_compute(sc, mode='train')

    # Calculate gradients in backward pass and update model weights
    if not EVAL_ONLY:
        optimizer.zero_grad()
        err_pred.backward()
        optimizer.step()

    # Occassionally save model weights
    if (SAVE_MODEL_EVERY_KTH is not None and SAVE_MODEL_EVERY_KTH > 0) and \
        ((it > 0 and (it + 1) % SAVE_MODEL_EVERY_KTH == 0) or it + 1 == NUM_TRAIN_ITER):
        torch.save(model.state_dict(), os.path.join(log_dir, 'model_it_%d' % (it + 1)))

    # Track training and validation statistics
    if it % 25 == 0:
        do_visualize = SAVE_PREDICTION_VISUALIZATAIONS and it % 100 == 0
        _, pred_and_input = _forward_compute(sc, mode='val', will_visualize=do_visualize)
        sc.print()
        sc.save()
        print("Iter: %d / %d" % (it, NUM_TRAIN_ITER))

        # Save prediction visualizations
        if do_visualize:

            fig = plt.figure(figsize=(14, 14))

            # TOP PART
            fig.add_subplot(5,5,1)
            plt.imshow(pred_and_input[0][0, :, :], vmin=0, vmax=1)
            plt.title('Baseclass')
            plt.axis('off')
            fig.add_subplot(5,5,2)
            plt.imshow(pred_and_input[0][1, :, :], vmin=0, vmax=1)
            plt.title('VMI')
            plt.axis('off')
            fig.add_subplot(5,5,3)
            plt.imshow(pred_and_input[0][2, :, :], vmin=0, vmax=1)
            plt.title('S. moist ind.')
            plt.axis('off')
            fig.add_subplot(5,5,4)
            plt.imshow(pred_and_input[0][3, :, :], vmin=0, vmax=1)
            plt.title('Soil moist')
            plt.axis('off')
            fig.add_subplot(5,5,5)
            plt.imshow(pred_and_input[0][4, :, :], vmin=0, vmax=1)
            plt.title('NMD')
            plt.axis('off')
            fig.add_subplot(5,5,6)
            plt.imshow(pred_and_input[0][5, :, :], vmin=0, vmax=1)
            plt.title('Bush height')
            plt.axis('off')
            fig.add_subplot(5,5,7)
            plt.imshow(pred_and_input[0][6, :, :], vmin=0, vmax=1)
            plt.title('Bush cov')
            plt.axis('off')
            fig.add_subplot(5,5,8)
            plt.imshow(pred_and_input[0][7, :, :], vmin=0, vmax=1)
            plt.title('Tree height')
            plt.axis('off')
            fig.add_subplot(5,5,9)
            plt.imshow(pred_and_input[0][8, :, :], vmin=0, vmax=1)
            plt.title('Tree cov')
            plt.axis('off')
            fig.add_subplot(5,5,10)
            plt.imshow(pred_and_input[0][9, :, :], vmin=0, vmax=1)
            plt.title('Height')
            plt.axis('off')

            # MID PART
            fig.add_subplot(5,5,11+1)
            plt.imshow(pred_and_input[1], vmin=0, vmax=5)
            res = dict((v,k) for k,v in WETLAND_TO_GT_IDX.items())
            unqs = np.unique(pred_and_input[1])
            contours = measure.find_contours(pred_and_input[3].astype(float), 0.5)
            for contours_entry in contours:
                plt.plot(contours_entry[:, 1], contours_entry[:, 0], color='r')
            plt.title('Pred map')
            plt.axis('off')
            fig.add_subplot(5,5,12+1)
            plt.imshow(pred_and_input[2], vmin=0, vmax=5)
            unqs_gt = np.unique(pred_and_input[2])
            for contours_entry in contours:
                plt.plot(contours_entry[:, 1], contours_entry[:, 0], color='r')
            plt.title('GT map')
            plt.axis('off')
            fig.add_subplot(5,5,13+1)
            plt.imshow(pred_and_input[4], vmin=0, vmax=1)
            plt.title('Support map')
            plt.axis('off')
            # Below we also show an "image" with the current prediction's
            # stats (mIoU, mean precision, mean recall)
            if False:
                fig.add_subplot(5,5,14)
                plt.imshow(0 * pred_and_input[4])
                miou = pred_and_input[5]
                iou_prec_recs = pred_and_input[6]
                prec = np.nanmean(iou_prec_recs[1, :])
                rec = np.nanmean(iou_prec_recs[2, :])
                prec_nbg = np.nanmean(iou_prec_recs[1, 1:])
                rec_nbg = np.nanmean(iou_prec_recs[2, 1:])
                plt.text(5,30,"(mIoU, prec, rec):\n(%.3f, %.3f, %.3f)" % (miou, prec, rec),  color='r')
                plt.text(5,60,"(prec-nbg, rec-nbg):\n(%.3f, %.3f)" % (prec_nbg, rec_nbg),  color='r')
                plt.text(5,90,"# classes support-GT: %d" % len(np.unique(pred_and_input[2][pred_and_input[3] > 0])),  color='r')
                plt.title('Pred stats')
                plt.axis('off')
            else:
                fig.add_subplot(5,5,14+1)
                plt.imshow(0 * pred_and_input[4])
                if len(unqs) > 1:
                    plt.text(5,10, res[unqs[1]], color='r')
                if len(unqs) > 2:
                    plt.text(5,25, res[unqs[2]], color='r')
                if len(unqs) > 3:
                    plt.text(5,40, res[unqs[3]], color='r')
                if len(unqs) > 4:
                    plt.text(5,55, res[unqs[4]], color='r')
                if len(unqs) > 5:
                    plt.text(5,70, res[unqs[5]], color='r')
                if len(unqs_gt) > 1:
                    plt.text(5,15, res[unqs_gt[1]], color='g')
                if len(unqs_gt) > 2:
                    plt.text(5,30, res[unqs_gt[2]], color='g')
                if len(unqs_gt) > 3:
                    plt.text(5,45, res[unqs_gt[3]], color='g')
                if len(unqs_gt) > 4:
                    plt.text(5,60, res[unqs_gt[4]], color='g')
                if len(unqs_gt) > 5:
                    plt.text(5,75, res[unqs_gt[5]], color='g')
                plt.title('Pred stats')
                plt.axis('off')

            # BOTTOM PART
            fig.add_subplot(5,5,16)
            plt.imshow(pred_and_input[0][0, :, :])
            plt.title('Base veg. (n.)')
            plt.axis('off')
            fig.add_subplot(5,5,17)
            plt.imshow(pred_and_input[0][1, :, :])
            plt.title('Wetland inv. (n.)')
            plt.axis('off')
            fig.add_subplot(5,5,18)
            plt.imshow(pred_and_input[0][2, :, :])
            plt.title('Soil moist ind. (n.)')
            plt.axis('off')
            fig.add_subplot(5,5,19)
            plt.imshow(pred_and_input[0][3, :, :])
            plt.title('Soil moist (n.)')
            plt.axis('off')
            fig.add_subplot(5,5,20)
            plt.imshow(pred_and_input[0][4, :, :])
            plt.title('Land cover (n.)')
            plt.axis('off')
            fig.add_subplot(5,5,21)
            plt.imshow(pred_and_input[0][5, :, :])
            plt.title('Bush height (n.)')
            plt.axis('off')
            fig.add_subplot(5,5,22)
            plt.imshow(pred_and_input[0][6, :, :])
            plt.title('Bush cov (n.)')
            plt.axis('off')
            fig.add_subplot(5,5,23)
            plt.imshow(pred_and_input[0][7, :, :])
            plt.axis('off')
            plt.title('Tree height (n.)')
            fig.add_subplot(5,5,24)
            plt.imshow(pred_and_input[0][8, :, :])
            plt.title('Tree cov (n.)')
            plt.axis('off')
            fig.add_subplot(5,5,25)
            plt.imshow(pred_and_input[0][9, :, :])
            plt.title('Height (n.)')
            plt.axis('off')

            plt.savefig(os.path.join(stat_train_dir, 'pred_map_%d.png' % it))
            plt.cla()
            plt.clf()
            plt.close('all')

print("Training completed!")
