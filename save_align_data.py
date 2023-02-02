import os
import sys
import numpy as np


"""
This is a quite messy script that operates on the outputs obtained by using the
script preprocess_wetlands.py, in order to ensure the value ranges are
reasonable, and that all of the input-output maps are aligned.

Note: See segment_all_of_sweden.py if you want to streamline the below code a bit;
it contains most of the steps but does this in fewer lines of code by looping instead.
"""

BASE_PATH_DATA = '../wetlands-data'
RESAVE_ANNOTS = True
RESAVE_DATA = True
SAVE_ALIGNED = True

# Read GT data
print("Reading GT data")
annot_maps = []
annot_ids = []
annot_names = []
annot_path = os.path.join(BASE_PATH_DATA, 'annot_maps_100x100')
tot_ctr = 0
for file in np.sort(os.listdir(annot_path)):
    annot_names.append(os.path.join(annot_path, file))
    id = file.split('_')
    id = id[-2] + '_' + id[-1]
    annot_ids.append(id)
    if RESAVE_ANNOTS:
        annot_map = np.load(os.path.join(annot_path, file))
        if np.max(annot_map) <= 5:
            continue
        # Setup ground truth
        gt_map = np.zeros_like(annot_map)
        # 'hogmosse' -- category 1
        gt_map[annot_map == 7110] = 1
        gt_map[annot_map == 7111] = 1
        # 'rikkarr' -- category 2
        gt_map[annot_map == 7230] = 2
        gt_map[annot_map == 7231] = 2
        gt_map[annot_map == 7232] = 2
        gt_map[annot_map == 7233] = 2
        # 'oppna_mosse' -- category 3
        gt_map[annot_map == 7140] = 3
        gt_map[annot_map == 7141] = 3
        gt_map[annot_map == 7142] = 3
        gt_map[annot_map == 7143] = 3
        # 'aapamyr' -- category 4
        gt_map[annot_map == 7310] = 4
        # 'kallor' -- category 5
        gt_map[annot_map == 7160] = 5
        np.save(os.path.join(annot_path, file)[:-4], gt_map)
        tot_ctr += 1
        if tot_ctr % 100 == 0:
            print("GT part", tot_ctr, len(os.listdir(annot_path)))

# Read nature type data
print("Reading nature type data")
naturetype_maps = []
naturetype_ids = []
naturetype_names = []
naturetype_path = os.path.join(BASE_PATH_DATA, 'naturetype_maps_100x100')
tot_ctr = 0
for file in np.sort(os.listdir(naturetype_path)):
    id = file.split('_')
    id = id[-2] + '_' + id[-1]
    naturetype_names.append(os.path.join(naturetype_path, file))
    naturetype_ids.append(id)
    tot_ctr += 1
    if tot_ctr % 100 == 0:
        print("Naturetype part", tot_ctr, len(annot_ids))

# Setup input map values to correct range
list_naturetype_paths = np.sort(os.listdir(naturetype_path))

# Check sanity of it all
assert len(naturetype_ids) == len(annot_ids)
for i in range(len(naturetype_ids)):
    assert naturetype_ids[i] == annot_ids[i]

# Read baseclass data
print("Reading baseclass data")
baseclass_maps = []
baseclass_ids = []
baseclass_names = []
baseclass_path = os.path.join(BASE_PATH_DATA, 'baseclass_input_maps_100x100')
baseclass_path_save = os.path.join(BASE_PATH_DATA, 'baseclass_input_maps_100x100_resaved')
if not os.path.isdir(baseclass_path_save):
    os.mkdir(baseclass_path_save)
if not RESAVE_DATA:
    baseclass_path = baseclass_path_save
uniques = []
tot_ctr = 0
for file in np.sort(os.listdir(baseclass_path)):
    id = file.split('_')
    id = id[-2] + '_' + id[-1]
    baseclass_names.append(os.path.join(baseclass_path, file))
    baseclass_ids.append(id)
    tot_ctr += 1
    if tot_ctr % 100 == 0:
        print("Baseclass part", tot_ctr, len(annot_ids))
    if RESAVE_DATA:
        baseclass_map = np.load(os.path.join(baseclass_path, file)).astype(float)
        baseclass_maps.append(baseclass_map)
        uniques.append(np.unique(baseclass_map))
if RESAVE_DATA:
    uniques = np.unique(np.concatenate(uniques))
    np.save(os.path.join(BASE_PATH_DATA, 'baseclass_uniques'), uniques)
else:
    uniques = np.load(os.path.join(BASE_PATH_DATA, 'baseclass_uniques.npy'))

# Setup input map values to correct range
list_baseclass_paths = np.sort(os.listdir(baseclass_path))
max_baseclass = len(uniques)
if RESAVE_DATA:
    print("Normalizing input")
    for i in range(len(baseclass_maps)):
        for j, unq in enumerate(uniques):
            if np.ndim(baseclass_maps[i]) == 2:
                baseclass_maps[i][baseclass_maps[i] == unq] = j
        if i % 100 == 0:
            print(i)
    # Normalize input map to [0, 1] range
    for i in range(len(baseclass_maps)):
        if np.ndim(baseclass_maps[i]) == 2:
            baseclass_maps[i] = baseclass_maps[i][:, :, np.newaxis] / max_baseclass
            assert np.min(baseclass_maps[i]) >= 0 and np.max(baseclass_maps[i]) <= 1
            save_path = os.path.join(baseclass_path_save, list_baseclass_paths[i])[:-4]
            np.save(save_path, baseclass_maps[i])

# Check sanity of it all
assert len(baseclass_ids) == len(annot_ids)
for i in range(len(baseclass_ids)):
    assert baseclass_ids[i] == annot_ids[i]

# Read VMI data
print("Reading VMI data")
vmi_maps = []
vmi_ids = []
vmi_names = []
vmi_path = os.path.join(BASE_PATH_DATA, 'vmi_input_maps_100x100')
vmi_path_save = os.path.join(BASE_PATH_DATA, 'vmi_input_maps_100x100_resaved')
if not os.path.isdir(vmi_path_save):
    os.mkdir(vmi_path_save)
if not RESAVE_DATA:
    vmi_path = vmi_path_save
uniques = []
tot_ctr = 0
for file in np.sort(os.listdir(vmi_path)):
    id = file.split('_')
    id = id[-2] + '_' + id[-1]
    vmi_names.append(os.path.join(vmi_path, file))
    vmi_ids.append(id)
    tot_ctr += 1
    if tot_ctr % 100 == 0:
        print("VMI part", tot_ctr, len(annot_ids))
    if RESAVE_DATA:
        vmi_map = np.load(os.path.join(vmi_path, file)).astype(float)
        vmi_maps.append(vmi_map)
        uniques.append(np.unique(vmi_map))
if RESAVE_DATA:
    uniques = np.unique(np.concatenate(uniques))
    np.save(os.path.join(BASE_PATH_DATA, 'vmi_uniques'), uniques)
else:
    uniques = np.load(os.path.join(BASE_PATH_DATA, 'vmi_uniques.npy'))

# Setup vmi map values to correct range
list_vmi_paths = np.sort(os.listdir(vmi_path))
max_input = len(uniques)
if RESAVE_DATA:
    print("Normalizing input")
    for i in range(len(vmi_maps)):
        for j, unq in enumerate(uniques):
            if np.ndim(vmi_maps[i]) == 2:
                vmi_maps[i][vmi_maps[i] == unq] = j
        if i % 100 == 0:
            print(i)
    # Normalize vmi map to [0, 1] range
    # TODO: HOW REASONABLE IS THIS? HAVE BACKGROUND CHANNEL AS SEPARATE? SAVE AS "RGB"? ETC?
    for i in range(len(vmi_maps)):
        if np.ndim(vmi_maps[i]) == 2:
            vmi_maps[i] = vmi_maps[i][:, :, np.newaxis] / max_input
            assert np.min(vmi_maps[i]) >= 0 and np.max(vmi_maps[i]) <= 1
            save_path = os.path.join(vmi_path_save, list_vmi_paths[i])[:-4]
            np.save(save_path, vmi_maps[i])

# Check sanity of it all
assert len(vmi_ids) == len(annot_ids)
for i in range(len(vmi_ids)):
    assert vmi_ids[i] == annot_ids[i]

# Read height map data
print("Reading height map data")
height_maps = []
height_ids = []
height_names = []
height_path = os.path.join(BASE_PATH_DATA, 'height_input_maps_100x100')
tot_ctr = 0
max_input = 0
for file in np.sort(os.listdir(height_path)):
    id = file.split('_')
    id = id[-2] + '_' + id[-1]
    height_names.append(os.path.join(height_path, file))
    height_ids.append(id)
    tot_ctr += 1
    if tot_ctr % 100 == 0:
        print("height part", tot_ctr, len(annot_ids))
    if RESAVE_DATA:
        height_map = np.load(os.path.join(height_path, file))
        if np.ndim(height_map) > 2:
            continue
        height_maps.append(height_map)
        max_input = max(max_input, np.max(height_map))
if RESAVE_DATA:
    max_input_height = max_input
    np.save(os.path.join(BASE_PATH_DATA, 'height_max'), max_input)
else:
    max_input_height = np.load(os.path.join(BASE_PATH_DATA, 'height_max.npy'))
# Check sanity of it all
assert len(height_ids) == len(annot_ids)
for i in range(len(height_ids)):
    assert height_ids[i] == annot_ids[i]

# Read soil moist index data
print("Reading soil moisture data")
soilmoistindex_maps = []
soilmoistindex_ids = []
soilmoistindex_names = []
soilmoistindex_path = os.path.join(BASE_PATH_DATA, 'soilmoistindex_input_maps_100x100')
soilmoistindex_path_save = os.path.join(BASE_PATH_DATA, 'soilmoistindex_input_maps_100x100_resaved')
if not os.path.isdir(soilmoistindex_path_save):
    os.mkdir(soilmoistindex_path_save)
if not RESAVE_DATA:
    soilmoistindex_path = soilmoistindex_path_save
uniques = []
tot_ctr = 0
for file in np.sort(os.listdir(soilmoistindex_path)):
    id = file.split('_')
    id = id[-2] + '_' + id[-1]
    soilmoistindex_names.append(os.path.join(soilmoistindex_path, file))
    soilmoistindex_ids.append(id)
    tot_ctr += 1
    if tot_ctr % 100 == 0:
        print("soilmoistindex part", tot_ctr, len(annot_ids))
    if RESAVE_DATA:
        soilmoistindex_map = np.load(os.path.join(soilmoistindex_path, file)).astype(float)
        soilmoistindex_maps.append(soilmoistindex_map)
        uniques.append(np.unique(soilmoistindex_map))
if RESAVE_DATA:
    uniques = np.unique(np.concatenate(uniques))
    np.save(os.path.join(BASE_PATH_DATA, 'soilmoistindex_uniques'), uniques)
else:
    uniques = np.load(os.path.join(BASE_PATH_DATA, 'soilmoistindex_uniques.npy'))

# Setup soilmoistindex map values to correct range
list_soilmoistindex_paths = np.sort(os.listdir(soilmoistindex_path))
max_input = len(uniques)
if RESAVE_DATA:
    print("Normalizing input")
    for i in range(len(soilmoistindex_maps)):
        for j, unq in enumerate(uniques):
            if np.ndim(soilmoistindex_maps[i]) == 2:
                soilmoistindex_maps[i][soilmoistindex_maps[i] == unq] = j
        if i % 100 == 0:
            print(i)
    # Normalize soilmoistindex map to [0, 1] range
    # TODO: HOW REASONABLE IS THIS? HAVE BACKGROUND CHANNEL AS SEPARATE? SAVE AS "RGB"? ETC?
    for i in range(len(soilmoistindex_maps)):
        if np.ndim(soilmoistindex_maps[i]) == 2:
            soilmoistindex_maps[i] = soilmoistindex_maps[i][:, :, np.newaxis] / max_input
            assert np.min(soilmoistindex_maps[i]) >= 0 and np.max(soilmoistindex_maps[i]) <= 1
            save_path = os.path.join(soilmoistindex_path_save, list_soilmoistindex_paths[i])[:-4]
            np.save(save_path, soilmoistindex_maps[i])

assert len(soilmoistindex_ids) == len(annot_ids)
for i in range(len(soilmoistindex_ids)):
    assert soilmoistindex_ids[i] == annot_ids[i]

# Read soil moisture (SGU) data
print("Reading soil moisture data")
soilmoist_maps = []
soilmoist_ids = []
soilmoist_names = []
soilmoist_path = os.path.join(BASE_PATH_DATA, 'soilmoist_input_maps_100x100')
soilmoist_path_save = os.path.join(BASE_PATH_DATA, 'soilmoist_input_maps_100x100_resaved')
if not os.path.isdir(soilmoist_path_save):
    os.mkdir(soilmoist_path_save)
if not RESAVE_DATA:
    soilmoist_path = soilmoist_path_save
uniques = []
tot_ctr = 0
for file in np.sort(os.listdir(soilmoist_path)):
    id = file.split('_')
    id = id[-2] + '_' + id[-1]
    soilmoist_names.append(os.path.join(soilmoist_path, file))
    soilmoist_ids.append(id)
    tot_ctr += 1
    if tot_ctr % 100 == 0:
        print("soilmoist part", tot_ctr, len(annot_ids))
    if RESAVE_DATA:
        soilmoist_map = np.load(os.path.join(soilmoist_path, file)).astype(float)
        soilmoist_maps.append(soilmoist_map)
        uniques.append(np.unique(soilmoist_map))
if RESAVE_DATA:
    uniques = np.unique(np.concatenate(uniques))
    np.save(os.path.join(BASE_PATH_DATA, 'soilmoist_uniques'), uniques)
else:
    uniques = np.load(os.path.join(BASE_PATH_DATA, 'soilmoist_uniques.npy'))

# Setup soilmoist map values to correct range
list_soilmoist_paths = np.sort(os.listdir(soilmoist_path))
max_input = len(uniques)
if RESAVE_DATA:
    print("Normalizing input")
    for i in range(len(soilmoist_maps)):
        for j, unq in enumerate(uniques):
            if np.ndim(soilmoist_maps[i]) == 2:
                soilmoist_maps[i][soilmoist_maps[i] == unq] = j
        if i % 100 == 0:
            print(i)
    # Normalize soilmoist map to [0, 1] range
    # TODO: HOW REASONABLE IS THIS? HAVE BACKGROUND CHANNEL AS SEPARATE? SAVE AS "RGB"? ETC?
    for i in range(len(soilmoist_maps)):
        if np.ndim(soilmoist_maps[i]) == 2:
            soilmoist_maps[i] = soilmoist_maps[i][:, :, np.newaxis] / max_input
            assert np.min(soilmoist_maps[i]) >= 0 and np.max(soilmoist_maps[i]) <= 1
            save_path = os.path.join(soilmoist_path_save, list_soilmoist_paths[i])[:-4]
            np.save(save_path, soilmoist_maps[i])

assert len(soilmoist_ids) == len(annot_ids)
for i in range(len(soilmoist_ids)):
    assert soilmoist_ids[i] == annot_ids[i]

# Read NMD data
print("Reading NMD data")
nmd_maps = []
nmd_ids = []
nmd_names = []
nmd_path = os.path.join(BASE_PATH_DATA, 'nmd_input_maps_100x100')
nmd_path_save = os.path.join(BASE_PATH_DATA, 'nmd_input_maps_100x100_resaved')
if not os.path.isdir(nmd_path_save):
    os.mkdir(nmd_path_save)
if not RESAVE_DATA:
    nmd_path = nmd_path_save
uniques = []
tot_ctr = 0
for file in np.sort(os.listdir(nmd_path)):
    id = file.split('_')
    id = id[-2] + '_' + id[-1]
    nmd_names.append(os.path.join(nmd_path, file))
    nmd_ids.append(id)
    tot_ctr += 1
    if tot_ctr % 100 == 0:
        print("nmd part", tot_ctr, len(annot_ids))
    if RESAVE_DATA:
        nmd_map = np.load(os.path.join(nmd_path, file)).astype(float)
        nmd_maps.append(nmd_map)
        uniques.append(np.unique(nmd_map))
if RESAVE_DATA:
    uniques = np.unique(np.concatenate(uniques))
    np.save(os.path.join(BASE_PATH_DATA, 'nmd_uniques'), uniques)
else:
    uniques = np.load(os.path.join(BASE_PATH_DATA, 'nmd_uniques.npy'))

# Setup nmd map values to correct range
list_nmd_paths = np.sort(os.listdir(nmd_path))
max_input = len(uniques)
if RESAVE_DATA:
    print("Normalizing input")
    for i in range(len(nmd_maps)):
        for j, unq in enumerate(uniques):
            if np.ndim(nmd_maps[i]) == 2:
                nmd_maps[i][nmd_maps[i] == unq] = j
        if i % 100 == 0:
            print(i)
    # Normalize nmd map to [0, 1] range
    # TODO: HOW REASONABLE IS THIS? HAVE BACKGROUND CHANNEL AS SEPARATE? SAVE AS "RGB"? ETC?
    for i in range(len(nmd_maps)):
        if np.ndim(nmd_maps[i]) == 2:
            nmd_maps[i] = nmd_maps[i][:, :, np.newaxis] / max_input
            assert np.min(nmd_maps[i]) >= 0 and np.max(nmd_maps[i]) <= 1
            save_path = os.path.join(nmd_path_save, list_nmd_paths[i])[:-4]
            np.save(save_path, nmd_maps[i])

assert len(nmd_ids) == len(annot_ids)
for i in range(len(nmd_ids)):
    assert nmd_ids[i] == annot_ids[i]

# Bush height data
print("Reading bush data")
bush_maps = []
bush_ids = []
bush_names = []
bush_path = os.path.join(BASE_PATH_DATA, 'bush_input_maps_100x100')
bush_path_save = os.path.join(BASE_PATH_DATA, 'bush_input_maps_100x100_resaved')
if not os.path.isdir(bush_path_save):
    os.mkdir(bush_path_save)
if not RESAVE_DATA:
    bush_path = bush_path_save
uniques = []
tot_ctr = 0
for file in np.sort(os.listdir(bush_path)):
    id = file.split('_')
    id = id[-2] + '_' + id[-1]
    bush_names.append(os.path.join(bush_path, file))
    bush_ids.append(id)
    tot_ctr += 1
    if tot_ctr % 100 == 0:
        print("bush part", tot_ctr, len(annot_ids))
    if RESAVE_DATA:
        bush_map = np.load(os.path.join(bush_path, file)).astype(float)
        bush_maps.append(bush_map)
        uniques.append(np.unique(bush_map))
if RESAVE_DATA:
    uniques = np.unique(np.concatenate(uniques))
    np.save(os.path.join(BASE_PATH_DATA, 'bush_uniques'), uniques)
else:
    uniques = np.load(os.path.join(BASE_PATH_DATA, 'bush_uniques.npy'))

# Setup bush map values to correct range
list_bush_paths = np.sort(os.listdir(bush_path))
max_input = len(uniques)
if RESAVE_DATA:
    print("Normalizing input")
    for i in range(len(bush_maps)):
        for j, unq in enumerate(uniques):
            if np.ndim(bush_maps[i]) == 2:
                bush_maps[i][bush_maps[i] == unq] = j
        if i % 100 == 0:
            print(i)
    # Normalize bush map to [0, 1] range
    # TODO: HOW REASONABLE IS THIS? HAVE BACKGROUND CHANNEL AS SEPARATE? SAVE AS "RGB"? ETC?
    for i in range(len(bush_maps)):
        if np.ndim(bush_maps[i]) == 2:
            bush_maps[i] = bush_maps[i][:, :, np.newaxis] / max_input
            assert np.min(bush_maps[i]) >= 0 and np.max(bush_maps[i]) <= 1
            save_path = os.path.join(bush_path_save, list_bush_paths[i])[:-4]
            np.save(save_path, bush_maps[i])

assert len(bush_ids) == len(annot_ids)
for i in range(len(bush_ids)):
    assert bush_ids[i] == annot_ids[i]

# Bush cover data
print("Reading bush cover data")
bush_cover_maps = []
bush_cover_ids = []
bush_cover_names = []
bush_cover_path = os.path.join(BASE_PATH_DATA, 'bush_cover_input_maps_100x100')
bush_cover_path_save = os.path.join(BASE_PATH_DATA, 'bush_cover_input_maps_100x100_resaved')
if not os.path.isdir(bush_cover_path_save):
    os.mkdir(bush_cover_path_save)
if not RESAVE_DATA:
    bush_cover_path = bush_cover_path_save
uniques = []
tot_ctr = 0
for file in np.sort(os.listdir(bush_cover_path)):
    id = file.split('_')
    id = id[-2] + '_' + id[-1]
    bush_cover_names.append(os.path.join(bush_cover_path, file))
    bush_cover_ids.append(id)
    tot_ctr += 1
    if tot_ctr % 100 == 0:
        print("bush cover part", tot_ctr, len(annot_ids))
    if RESAVE_DATA:
        bush_cover_map = np.load(os.path.join(bush_cover_path, file)).astype(float)
        bush_cover_maps.append(bush_cover_map)
        uniques.append(np.unique(bush_cover_map))
if RESAVE_DATA:
    uniques = np.unique(np.concatenate(uniques))
    np.save(os.path.join(BASE_PATH_DATA, 'bush_cover_uniques'), uniques)
else:
    uniques = np.load(os.path.join(BASE_PATH_DATA, 'bush_cover_uniques.npy'))

# Setup height map values to correct range
list_bush_cover_paths = np.sort(os.listdir(bush_cover_path))
max_input = len(uniques)
if RESAVE_DATA:
    print("Normalizing input")
    for i in range(len(bush_cover_maps)):
        for j, unq in enumerate(uniques):
            if np.ndim(bush_cover_maps[i]) == 2:
                bush_cover_maps[i][bush_cover_maps[i] == unq] = j
        if i % 100 == 0:
            print(i)
    # Normalize height map to [0, 1] range
    # TODO: HOW REASONABLE IS THIS? HAVE BACKGROUND CHANNEL AS SEPARATE? SAVE AS "RGB"? ETC?
    for i in range(len(bush_cover_maps)):
        if np.ndim(bush_cover_maps[i]) == 2:
            bush_cover_maps[i] = bush_cover_maps[i][:, :, np.newaxis] / max_input
            assert np.min(bush_cover_maps[i]) >= 0 and np.max(bush_cover_maps[i]) <= 1
            save_path = os.path.join(bush_cover_path_save, list_bush_cover_paths[i])[:-4]
            np.save(save_path, bush_cover_maps[i])

assert len(bush_cover_ids) == len(annot_ids)
for i in range(len(bush_cover_ids)):
    assert bush_cover_ids[i] == annot_ids[i]

# Tree height data
print("Reading tree data")
tree_maps = []
tree_ids = []
tree_names = []
tree_path = os.path.join(BASE_PATH_DATA, 'tree_input_maps_100x100')
tree_path_save = os.path.join(BASE_PATH_DATA, 'tree_input_maps_100x100_resaved')
if not os.path.isdir(tree_path_save):
    os.mkdir(tree_path_save)
if not RESAVE_DATA:
    tree_path = tree_path_save
uniques = []
tot_ctr = 0
for file in np.sort(os.listdir(tree_path)):
    id = file.split('_')
    id = id[-2] + '_' + id[-1]
    tree_names.append(os.path.join(tree_path, file))
    tree_ids.append(id)
    tot_ctr += 1
    if tot_ctr % 100 == 0:
        print("tree part", tot_ctr, len(annot_ids))
    if RESAVE_DATA:
        tree_map = np.load(os.path.join(tree_path, file)).astype(float)
        tree_maps.append(tree_map)
        uniques.append(np.unique(tree_map))
if RESAVE_DATA:
    uniques = np.unique(np.concatenate(uniques))
    np.save(os.path.join(BASE_PATH_DATA, 'tree_uniques'), uniques)
else:
    uniques = np.load(os.path.join(BASE_PATH_DATA, 'tree_uniques.npy'))

# Setup tree map values to correct range
list_tree_paths = np.sort(os.listdir(tree_path))
max_input = len(uniques)
if RESAVE_DATA:
    print("Normalizing input")
    for i in range(len(tree_maps)):
        for j, unq in enumerate(uniques):
            if np.ndim(tree_maps[i]) == 2:
                tree_maps[i][tree_maps[i] == unq] = j
        if i % 100 == 0:
            print(i)
    # Normalize tree map to [0, 1] range
    # TODO: HOW REASONABLE IS THIS? HAVE BACKGROUND CHANNEL AS SEPARATE? SAVE AS "RGB"? ETC?
    for i in range(len(tree_maps)):
        if np.ndim(tree_maps[i]) == 2:
            tree_maps[i] = tree_maps[i][:, :, np.newaxis] / max_input
            assert np.min(tree_maps[i]) >= 0 and np.max(tree_maps[i]) <= 1
            save_path = os.path.join(tree_path_save, list_tree_paths[i])[:-4]
            np.save(save_path, tree_maps[i])

assert len(tree_ids) == len(annot_ids)
for i in range(len(tree_ids)):
    assert tree_ids[i] == annot_ids[i]

# Tree cover data
print("Reading tree cover data")
tree_cover_maps = []
tree_cover_ids = []
tree_cover_names = []
tree_cover_path = os.path.join(BASE_PATH_DATA, 'tree_cover_input_maps_100x100')
tree_cover_path_save = os.path.join(BASE_PATH_DATA, 'tree_cover_input_maps_100x100_resaved')
if not os.path.isdir(tree_cover_path_save):
    os.mkdir(tree_cover_path_save)
if not RESAVE_DATA:
    tree_cover_path = tree_cover_path_save
uniques = []
tot_ctr = 0
for file in np.sort(os.listdir(tree_cover_path)):
    id = file.split('_')
    id = id[-2] + '_' + id[-1]
    tree_cover_names.append(os.path.join(tree_cover_path, file))
    tree_cover_ids.append(id)
    tot_ctr += 1
    if tot_ctr % 100 == 0:
        print("tree cover part", tot_ctr, len(annot_ids))
    if RESAVE_DATA:
        tree_cover_map = np.load(os.path.join(tree_cover_path, file)).astype(float)
        tree_cover_maps.append(tree_cover_map)
        uniques.append(np.unique(tree_cover_map))
if RESAVE_DATA:
    uniques = np.unique(np.concatenate(uniques))
    np.save(os.path.join(BASE_PATH_DATA, 'tree_cover_uniques'), uniques)
else:
    uniques = np.load(os.path.join(BASE_PATH_DATA, 'tree_cover_uniques.npy'))

# Setup soilcover map values to correct range
list_tree_cover_paths = np.sort(os.listdir(tree_cover_path))
max_input = len(uniques)
if RESAVE_DATA:
    print("Normalizing input")
    for i in range(len(tree_cover_maps)):
        for j, unq in enumerate(uniques):
            if np.ndim(tree_cover_maps[i]) == 2:
                tree_cover_maps[i][tree_cover_maps[i] == unq] = j
        if i % 100 == 0:
            print(i)
    # Normalize soilcover map to [0, 1] range
    # TODO: HOW REASONABLE IS THIS? HAVE BACKGROUND CHANNEL AS SEPARATE? SAVE AS "RGB"? ETC?
    for i in range(len(tree_cover_maps)):
        if np.ndim(tree_cover_maps[i]) == 2:
            tree_cover_maps[i] = tree_cover_maps[i][:, :, np.newaxis] / max_input
            assert np.min(tree_cover_maps[i]) >= 0 and np.max(tree_cover_maps[i]) <= 1
            save_path = os.path.join(tree_cover_path_save, list_tree_cover_paths[i])[:-4]
            np.save(save_path, tree_cover_maps[i])

assert len(tree_cover_ids) == len(annot_ids)
for i in range(len(tree_cover_ids)):
    assert tree_cover_ids[i] == annot_ids[i]

# Save aligned maps
if SAVE_ALIGNED:
    for i, baseclass_name in enumerate(baseclass_names):
        annot_name = annot_names[i]
        combined_name = annot_name
        combined_name = combined_name.replace('annot_maps_100x100', 'aligned_maps')
        combined_name = combined_name.replace('annotation_map_gt', 'aligned_map')
        if os.path.isfile(combined_name):
            continue
        naturetype_name = naturetype_names[i]
        vmi_name = vmi_names[i]
        height_name = height_names[i]
        soilmoistindex_name = soilmoistindex_names[i]
        soilmoist_name = soilmoist_names[i]
        nmd_name = nmd_names[i]
        bush_name = bush_names[i]
        bush_cover_name = bush_cover_names[i]
        tree_name = tree_names[i]
        tree_cover_name = tree_cover_names[i]
        baseclass_map = np.load(baseclass_name)
        gt_map = np.load(annot_name)
        naturetype_map = np.load(naturetype_name)
        vmi_map = np.load(vmi_name)
        height_map = np.load(height_name)
        soilmoistindex_map = np.load(soilmoistindex_name)
        soilmoist_map = np.load(soilmoist_name)
        nmd_map = np.load(nmd_name)
        bush_map = np.load(bush_name)
        bush_cover_map = np.load(bush_cover_name)
        tree_map = np.load(tree_name)
        tree_cover_map = np.load(tree_cover_name)
        combined_map = np.concatenate([baseclass_map, vmi_map, soilmoistindex_map, soilmoist_map,
                                       nmd_map, bush_map, bush_cover_map, tree_map, tree_cover_map,
                                       height_map[:, :, np.newaxis] / max_input_height, gt_map[:, :, np.newaxis],
                                       naturetype_map[:, :, np.newaxis]], axis=2)
        np.save(combined_name, combined_map)
        if i % 100 == 0:
            print(i, len(baseclass_names))
    print("SAVED ALL")
    sys.exit(0)
