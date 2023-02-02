import os
import sys
import time
import tifffile
import numpy as np

"""
This script is for preprocessing the various input and output maps (see the report for details)
and saving the maps as CROP_SIZE x CROP_SIZE rectangles.
"""

BASE_PATH_DATA = '../wetlands-data'
DATA_PATH = os.path.join(BASE_PATH_DATA, 'Naturtypskartan_RIKS', 'NNK_YTA.shp')
WETLAND_CODES_TO_NAMES = {7110: 'hogmosse', 7111: 'hogmosse',
                          7230: 'rikkarr', 7231: 'rikkarr', 7232: 'rikkarr', 7233: 'rikkarr',
                          7140: 'oppna_mosse', 7141: 'oppna_mosse', 7142: 'oppna_mosse', 7143: 'oppna_mosse',
                          7310: 'aapamyr',
                          7160: 'kallor'}
CROP_SIZE = 100  # Saves the data as CROP_SIZE x CROP_SIZE rectangles
START_H = 0  # Change from (START_H, START_W)=(0, 0) to something else, to create a "shifted" grid of CROP_SIZE x CROP_SIZE rectangles
START_W = 0

# The below part precomputes maps related to the ground truth and the nature type map in general
if True:

    SAVE_GT_MAPS = False  # False --> save naturetype maps instead. NOTE: First run this code block with False, then again with True.

    if SAVE_GT_MAPS:
        tif_path = os.path.join(BASE_PATH_DATA, 'NNK_YTAvatmark.tif')
        key_path = os.path.join(BASE_PATH_DATA, 'NNKvatmarkID.txt')
    else:
        tif_path = os.path.join(BASE_PATH_DATA, 'NNK_YTA.tif')
        key_path = os.path.join(BASE_PATH_DATA, 'NNKrasterValuetoNNKkod.txt')
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
    im = np.array(tifffile.imread(tif_path))
    H, W = im.shape  # 146258, 64034

    # Set up range over which to iterate
    if not SAVE_GT_MAPS:
        # List all gt maps, to speed up (and save disk space for!) the naturetype save process
        list_gt_maps = np.sort(os.listdir(os.path.join(BASE_PATH_DATA, 'annot_maps_100x100')))
        ijs = [[int(vv.split('.')[0].split('_')[-2]), int(vv.split('.')[0].split('_')[-1])] for vv in list_gt_maps]
    else:
        ijs = []
        for i in range(START_H, H - CROP_SIZE, CROP_SIZE):
            for j in range(START_W, W - CROP_SIZE, CROP_SIZE):
                ijs.append([i, j])

    # Save sub images
    print("SAVING SUB IMAGES")
    H, W = im.shape
    tot_ctr = 0
    nbr_saved_im = 0
    for ij in ijs:
        i = ij[0]
        j = ij[1]
        tot_ctr += 1
        if tot_ctr % 100 == 0:
            print("counter, number of saved", tot_ctr, nbr_saved_im)
        sub_im = im[i : i + CROP_SIZE, j : j + CROP_SIZE].astype(int)
        assert np.all(sub_im >= 0) and np.all(sub_im <= 255)
        for key in range(256):
            if key in key_to_nature_code:
                value = key_to_nature_code[key]
                if (SAVE_GT_MAPS and value in WETLAND_CODES_TO_NAMES) or (not SAVE_GT_MAPS and value not in WETLAND_CODES_TO_NAMES):
                    sub_im[sub_im == key] = value
                else:
                    sub_im[sub_im == key] = 0
            else:
                # Background (not a specific nature type)
                sub_im[sub_im == key] = 0
        if SAVE_GT_MAPS and np.count_nonzero(sub_im) > 0:
            save_map_name = os.path.join(BASE_PATH_DATA, 'annot_maps_100x100/annotation_map_gt_%d_%d.npy' % (i, j))
            np.save(save_map_name, sub_im)
            nbr_saved_im += 1
        elif not SAVE_GT_MAPS:
            save_map_name = os.path.join(BASE_PATH_DATA, 'naturetype_maps_100x100/naturetype_map_%d_%d.npy' % (i, j))
            np.save(save_map_name, sub_im)
            nbr_saved_im += 1
    print("DONE OVERALL, SAVED NBR IM: ", nbr_saved_im)
    sys.exit(0)

# This part is for precomputing model inputs.
# NOTE: First run the other code block below, i.e. set its outer boolean to True.
# Only after having completed that, you may set the below boolean to True
# and run the below.
if False:
    tif_paths = [os.path.join(BASE_PATH_DATA, 'VatmarkerBasklass.tif'),
                 os.path.join(BASE_PATH_DATA, 'VMIobjekttyp.tif'),
                 os.path.join(BASE_PATH_DATA, 'NhSverige.tif'),
                 os.path.join(BASE_PATH_DATA, 'MarfuktighetsindexNMD.tif'),
                 os.path.join(BASE_PATH_DATA, 'SLUmarkfuktighet.tif'),
                 os.path.join(BASE_PATH_DATA, 'NMDbas.tif'),
                 os.path.join(BASE_PATH_DATA, 'Objekthojd05_5.tif'),
                 os.path.join(BASE_PATH_DATA, 'Objekttackning05_5.tif'),
                 os.path.join(BASE_PATH_DATA, 'Objekthojd5_45.tif'),
                 os.path.join(BASE_PATH_DATA, 'Objekttackning5_45.tif')]
    save_names = [os.path.join(BASE_PATH_DATA, 'baseclass_input_maps_100x100/baseclass_map_'),
                  os.path.join(BASE_PATH_DATA, 'vmi_input_maps_100x100/vmi_map_'),
                  os.path.join(BASE_PATH_DATA, 'height_input_maps_100x100/height_map_'),
                  os.path.join(BASE_PATH_DATA, 'soilmoistindex_input_maps_100x100/soil_moist_index_map_'),
                  os.path.join(BASE_PATH_DATA, 'soilmoist_input_maps_100x100/soil_moist_map_'),
                  os.path.join(BASE_PATH_DATA, 'nmd_input_maps_100x100/nmd_map_'),
                  os.path.join(BASE_PATH_DATA, 'bush_input_maps_100x100/bush_map_'),
                  os.path.join(BASE_PATH_DATA, 'bush_cover_input_maps_100x100/bush_cover_map_'),
                  os.path.join(BASE_PATH_DATA, 'tree_input_maps_100x100/tree_map_'),
                  os.path.join(BASE_PATH_DATA, 'tree_cover_input_maps_100x100/tree_cover_map_')]
    for outer_it, tif_path in enumerate(tif_paths):
        print("READING IMAGES")
        im = np.array(tifffile.imread(tif_path))
        H, W = im.shape

        # Preprocess range of values of the maps
        if 'NhSverige' in tif_path:
            im[im < 0] = 0
        if 'Objekthojd' in tif_path:
            im[im == 255] = 0
        elif 'Objekttackning' in tif_path:
            im[im == 255] = 0
        elif 'VMI' in tif_path:
            im[im == 128] = 0
        elif 'SLUmarkfuktighet' in tif_path:
            im[im != 255] += 1  # instead of 0, ..., N-1 --> 1, ..., N
            im[im == 255] = 0
        elif 'MarfuktighetsindexNMD' in tif_path:
            im[im < 0] = 0
            im[:H//2, :W//2] = np.round(im[:H//2, :W//2])
            im[H//2:, :W//2] = np.round(im[H//2:, :W//2])
            im[:H//2, W//2:] = np.round(im[:H//2, W//2:])
            im[H//2:, W//2:] = np.round(im[H//2:, W//2:])
            im = im.astype(int)

        # Save sub images
        print("SAVING SUB IMAGES")
        # Ratios are used to correctly scale relative to the naturetype data
        H_ratio = H / 146258  # 146258 is the naturetype data height
        W_ratio = W / 64034  # 64034 is the naturetype data width

        # List all gt maps, to speed up (and save disk space for!) the input map save process
        list_gt_maps = np.sort(os.listdir(os.path.join(BASE_PATH_DATA, 'annot_maps_100x100')))
        ijs = [[int(vv.split('.')[0].split('_')[-2]), int(vv.split('.')[0].split('_')[-1])] for vv in list_gt_maps]

        # Iterate and save
        nbr_saved_im = 0
        for tot_ctr, ij in enumerate(ijs):
            i = ij[0]
            j = ij[1]
            if tot_ctr % 100 == 0:
                print("counter, number of saved", tot_ctr, nbr_saved_im)
            sub_im = im[i : i + CROP_SIZE, j : j + CROP_SIZE]

            # OBS: Change the save_name variable based on which map you are saving!!!
            save_name = save_names[outer_it] + '%d_%d.npy' % (i, j)
            #save_name = os.path.join(BASE_PATH_DATA, 'baseclass_input_maps_100x100/baseclass_map_%d_%d.npy' % (i, j))
            #save_name = os.path.join(BASE_PATH_DATA, 'vmi_input_maps_100x100/vmi_map_%d_%d.npy' % (i, j))
            #save_name = os.path.join(BASE_PATH_DATA, 'height_input_maps_100x100/height_map_%d_%d.npy' % (i, j))
            #save_name = os.path.join(BASE_PATH_DATA, 'soilmoistindex_input_maps_100x100/soil_moist_index_map_%d_%d.npy' % (i, j))
            #save_name = os.path.join(BASE_PATH_DATA, 'soilmoist_input_maps_100x100/soil_moist_map_%d_%d.npy' % (i, j))
            #save_name = os.path.join(BASE_PATH_DATA, 'nmd_input_maps_100x100/nmd_map_%d_%d.npy' % (i, j))
            #save_name = os.path.join(BASE_PATH_DATA, 'bush_input_maps_100x100/bush_map_%d_%d.npy' % (i, j))
            #save_name = os.path.join(BASE_PATH_DATA, 'bush_cover_input_maps_100x100/bush_cover_map_%d_%d.npy' % (i, j))
            #save_name = os.path.join(BASE_PATH_DATA, 'tree_input_maps_100x100/tree_map_%d_%d.npy' % (i, j))
            #save_name = os.path.join(BASE_PATH_DATA, 'tree_cover_input_maps_100x100/tree_cover_map_%d_%d.npy' % (i, j))
            #save_name = os.path.join(BASE_PATH_DATA, 'soilcover_input_maps_100x100/soil_cover_map_%d_%d.npy' % (i, j))
            
            np.save(save_name, sub_im)
            nbr_saved_im += 1
        print("DONE, SAVED NBR IM: ", nbr_saved_im)
    print("DONE OVERALL")
