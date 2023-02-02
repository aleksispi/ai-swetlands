# AI-Swetlands: A pre-study about using artificial intelligence for semantic segmentation of Swedish wetland types

This repo contains Python- and PyTorch-based code for training and evaluating wetland segmentation models, that was used in a [pre-study about AI for wetland monitoring in Sweden](https://github.com/aleksispi/ai-swetlands/blob/main/ai-swetlands.pdf) during the fall of 2022. A popular scientific description in Swedish can be hound [here](https://www.naturvardsverket.se/om-oss/aktuellt/nyheter-och-pressmeddelanden/ai-teknik-testas-for-att-identifiera-vatmarker/).

### Funding and acknowledgements
This study was conducted by [Aleksis Pirinen](https://www.ri.se/sv/person/aleksis-pirinen), [RISE Research Institutes of Sweden](https://www.ri.se/en), for the [Swedish Environmental Protection Agency](https://www.naturvardsverket.se/en) (contact person and collaboration partner: [Matti Ermold](https://scholar.google.se/citations?user=2LwCBW8AAAAJ&hl=sv)) during September - November, 2022. The Swedish Environmental Protection Agency funded this work. We would like to thank [William Lidberg](https://www.slu.se/cv/william-lidberg2/) at the [Swedish University of Agricultural Sciences](https://www.slu.se/en/) for all the help with transforming model predictions to a format that works when visualizing in GIS.

### Data preprocessing
_Note: Before starting with this data preprocessing step, ensure you have access to the various tiff-files referenced in `preprocess_wetlands.py`._

The first file to consider is `preprocess_wetlands.py`, which will preprocess the input and output maps and save them as non-overlapping rectangles of a size that can be specified in `preprocess_wetlands.py` (100 x 100 patches were used in the pre-study). After this, use the file `save_and_align_data.py` to complete the preprocessing and data alignment.

### Training a model for wetland segmentation
_Note: Ensure you have completed the data preprocessing step above prior to this._

Model training (and validation on validation data) is performed using `wetland_training.py`. See the file `plot_results.py` if you are interested in tracking the progress of mIoU and other training statistics. Models are saved during and upon completion of training, and can subsequently be used for example to map Swedish wetlands across all of Sweden (see below). 

### Using a pre-trained segmentation model across all of Sweden to perform a full wetland mapping
_Note: Ensure you have trained a segmentation model (see above) prior to this._

The file `segment_all_of_sweden.py` contains the code needed to evaluate a pre-trained segmentation model at every location of Sweden. This can be very memory-demanding for the computer, so there is an option to run this code in "chunks" that splits Sweden into N subsets, by using the shell script `launch_loop.sh`.

Once you have evaluated the segmentation network in all locations, the results can be merged into a unified "wetland map" spanning all of Sweden, by using the script `merge_tiffs.py`. Finally, if you are interested in exporting the results to GIS, consider the file `write_geotiff.py`.
