# DEM_segmentation

## Setup

### Prerequisites
- Tensorflow
- keras
- gdal 
  + Anaconda: `conda install -c conda-forge gdal`
  + pip : `pip install GDAL`

- Clone this repo:
```bash
git clone git@github.com:fffibonacci/DEM_segmentation.git
cd DEM_segmentation
```


## Generate dataset
###  Original TIFF files: 

  DEM:
  
    mclearn_roi.tif

  Derivatives:
  
    mclean_roi_slope.tif 
    mclean_roi_aspect.tif 
    mclean_roi_rough.tif 
    mclean_roi_tpi.tif 
    mclean_roi_tri.tif
  

  Merge To Generate:
  
    mclean_roi_merge.tif 

  Annotated Mask:
  
    annotated_mask.tif where 1 indicates annotated, 0 not

  Binary Mask:
  
    cl1.tif -- 5 m buffer around the features  with 1 means having feature there
    10m_buffer_3443.tif -- 10 m buffer around the features 

  
### 1.  Merge to 6 bands 
Combine the separate bands in a single image;  all bands will be initialized using 0 

```gdal_merge.py -separate -init 0 -o mclean_roi_merge.tif mclean_roi_slope.tif mclean_roi_aspect.tif mclean_roi_rough.tif mclean_roi_tpi.tif mclean_roi_tri.tif mclearn_roi.tif```

### 2. Create Tiled Images (Ex: Size 128x128 with overlap 64)

```
gdal_retile.py -v -r near -ps 128 128 -co “TILED=YES”  -targetDir frames_128_overlap  -tileIndex  tiles_frames  -overlap 64     -csv frames.csv  -csvDelim ,  mclean_roi_merge.tif 
```
```
gdal_retile.py -v -r near -ps 128 128 -co “TILED=YES”  -targetDir masks_128_overlap  -tileIndex  tiles_frames  -overlap 64     -csv masks.csv  -csvDelim ,  cl1.tif 
```

(or  10m_buffer_3443.tif for 10 m buffered feature)

```
gdal_retile.py -v -r near -ps 128 128 -co “TILED=YES”  -targetDir annotations_128_overlap  -tileIndex  tiles_annotations  -overlap 64     -csv annotations.csv  -csvDelim ,  annotated_mask.tif 
```

### 3. Filter the image files
After rename all the tiled image files, run notebook from [/DEM_segmentation/preprocess/selection_run.ipynb](https://github.com/fffibonacci/DEM_segmentation/blob/master/preprocess/selection_run.ipynb)

By running this notebook, we select the same number of files with features (image containing more than 100 labelled pixels) and without features.


## Train and Test (cross validation)
Train_frame_path and train_mask_path contains npy tile files.

```python cv_main.py```
