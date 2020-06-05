

# Landsat-8 Classification 

---------------------


### Prerequisites

Python >=3.5

Tensorflow >=2.0

[GDAL](https://gdal.org/)

[Rasterio](https://github.com/mapbox/rasterio/)

[Shapely](https://github.com/Toblerity/Shapely/)


### Sites for Training

The input training sites are provided in the file: 

```bash
./sites_train.csv
```

![sites](https://github.com/shengsheng555/landsat-cls/blob/master/figure/sites.jpg?raw=true)



### Download and Process Data

```bash
python data_processing.py
```

The script downloads Landsat-8 and NLCD(2016) data from Amazon S3.

For Landsat-8, input Lat/Lon are converted to Path/Row to match
the product ID of scenes. The level 1 scenes in 2016 with corresponding path/row and the least cloud coverage are downloaded from Amazon S3 Storage. OLI bands 1-7 and 9 are extracted.

The whole process might take a long time and requires at least 22G disk space, depending on the exact input sites.

For each input site, the script crops a 3840 m x 3840 m rectangle images, each with the given site located in the center.

The extracted data are saved as a numpy array with datatype "uint16" and shape (sample_index, x-coor, y-coor, band).

### Extracted Data Example


Below is an example. First 8 images are from Landsat-8. The last image is from NLCD.

![example](https://github.com/shengsheng555/landsat-cls/blob/master/figure/example.png?raw=true)


### Train the CNN

```bash
python train.py
```

The script reads numpy array, generates small patches and train the CNN.

### Classify sites using pretrained model

```bash
python classify.py
```

The script loads the pretrained model from:

```bash
./pretrained.hdf5
```

The sites for classification are given in: 

```bash
./sites_classify.csv
```

Landsat-8 data are downloaded, cropped and classified.

The output array contains class labels:

```bash
./arr_cls.npy 
```

As the model is patch-based, the boundary pixels are unclassified. One can preset the size of image to classify a larger area.


### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
