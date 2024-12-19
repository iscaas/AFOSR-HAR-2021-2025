# Dataset Preparation

We provide our labels in the `labels` directory, for the `HMDB51`, `UCF50`, and `UCF101` dataset.

## HMDB51

1. We downloaded the offical version of HMDB51 dataset from the dataset provider [SERRE LAB](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads) and videos are placed in the data directory.

2. Once the dataset has been fully downloaded, generate the CSV files for training and validation using `csv_writer.py`, as `train.csv` and `val.csv`. The required format for the CSV files is as follows:

```
<path_1>,<label_1>
<path_2>,<label_2>
...
<path_n>,<label_n>
```
where `<path_i>` points to a video file, and `<label_i>` is an integer between `0` and `num_classes - 1`.

|       Dataset      | Number of Classes |  Total Number of Videos   | Train | Validation | Average Video Duration | Resoultion |
|:----------------:|:----------:|:----------:|:---:|:-------:|:-------:|:-------:|
| HMDB51 |    51   | 6766 | 5441 |  1335  | 1 ~ 10 sec | 320 × 240 |


## UCF50

1. We downloaded the offical version of UCF50 dataset from the dataset provider [UCF Center for Research in Computer Vision](https://www.crcv.ucf.edu/data/UCF50.php) and videos are placed in the data directory.

2. Once the dataset has been fully downloaded, generate the CSV files for training and validation using `csv_writer.py`, as `train.csv` and `val.csv`. The required format for the CSV files is as follows:

```
<path_1>,<label_1>
<path_2>,<label_2>
...
<path_n>,<label_n>
```
where `<path_i>` points to a video file, and `<label_i>` is an integer between `0` and `num_classes - 1`.

|       Dataset      | Number of Classes |  Total Number of Videos   | Train | Validation | Average Video Duration | Resoultion |
|:----------------:|:----------:|:----------:|:---:|:-------:|:-------:|:-------:|
| UCF50 |    50   | 6681 | 5360 |  1321  | 1 ~ 10 sec | 320 × 240 |


## UCF101

1. We downloaded the offical version of UCF101 dataset from the dataset provider [UCF Center for Research in Computer Vision](https://www.crcv.ucf.edu/data/UCF101.php) and videos are placed in the data directory.

2. Once the dataset has been fully downloaded, generate the CSV files for training and validation using `csv_writer.py`, as `train.csv` and `val.csv`. The required format for the CSV files is as follows:

```
<path_1>,<label_1>
<path_2>,<label_2>
...
<path_n>,<label_n>
```
where `<path_i>` points to a video file, and `<label_i>` is an integer between `0` and `num_classes - 1`.

|       Dataset      | Number of Classes |  Total Number of Videos   | Train | Validation | Average Video Duration | Resoultion |
|:----------------:|:----------:|:----------:|:---:|:-------:|:-------:|:-------:|
| UCF101 |    101   | 13320 | 10690 |  2630  | 3 ~ 10 sec | 320 × 240 |


<div align="center">
    <b>We provided the csv files (train.csv and val.csv) for each dataset in data directory, for the sake of understanding.</b>
</div>

