# Offline Knowledge Distillation 
A 3DCNN-Based Knowledge Distillation Framework for Human Activity Recognition [paper](https://www.mdpi.com/2313-433X/9/4/82)

<img src="readme_images/framework.gif" width="800"/>

## Introduction
This repo contains the implementation of our proposed 3DCNN based knowledge distillation approach for human activity recognition, the prerequisite libraries, and the obtained quantiative results across different human activity recognition datasets.  

## Installation
This code is written in python 3.8. Install Anaconda python 3.8 and clone the repo using the following command:
```
git clone https://github.com/hayatkhan8660-maker/Response-based-offline-knowledge-distillation-for-HAR
```
## Prerequisites
### Recommended Environment
- Python 3.8
- Tensorflow 2.7
- Keras 2.7
  
### Dependencies
The following libraries need to be installed
- numpy == 1.24.4
- sklearn == 1.3.0
- scikitplot == 0.3.7
- matplotlib == 3.7.3
- tensorboard == 2.14.0
- scipy == 1.10.1

run ```pip install -r requirements.txt``` to install all the dependencies.

### Data Preparation
We have conducted experiments on four HAR datasets, including UCF11, UCF50, HMDB51, and UCF101 datasets. 
- UCF11 [dataset link](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php)
- UCF50 [dataset link](https://www.crcv.ucf.edu/data/UCF50.php)
- HMDB51 [dataset link](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#dataset)
- UCF101 [dataset link](https://www.crcv.ucf.edu/data/UCF101.php)

Before training, we have converted each video dataset to frames data using ```video_to_frames.py``` file as follows: 
```
python video_to_frames.py --input_path Datasets/dataset_name/ --sequence_length 16 --frame_height 224 --frame_width 224 --output_path Output_dataset/dataset_name/
``` 
The above ```video_to_frames.py``` file will generate five files. 1) ```Format_Time.csv``` listing amount of time and memory taken for frames extraction processes per class. 2) ```Format_Time_Summary.txt``` provide the runtime history (in terms of time and memoray allocation) of frames extraction process. 3) ```frames.npy``` contains video frames for the entire dataset (with a skip_frames_window = total_num_frames/sequence_length). 4) ```labels.npy``` contains numerical labels of each video. 5) ```video_file_paths.npy``` contains the path of all videos of a given dataset.

After preparing all datasets (converting from videos to frames using ```video_to_frames.py``` file), your dataset directory should have the following structure:
```
Frames Datasets
├── UCF11
│   ├── Format_Time.csv
│   ├── Format_Time_Summary
|   ├── frames.npy
|   ├── labels.npy
|   ├── video_file_paths.npy
├── UCF50
│   ├── Format_Time.csv
│   ├── Format_Time_Summary
│   ├── frames.npy
│   ├── labels.npy
│   ├── video_file_paths.npy
├── HMDB51
│   ├── Format_Time.csv
│   ├── Format_Time_Summary
│   ├── frames.npy
│   ├── labels.npy
│   ├── video_file_paths.npy
├── UCF101
│   ├── Format_Time.csv
│   ├── Format_Time_Summary
│   ├── frames.npy
│   ├── labels.npy
│   ├── video_file_paths.npy
```
Note: Sequence length, frames width, and frame height are subjective, one can choose different values for these arguements. In our work we used (sequence length = 16), (frame height = 224), and (frame width = 224). 

## Training
Since we proposed an offline knowledge distillation based approach, where a teacher model is pre-trained and a student model yet to be trained. So, before knowledge distillation one must either fintued a pre-trained C3D model on the dataset (same dataset going to use for knowledge distillation) or locally train C3D model from the scratch. In this work, we assess both type of teachers in terms of knowledge distillation performance. To train a teacher model run the following command:
```
python Teacher_training.py --data [path/././] --annotations [path/././] --batch_size [some postive integer] --epochs [some positive integer] --output_path [path/././] --log_path [path/././] --task [task name]
```
- ``` --data ``` path to training data (containing video frames)
- ``` --annotations ``` path to annotations of training data (labels per videos for the entire dataset)
- ``` --batch_size ``` size of the batch (can be choose based on the size of dataset and available computational resources)
- ``` --epochs ``` number of epochs for training a model 
- ``` --output_path ``` path to directory for saving trained model (file extension should be .h5)
- ``` --log_path ``` path to directory for saving trained training history (file extension should be .npy)
- ``` --task ``` takes task either (fintuning_C3D_Sports1M) for finetuning pre-trained C3D teacher model (reffered as TUTL in paper) previously trained on Sports 1Million Dataset or (train_local_teacher) for training C3D teacher model (reffered as TFS in paper) from scratch.

### Fintuning Pre-trained C3D Teacher
```
python Teacher_training.py --data Frames_datasets/dataset_name/frames.npy --annotations Frames_datasets/dataset_name/labels.npy --batch_size 8 --epochs 50 --output_path trained_models/model_dir/model_name.h5 --log_path training_histories/model_dir/hist_50 --task fintuning_C3D_Sports1M
```
### Training C3D Teacher from Scratch
```
python Teacher_training.py --data Frames_datasets/dataset_name/frames.npy --annotations Frames_datasets/dataset_name/labels.npy --batch_size 8 --epochs 50 --output_path trained_models/model_dir/model_name.h5 --log_path training_histories/model_dir/hist_50 --task train_local_teacher
```
### Knowledge Distillation Training
The knowledge distillation training file can be find with the name ``` KD_training.py ```. To start knowledge distillation training, run the follwing command:
```
python KD_training.py --data [path/././] --annotations [path/././] --batch_size [some postive integer] --epochs [some positive integer] --temperature [some positive integer] --source [oath/././] --output_path [path/././] --log_path [path/././]
```
- ``` --data ``` path to training data (containing video frames)
- ``` --annotations ``` path to annotations of training data (labels per videos for the entire dataset)
- ``` --batch_size ``` size of the batch (can be choose based on the size of dataset and available computational resources)
- ``` --epochs ``` number of epochs for training a model
- ``` --temperature ``` takes an integer value used for smoothing the softmax probabilities. The smoothen pobabilities of teacher and student models help in the convergence of distillation loss.
- ``` --source ``` path to trained teacher model (either finetuned C3D teacher model (reffered as TUTL in paper) or locally trained C3D teacher model (reffered as TFS in paper) from the scratch). 
- ``` --output_path ``` path to directory for saving trained model (file extension should be .h5)
- ``` --log_path ``` path to directory for saving trained training history (file extension should be .npy)

For instance, to train student 3DCNN model under the supervision of finetuned C3D teacher model (reffered as TUTL in paper), run the following command: 
```
python KD_training.py --data Frames_dataset/UCF101/frames.npy --annotations Frames_dataset/UCF101/labels.npy --batch_size 8 --epochs 100 --temperature 10 --source trained_models/UCF50/pretrained_C3D_teacher_S1M_UCF101.h5 --output_path KD_trained_models/UCF101/KD_student_trained_UCF101.h5 --log_path KD_training_histories/UCF101/KD_train_hist_UCF101
```
## Evaluation of Trained Models
To evaluate trained models, run the ```eval.py``` using following command:
```
python eval.py --data [path/././] --annotations [path/././]  --recognizer [path/././]
```
- ``` --data ``` path to test data (one can load the entire dataset and later split the dataset into training and test sets.)
- ``` --annotations ``` path to annotations of test data (labels per videos for the test set)
- ``` --recognizer ``` path to the trained model
For instance, to evaluate a trained model (trained on UCF101 dataset) on the test set of UCF101 dataset, run the above command as follows:
```
python eval.py --data Frames_dataset/UCF101/frames.npy --annotations Frames_dataset/UCF101/labels.npy --recognizer trained_models_data/UCF101/Models/Student_with_KD_T10_under_Teacher_with_pretrained_weights_UCF101.h5
```
Quantitative comparison of the obtained accuracies with different models and settings on UCF11, UCF50, HMDB51, and UCF101 datasets.

| Model  | UCF11 | UCF50| HMDB51 | UCF101 |                                  
| ------------- | ------------- | ------------- | ------------- | ------------- |               
| Student w/o KD | 88.71 | 95.81 | 88.24 | 93.74 |                           
| TFS Teacher | 95.86 | 96.40 | 90.67 | 95.04 |
| TUTL Teacher | 99.70 | 98.37 | 93.10 | 98.83 |
| Student<sub>3DCNN</sub>-TFS (T=1) | 95.73| 96.25 | 88.62 | 95.11 |
| Student<sub>3DCNN</sub>-TFS (T=5) | 96.84 | 96.70 | 88.25 | 95.64 |
| Student<sub>3DCNN</sub>-TFS (T=10) | 97.56 | 97.60 | 91.55 | 96.73 |
| Student<sub>3DCNN</sub>-TFS (T=15) | 96.43 | 96.91 | 87.22 | 96.11 |
| Student<sub>3DCNN</sub>-TFS (T=20) | 96.58 | 97.21 | 86.92 | 96.20 |
| Student<sub>3DCNN</sub>-TUTL (T=1) | 95.12 | 96.65 | 89.80 | 96.24 |
| Student<sub>3DCNN</sub>-TUTL (T=5) | 97.45 | 97.15 | 89.14 | 96.80 |
| Student<sub>3DCNN</sub>-TUTL (T=10) | 98.78 | 97.71 | 92.89 | 97.36 |
| Student<sub>3DCNN</sub>-TUTL (T=15) | 98.17 | 97.53 | 89.43 | 96.58 |
| Student<sub>3DCNN</sub>-TUTL (T=20) | 97.88 | 97.64 | 89.66 | 95.90 |


## Citation
Please cite our paper, if you want to reproduce the results using this code.
```
@article{ullah20233dcnn,
  title={A 3DCNN-Based Knowledge Distillation Framework for Human Activity Recognition},
  author={Ullah, Hayat and Munir, Arslan},
  journal={Journal of Imaging},
  volume={9},
  number={4},
  pages={82},
  year={2023},
  publisher={MDPI}
}
```

```
Ullah, H., & Munir, A. (2023). A 3DCNN-Based Knowledge Distillation Framework for Human Activity Recognition.
Journal of Imaging, 9(4), 82.
```
