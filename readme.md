# Readme

## Environments:

The code works well with Python 3.7 and Pytorch 1.8.1.

## File Folders:

1. Folder [data_normalized](https://github.com/Trenchant-ymz/DeepLearning/tree/master/data_normalized) contains normalized data used for taining/validation/testing. Refer to [data_description.md](https://github.com/Trenchant-ymz/DeepLearning/blob/master/data_normalized/data_description.md) for details of the data.

## Files

1. [main.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/main.py) is used to train/test a model with data in folder [data_normalized](https://github.com/Trenchant-ymz/DeepLearning/tree/master/data_normalized).
2. [nets.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/nets.py) defines the model.
3. [obddata.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/obddata.py) defines the dataloader.
4. [best.mdl](https://github.com/Trenchant-ymz/DeepLearning/blob/master/best.mdl) contains the information of a trained model.
5. [prediction_result.csv](https://github.com/Trenchant-ymz/DeepLearning/blob/master/prediction_result.csv) is an example prediction result. 



Change Log
-----

### 2021/4/10
Version 1.0

Version 1.1: In [main.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/main.py): Add "map_location=device" in "model.load_state_dict()".

Version 1.2: np.array(tensor.cpu())