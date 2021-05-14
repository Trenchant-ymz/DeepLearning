# Readme

## Environments:

The code works well with Python 3.7, Pytorch 1.8.1, and **osmnx 0.16.1**.

## File Folders:

1. Folder [normalized data](https://github.com/Trenchant-ymz/DeepLearning/tree/master/normalized%20data) 
   contains normalized data used for taining/validation/testing. 
   Refer to [data_description.md](https://github.com/Trenchant-ymz/DeepLearning/blob/master/normalized%20data/data_description.md) 
   for details of the data.
2. Folder [GraphDataInBox](https://github.com/Trenchant-ymz/DeepLearning/tree/master/GraphDataInBbox) 
   contains a graph downloaded from [openstreemap](https://www.openstreetmap.org/)
   : "osmGraph.graphhml".
3. Folder [pretrained models](https://github.com/Trenchant-ymz/DeepLearning/tree/master/pretrained%20models)
contains a pretrained energy estimation model and a pretrained travel time estimation model with 12 features considered.
4. Folder [statistical data](https://github.com/Trenchant-ymz/DeepLearning/tree/master/statistical%20data)
contains the statistical data of our data used in the pre-training stage, including the 
   [statistical characteristics (mean/std)](https://github.com/Trenchant-ymz/DeepLearning/blob/master/statistical%20data/mean_std.csv)
   (mean/std) of the numerical features; the one-hot encoding of some categorical features 
   ([endpoints of segments.csv](https://github.com/Trenchant-ymz/DeepLearning/blob/master/statistical%20data/endpoints_dictionary.csv),
   [road type.csv](https://github.com/Trenchant-ymz/DeepLearning/blob/master/statistical%20data/road_type_dictionary.csv))
   
   
## Files
1. In [eco routing.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/eco%20routing.py), a(n)
   shortest/ eco/ fastest route can be extracted given a request of a origin-destination pair and a bounding-box.
2. [training and testing.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/training%20and%20testing.py) 
   is used to train/test a model with data in 
   folder [normalized data](https://github.com/Trenchant-ymz/DeepLearning/tree/master/normalized%20data).
   In the "main" function in the script, use mode="train" to train a model and save it to 
   folder [pretrained model](https://github.com/Trenchant-ymz/DeepLearning/tree/master/pretrained%20model);
   use mode="test" to test the performance of the model;
   use mode="test" and output = True to test the model and save the ground truth / estimated value into 
   [prediction_result.csv](https://github.com/Trenchant-ymz/DeepLearning/blob/master/prediction_result.csv)
3. [nets.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/nets.py) defines the deep learning estimation model.
4. [obddata.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/obddata.py) defines the dataloader.
5. [estimationModel.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/estimationModel.py)
   loads the pretrained model from folder [pretrained models](https://github.com/Trenchant-ymz/DeepLearning/tree/master/pretrained%20models) 
6. [edgeGdfPreprocessing.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/edgeGdfPreprocessing.py)
   extracts features of segments of the edge.gdf from the openstreetmap.
   
7. [pathGraph.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/pathGraph.py),
   [osmgraph.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/osmgraph.py),
   [spaitalShape.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/spaitalShape.py) and
   [window.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/window.py)
   are classes used for finding the shortest/ eco/ fastest route.



Change Log
-----

### 2021/4/10
Version 1.0

Version 1.1: In [main.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/main.py): Add "map_location=device" in "model.load_state_dict()".

Version 1.2: np.array(tensor.cpu())

### 2021/4/30
version 2.0: 6 categorical features are added into the model.

### 2021/5/13
version 3.0: Add an eco-routing script "eco routing.py" which can 
estimating the eco route/ the shortest route/ the fastest route using the model trained by main.py

### Coming Soon
version 3.5: Add a functions which can calculate length/ energy/ time for a given path.