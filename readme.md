# Readme

## Environments:

The code works well with [python](https://www.python.org/) 3.7, 
[pytorch](https://pytorch.org/) 1.8.1, 
and **[osmnx](https://github.com/gboeing/osmnx)  0.16.1**.

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
   [prediction_result.csv](https://github.com/Trenchant-ymz/DeepLearning/blob/master/prediction_result.csv).
3. [nets.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/nets.py) defines the deep learning estimation model.
4. [obddata.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/obddata.py) defines the dataloader.
5. [estimationModel.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/estimationModel.py)
   loads the pretrained model from folder [pretrained models](https://github.com/Trenchant-ymz/DeepLearning/tree/master/pretrained%20models). 
6. [edgeGdfPreprocessing.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/edgeGdfPreprocessing.py)
   extracts features of segments of the edge.gdf from the openstreetmap.
   
7. [routingAlgorithms.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/routingAlgorithms.py),
[pathGraph.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/pathGraph.py),
   [osmgraph.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/osmgraph.py),
   [spaitalShape.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/spaitalShape.py) and
   [window.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/window.py)
   are functions/classes used in [eco routing.py](https://github.com/Trenchant-ymz/DeepLearning/blob/master/eco%20routing.py).



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

### 2021/5/19
version 4.0: Extract elevation data using ESRI Summarize Elevation API; Add functions which calculate length/ energy/ time for each path.

### 2021/5/26
version 4.1: Add a function which plots the calculated routes into a **vector** figure.

### 2021/9/3
version 5.0: Using a new routing algorithm(A*) which reduces the routing time from 3.3 hours to 1.5 hour 

### 2021/9/13
version 5.2: Using RBTree to store the value of nodes in the graph; Using dictionary to store the features of edges.

### 2021/9/19
version 5.3: Implement a Look-up-table method: Precalculate the fuel consumption of all the segments and save it to a look-up-table. 
Reducing time complexity of eco-routing algorithm from 12 minutes to 140s.