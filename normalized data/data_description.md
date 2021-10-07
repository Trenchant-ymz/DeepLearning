# Description of Data in Filefolder ”normalized data"

## Folders:

In folder [10](https://github.com/Trenchant-ymz/DeepLearning/tree/master/normalized%20data/10): 80% data are used for training; 10% for validation; 10% for testing

In folder [20](https://github.com/Trenchant-ymz/DeepLearning/tree/master/normalized%20data/20): 60% data are used for training; 20% for validation; 20% for testing

## File Format:
There are five columns in each csv file:
1. numerical features (*Normalized*)
    - speed limit, 
    - mass, 
    - elevation change, 
    - previous orientation, 
    - length, 
    - direction angle
2. label
    - fuel consumption (*l*)
    - time (*s*)
3. segment_id : *id* of the segment
4. length of the trip: a trip is defined by a **real** trajectory; the length of a trip is defined as the number of **segments** in the trip
5. position in the trip 
6. road_type (*categorical feature*)  
7. time_stage (*categorical feature*)  
8. week_day (*categorical feature*)  
9. lanes (*categorical feature*)  
10. bridge (*categorical feature*)  
11. endpoint_1 (*categorical feature*)  
12. endpoint_2 (*categorical feature*)  

## Data Format:
1. For numerical feature data, we normalize these features into a normal distribution N(0,1), and the 
original mean/std are saved in 
   [mean_std.csv](https://github.com/Trenchant-ymz/DeepLearning/blob/master/statistical%20data/mean_std.csv).
   
2. For categorical features, we use one-hot encoding method to encoding them. See 
   [endpoints of segments.csv](https://github.com/Trenchant-ymz/DeepLearning/blob/master/statistical%20data/endpoints_dictionary.csv),
   [road type.csv](https://github.com/Trenchant-ymz/DeepLearning/blob/master/statistical%20data/road_type_dictionary.csv)




Change Log
-----

### 2021/4/10
Create a new data_description.md

### 2021/5/14
Add description of the 6 categorical features

### 2021/5/14
Using simulated data (generated from the physics model) to train the NN model