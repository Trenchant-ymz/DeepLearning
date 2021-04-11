# Description of Data in Filefolder data_normalized

## Folders:

In folder [10](https://github.com/Trenchant-ymz/DeepLearning/tree/master/data_normalized/10): 80% data are used for training; 10% for validation; 10% for testing

In folder [20](https://github.com/Trenchant-ymz/DeepLearning/tree/master/data_normalized/20): 60% data are used for training; 20% for validation; 20% for testing

## File Format:
There are five columns in each csv file:
1. data list (*Normalized*)
    - position, 
    - road type, 
    - speed limit, 
    - mass, 
    - elevation change, 
    - previous orientation, 
    - length, 
    - direction angle
2. label
    - fuel consumption (*10ml*): *10ml* is used as the unit of fuel consumption so that fuel and time consumption are of the same order of magnitude.
    - time (*s*)
3. segment_id : *id* of the segment
4. length of the trip: a trip is defined by a **real** trajectory; the length of a trip is defined as the number of **segments** in the trip
5. position in the trip

## Data Format:
There are eight features in a data list:
1. position: relative position in trip
    - mean before normalization: 74.207860,
    - std before normalization: 70.476971 
2. road type: sort road types according to their average speed limits:
    - mean: 16.630189, 
    - std: 4.874619
3. speed limit:
    - mean (km/h): 79.784291, 
    - std: 21.937249
4. mass:
    - mean(kg): 23151.526634, 
    - std: 8291.613022
5. elevation change:
    - mean(m): -0.041187,
    - std: 8.632725
6. previous orientation: Turning angle from the previous segment (0 in default, right turn > 0, left turn < 0)
    - mean(degree): -2.066254,
    - std: 35.908657
7. length: :
    - mean(m): 600.016867,
    - std: 893.074010
8. direction angle: Direction angle of the segment based on east direction (north>0)
    - mean(degree): 1.896802,
    - std: 102.726617	




Change Log
-----

### 2021/4/10
Create a new data_description.md
