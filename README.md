# Spatial Transformation Function
In this repository, there are two versions of the spatialtransform function. One in R and the other in Python. You can use either of them in their repsective folders. Below is documentation of for each function. 

## R
### transform_to_ind(formula, trainData, trainLocs, testData, testLocs, MaternParam=NULL, smoothness = 0.5, M = 30, ncores)

#### Description
**transform_to_ind** is a function designed to decorrelate spatially dependent data, specifically in the continuous univariate case.

#### Arguments 
* formula: An object of class "formula" describing the model to be decorrelated.
* trainData: An object of class data.frame containing the training data with the response variable provided.
* trainLocs: A matrix object containing the coordinates of the training data. The dimensions should be nx2.
* testData: An object of class data.frame containing the data to be predicted.
* testLocs: A matrix object containing the coordinates of the test data. The dimensions should be nx2.
* MaternParams: A vector of two parameters: range and nugget. **Range** represents how fast the correlation decays with distance and **nugget** represents the variability in one location. The default is NULL (rng, nug) where the range and nugget parameter are estimated automatically.
* smoothness: The smoothness parameter, which controls the smoothness of the function. The default is 1/2 which results in an exponential kernel. 
* M: The number of neighbors to consider when creating a correlation matrix for each individual observation. The default is 30.
* ncores: The number of cores to parallelize the decorrelation process.

  

### back_transform_to_spatial(preds, transformObj)

#### Description
**back_transform_to_spatial** is a function designed to back transform the predictions to their spatial state in the continous univariate case.

#### Arguments 
* preds: A vector of predictions from the machine learning model
* transformObj: The object outputted **IndData** after running **transform_to_ind** 




#### Look to the folder named **R Function** for guidance on how to implemenet these two functions. 

## Python 

## transform_to_ind(formula, trainData, trainLocs, testData, testLocs, range_param, nugget, smoothness = 0.5, M = 30, ncores)

## Description
**transform_to_ind** is a function designed to decorrelate spatially dependent data, specifically in the continuous univariate case.

### Arguments 
* formula: An object of class "formula" describing the model to be decorrelated.
* trainData: An object of class data.frame containing the training data with the response variable provided.
* trainLocs: A matrix object containing the coordinates of the training data. The dimensions should be nx2.
* testData: An object of class data.frame containing the data to be predicted.
* testLocs: A matrix object containing the coordinates of the test data. The dimensions should be nx2.
* range_param: A value that represents how fast the correlation decays with distance
* nugget: A value that represents the variability in one location. 
* smoothness: The smoothness parameter, which controls the smoothness of the function. The default is 1/2 which results in an exponential kernel. 
* M: The number of neighbors to consider when creating a correlation matrix for each individual observation. The default is 30.
* ncores: The number of cores to parallelize the decorrelation process. At this time keep the ncores at 1 (We will be updating this shortly). 

  

## back_transform_to_spatial(preds, transformObj)

## Description
**back_transform_to_spatial** is a function designed to back transform the predictions to their spatial state in the continous univariate case.

### Arguments 
* preds: A vector of predictions from the machine learning model
* transformObj: The object outputted **IndData** after running **transform_to_ind** 




## Look to folder named **Python Function** for guidance on how to implemenet these two functions. 

