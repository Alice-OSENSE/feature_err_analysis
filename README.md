# feature_err_analysis
Analyze cv features's performance as a predictor of crowd size 

## Table of Content
- [Introduction](#introduction)
- [Result](#result)

## Introduction
Often, we want to estimate the crowd size from a given image. This project is to analyze whether simple, traditional cv features are robust predictors of crowd sizes. We assume that:
- The camera is static
- It's possible to obtain an image of background i.e. only showing static objects and no-people in sight
- The lighting is more or less static

Given the above assumptions, we chose the [UCSD Pedestrian Dataset](http://www.svcl.ucsd.edu/projects/peoplecnt/) for the experiment. In the UCSD Pedestrian Dataset, 

`Note`: Below,pmap refers to "perspective map".  

## Result
The description of each feature can be found here:
The visualization can be found here:

| Feature | with pmap | MSE |
|  ---    |   ---     | --- |
|Segment size| []     | 12.663326  |
|Segment size| [x]    |  6.231250 |
|Segment perimeter|[] | 16.910991  |
|Segment perimeter|[x]|  NA  |
||||
