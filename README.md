# Blood Glucose Forecasting

The objective of this project is to  to make predictions of future blood glucose values, based on measured values, 60 minutes into the future.  

## Getting Started

For a better experience in results visualization, graphing and training experience use Google Colab, Spyder or another simylar preferred tool.

### Prerequisites

tensorflow==2.0.0
tensorboard==2.0.1
pandas==0.25.1
numpy==1.16.5
matplotlib==3.1.1
sklearn==0.21.3


### Installing

You can run in your local environment or in an anaconda environment (recommended).

```
conda create -n bctechnologies python=3.7 anaconda
```

Install tensorflow and other packages:

```
pip install tensorflow
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn

```

## Folders

   
    ├── data                  # Model files and logs (subfolders will be created automatically)
    │   ├── blood-glucose-data.csv
    │   ├── distance-activity-data.csv         
    │   └── heart-rate-data.csv
    ├── models                # Model files and logs (subfolders will be created automatically)
    ├── hypertuning.py        # Hyperparameter optimization
    ├── ml_pipeline.py        # Machine Learning Pipeline
    ├── README.md 
    └── Requirements.txt

    
##Metrics evaluated

Accuracy by Clarke Error Grid Zones: (https://en.m.wikipedia.org/wiki/Clarke_Error_Grid)
RMSE (Root Mean Square Error)
    
## Sample of result

```
Overall RMSE with normalization:  0.36561007741476476
Overall RMSE without normalization:  30.55448697159304
Data points by zone:  [23855, 6354, 113, 1958, 0]
Accuracy in zones A:  73.9002478314746 %
Accuracy in zones A, B:  93.58426270136307 %
Accuracy in zones A, B, C:  93.93432465923172 %
Accuracy in zones A, B, C, D:  100.0 %
Accuracy in zones A, B, C, D, E:  100.0 %
```

[Clarke Error Grid](https://ibb.co/6gsDkfV)


## Built With

* [Tensorflow](https://www.tensorflow.org/)
* [Scikit-learn](https://scikit-learn.org/)

## Authors

* **Daniel Antonio da Silva** - [GitHub](https://github.com/dasilvadaniel)

## Acknowledgments

* [VanHack](https://vanhack.com/)

* Inspiration:
* * https://www.tensorflow.org/tutorials/structured_data/time_series
* * https://machinelearningmastery.com/multi-step-time-series-forecasting-with-machine-learning-models-for-household-electricity-consumption/
* * https://github.com/suetAndTie/ClarkeErrorGrid