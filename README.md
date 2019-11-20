## Bio Conscious Technologies

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

```
pip install tensorflow
```

### Installing

You can run in your local environment or in an anaconda environment (recommended).

```
conda create -n bctechnologies python=3.7 anaconda

```

Install tensorflow and other packages:

```
pip install tensorflow
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
    
## Result

```
RMSE normalized for 5 minutes:  0.21814742393029624
RMSE without normalization for 5 minutes:  18.23084970169596
Accuracy in zones A:  89.36802973977696 %
Accuracy in zones A, B:  96.31970260223048 %
Accuracy in zones A, B, C:  96.43122676579927 %
Accuracy in zones A, B, C, D:  100.0 %
Accuracy in zones A, B, C, D, E:  100.0 %
```
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