# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.


# so here we wahe model tracking - we save what we want (with log) and where we want
# now we can have a 'project': saving to X so they can run it





import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from sktime.utils import mlflow_sktime  
from sktime.datasets import load_airline  
from sktime.forecasting.arima import ARIMA  

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Split the data into training and test sets. 1 year for test
    y = load_airline()
    y_train = y['1949':'1958']
    y_test = y['1959-01':'1959-12']

    final_run = bool(sys.argv[1]) if len(sys.argv) >= 1 else 0


    with mlflow.start_run():
        arima = ARIMA()
        arima.fit(y = y_train, X=y_train) # best exo ever

        arima_pred = arima.predict(fh=[1,2,3,4,5,6,7,8,9,10,11,12], X=y_test)

        (rmse, mae, r2) = eval_metrics(y_test, arima_pred)

        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        print("  Final_Run_Bool: %s" % final_run)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_param('cutoff', arima.cutoff)
        mlflow.log_param('final_run', final_run)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        artifact_file = 'arima_model'
        print('tracking:', tracking_url_type_store)
        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            # mlflow.sklearn.log_model(arima, "model", registered_model_name="sktimeArima")
            print('saved to file')

            ### SAVE MODEL IS TO SAVE LOCALLY, TO READ IT PROPERLY - LOG IT
            mlflow_sktime.log_model(sktime_model=arima, artifact_path=artifact_file) 
        else:
            # mlflow.sklearn.log_model(arima, "model")
            print('saved not to file')
            # why no artifacts saved?? when using sktime
            # but this also saves the copy locally - in the folder in which i run all components


            mlflow_sktime.log_model(sktime_model=arima, artifact_path=artifact_file) 

## training /tracking the model - done
## view staff - done
## logging the model (mlflow models) - done

# packaging the code