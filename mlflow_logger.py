import mlflow
import os
import logging
import getpass
import numpy as np
from datetime import datetime

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MLFlow_Logger:
    def __init__(self,experiment_name, mode='train'):

        # remote_server_uri = "https://10.153.51.172:4040" # set to your server URI
        # mlflow.set_tracking_uri("https://172.30.236.213:4040")
        mlflow.set_tracking_uri("file:///mnt/data/liao/yujia/mlflow/Ecoal_mask_time/mlruns/")
        mlflow.set_experiment(experiment_name)

    def metric_summary(self, key, value, step):
        'log scalar variable'
        mlflow.log_metric(key=key, value=value, step=step)
    
    def metrics_summary(self, value, step):
        'log scalar variables'
        mlflow.log_metrics(metrics=value, step=step)

    def artifact_summary(self, local_path):
        'log artifacts from given local path'
        mlflow.log_artifact(local_path=local_path)
    
    def save_and_log_image(self, tags, images, local_path, step):
        'images in tensor form'
        for i in range(len(images)):
            save_image(images[i], '%s/latest_%s_%03d.png' % (local_path,tags, i)) 
            mlflow.log_artifact(local_path='%s/latest_%s_%03d.png' % (local_path,tags, i))
    
    def args_summary(self, args):
        params_dict = {}
        for key, val in sorted(vars(args).items()):
            params_dict.update({key: val})
        mlflow.log_params(params_dict)

if __name__ == '__main__':
    with open("/home/configs/convotherm_bakeoff.yaml") as j:
       test_config = yaml.safe_load(j)
    
    cvB =  ml_logger(test_config["training_config"])
