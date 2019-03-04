from celery_config import celery_app
import pickle
import json
from appConfig import model_folder
import os
import numpy as np

from sklearn import svm

@celery_app.task(name='mytasks.hello')
def hello(a, b):
    return a + b


@celery_app.task(name='mytasks.getmodel')
def getModel():

    return NotImplementedError

@celery_app.task(name='mytasks.trainModel')
def trainModel(X, outliers_fraction, id):
    data_string = json.loads(X)
    data = np.array(data_string)

    s = svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                    gamma=0.1)

    s.fit(data)
    model_file = "svm_model_{}.pkl".format(id)
    model_path = os.path.join(model_folder, model_file)
    with open(model_path, 'wb') as file:
        pickle.dump(s, file)

    return model_path
