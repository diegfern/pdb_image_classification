import time
import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from modules.architectures import Models
from sklearn import preprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf


data_path = sys.argv[1]
dim = int(sys.argv[2])
architecture = sys.argv[3]
export_path = sys.argv[4]
apply_scale = int(sys.argv[5])
number_epochs = int(sys.argv[6])
task = sys.argv[7]

time_inicio = time.time()

train_dataset, test_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        color_mode="grayscale",
        subset="both",
        seed=123,
        batch_size=32)


AUTOTUNE = tf.data.AUTOTUNE
labels = np.array(train_dataset.class_names)

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)


if apply_scale == 1:
    #scale
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))
    
#instance object
models = Models(train_dataset, test_dataset, dim, labels, task, architecture)

models.fit_models(number_epochs, 1)

metrics = models.get_metrics()
time_fin = time.time()
delta_time = round(time_fin - time_inicio, 4)

metrics["total_time"] = delta_time
metrics["dataset"] = data_path.split("/")[-2]
metrics["epochs"] = number_epochs

if not os.path.exists(export_path):
    os.mkdir(export_path)

export_file=f"{export_path}results_size-{dim}_scale-{apply_scale}_task-{task}.json" 
with open(
        export_file,
        mode="w", 
        encoding="utf-8") as file:
    json.dump(metrics, file)
print(f"Saved in {export_file}")
print(pd.json_normalize(metrics))
