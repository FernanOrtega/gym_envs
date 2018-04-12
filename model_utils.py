import os

from keras.models import model_from_yaml
from keras.optimizers import RMSprop


def load_keras_model(model_path):
    model_yaml_path = model_path + ".yaml"
    model_h5_path = model_path + ".h5"
    if not os.path.exists(model_yaml_path) or not os.path.exists(model_h5_path):
        return None
    # load YAML and create model
    yaml_file = open(model_yaml_path, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    model.compile(RMSprop(), 'MSE')
    # load weights into new model
    model.load_weights(model_h5_path)

    return model


def save_model(model, model_path):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(model_path + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(model_path + ".h5")