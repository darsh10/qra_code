

from .dan import DAN
from .lstm import LSTM

MODEL_CLASS = {
    'dan': DAN,
    'lstm' : LSTM
}

def get_model_class(model_name):
    model_name = model_name.lower()
    model = MODEL_CLASS.get(model_name, None)
    if model is None:
        raise Exception("Unknown model class: {}".format(
            model_name
        ))
    return model

def get_decoder_model_class():
    model_name = "lstm_decoder"
    model = MODEL_CLASS.get(model_name, None)
    if model is None:
        raise Exception("Unknown model class: {}".format(
            model_name
        ))
    return model
