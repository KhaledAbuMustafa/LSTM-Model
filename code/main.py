from data_loader import DataLoader
from data_preparation import Data_prep
from sine_generator import SineGenerator
from model_generator import ModelGenerator
import os



def main():
    data_loader = DataLoader()
    data_preparation = Data_prep()
    sine_generator = SineGenerator()
    model_generator = ModelGenerator()


    # Curve to be predicted
    file = r"file"
    X,y = data_loader.read_data_from_file(file)
    model_generator.run_pred(X, y)