import numpy as np
from train import Train 
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=0.1, help="learning rate")
parser.add_argument("--epochs", default=3, help="number of epochs to train model")
parser.add_argument("--batch_size", default=32, help="Batch size used to train the model and to dump images/attributes from dataset")

args = parser.parse_args()

T = Train(args.learning_rate)
T.training(epochs = args.epochs, batch_size = args.batch_size)


