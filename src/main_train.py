import numpy as np
from train import Train 
import argparse




parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=0.1, help="learning rate")
parser.add_argument("--epochs", default=3, type=int, help="number of epochs to train model")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size used to train the model and to dump images/attributes from dataset")
parser.add_argument("--attributs", default=["Smiling","Young","Pointy_Nose"], nargs="+", help="Name attributs")
# parser.add_argument("--weights", default="", help="Path to model already saved")



args = parser.parse_args()

#, nargs="+", action="append" , type=str type=attr_flag,
T = Train(args.learning_rate)
T.training(epochs = args.epochs, batch_size = args.batch_size, attributs=args.attributs)




