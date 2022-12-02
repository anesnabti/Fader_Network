import numpy as np
from train import Train 
import argparse




parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=0.1, help="learning rate")
parser.add_argument("--epochs", default=3, type=int, help="number of epochs to train model")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size used to train the model and to dump images/attributes from dataset")
parser.add_argument("--attributs", default=["Smiling"], nargs="+", help="Name attributs")
parser.add_argument("--nbr_itr", default=100, type=int, help="number of iteration per epoch")
#["C:/Users/33660/Desktop/Etudes/Master_2/SEMESTRE_1/MACHINE_LEARNING_AVANCEE/Projet/Fader_Network/utils/models/discriminator_Smiling_Young_Pointy_Nose_30-11-2022_14h06.h5 ", "C:/Users/33660/Desktop/Etudes/Master_2/SEMESTRE_1/MACHINE_LEARNING_AVANCEE/Projet/Fader_Network/utils/models/AutoEncoder_Smiling_Young_Pointy_Nose_30-11-2022_14h06.h5"]
#parser.add_argument("--weights", default=" " , type=str, nargs="+", help="Path to model already saved (Discriminator/AutoEncoder)" )


args = parser.parse_args()

T = Train(lr = args.learning_rate, attributs=args.attributs, epochs=args.epochs, nbr_itr_epoch=args.nbr_itr)
T.training(batch_size = args.batch_size)
T.plot_loss()

