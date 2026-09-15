#our project shared path and hyperparameters are defined here

from pathlib import Path
BASE_PATH = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_PATH/"clothing-dataset-small"
TRAIN_DIR = DATA_DIR/"train"
VAL_DIR = DATA_DIR/"validation"
TEST_DIR = DATA_DIR/"test"
MODELS_DIR = BASE_PATH/"src"/"models"
PLOTS_DIR = BASE_PATH/"src"/"plots"
INPUT_SIZE = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SIZE_INNER = 100 #number of nodes in the hidden layers
DROPRATE = 0.2 #this means that 20% of the nodes will be dropped
EPOCHS = 80

#the ten labels, in the alphabetical order keras assigned during training
#used when the training folders are not present, e.g. inside a container
CLASSES = ["dress", "hat", "longsleeve", "outwear", "pants",
           "shirt", "shoes", "shorts", "skirt", "t-shirt"]