# config.py

# Dataset
DATASET_NAME = "mschonhardt/georges-1913-normalization"
TEST_SIZE = 0.2
VAL_SPLIT = 0.5
RANDOM_SEED = 42

# Vocabulary
SPECIAL_TOKENS = {
    "<pad>": 0,   # Padding token
    "<sos>": 1,   # Start of sequence
    "<eos>": 2,   # End of sequence
    "<unk>": 3    # Unknown token
}

# Model parameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.3
MAX_LENGTH = 100

# Training
BATCH_SIZE = 4096
LEARNING_RATE = 0.0005
EPOCHS = 10
TEACHER_FORCING_RATIO = 0.5
CLIP_GRAD_NORM = 1.0

# Scheduler
SCHEDULER_PATIENCE = 2
SCHEDULER_MODE = "min"
SCHEDULER_VERBOSE = True

# File paths
MODEL_SAVE_PATH = "normalization_model.pth"
VOCAB_SAVE_PATH = "vocab.pkl"
