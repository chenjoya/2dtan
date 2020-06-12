import os

from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "TAN"
_C.MODEL.WEIGHT = ""

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.NUM_PRE_CLIPS = 256
_C.INPUT.PRE_QUERY_SIZE = 300

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL.TAN = CN()
_C.MODEL.TAN.NUM_CLIPS = 128

_C.MODEL.TAN.FEATPOOL = CN()
_C.MODEL.TAN.FEATPOOL.POOLER = "FeatAvgPool"
_C.MODEL.TAN.FEATPOOL.INPUT_SIZE = 4096
_C.MODEL.TAN.FEATPOOL.HIDDEN_SIZE = 512
_C.MODEL.TAN.FEATPOOL.KERNEL_SIZE = 2

_C.MODEL.TAN.FEAT2D = CN()
_C.MODEL.TAN.FEAT2D.POOLING_COUNTS = [15,8,8,8]

_C.MODEL.TAN.INTEGRATOR = CN()
_C.MODEL.TAN.INTEGRATOR.QUERY_HIDDEN_SIZE = 512
_C.MODEL.TAN.INTEGRATOR.LSTM = CN()
_C.MODEL.TAN.INTEGRATOR.LSTM.NUM_LAYERS = 3
_C.MODEL.TAN.INTEGRATOR.LSTM.BIDIRECTIONAL = False

_C.MODEL.TAN.PREDICTOR = CN() 
_C.MODEL.TAN.PREDICTOR.HIDDEN_SIZE = 512
_C.MODEL.TAN.PREDICTOR.KERNEL_SIZE = 5
_C.MODEL.TAN.PREDICTOR.NUM_STACK_LAYERS = 8

_C.MODEL.TAN.LOSS = CN()
_C.MODEL.TAN.LOSS.MIN_IOU = 0.3
_C.MODEL.TAN.LOSS.MAX_IOU = 0.7

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 12
_C.SOLVER.LR = 0.01
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.TEST_PERIOD = 1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.MILESTONES = (8, 11)

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.NMS_THRESH = 0.4
 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
