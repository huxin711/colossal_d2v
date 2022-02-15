BATCH_SIZE = 256
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 3e-2

TENSOR_PARALLEL_SIZE = 4
TENSOR_PARALLEL_MODE = '2d'

NUM_EPOCHS = 800
WARMUP_EPOCHS = 40
max_norm = 2.

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

seed = 77

LOG_PATH = f"./vit_{TENSOR_PARALLEL_MODE}_cifar10_tp{TENSOR_PARALLEL_SIZE}_bs{BATCH_SIZE}_lr{LEARNING_RATE}/"


find_unused_parameters = True