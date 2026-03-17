from examples.experiments.task1_pick_banana_basket.config import TrainConfig as PickBananaBasketTrainConfig
from examples.experiments.task2_insert_block.config import TrainConfig as InsertBlockConfig
from examples.experiments.task3_insert_circle.config import TrainConfig as InsertCircleConfig
from examples.experiments.task4_erase_whiteboard.config import TrainConfig as EraseWhiteboardConfig
    
CONFIG_MAPPING = {
    "task1_pick_banana_basket": PickBananaBasketTrainConfig,
    "task2_insert_block": InsertBlockConfig,
    "task3_insert_circle": InsertCircleConfig,
    "task4_erase_whiteboard": EraseWhiteboardConfig,
}