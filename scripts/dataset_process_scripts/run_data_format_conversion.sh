#!/bin/bash
export http_proxy=http://192.168.32.28:18000 && export https_proxy=http://192.168.32.28:18000


python data_format_conversion.py \
  --input-raw /media/zhourui/data/aliyunpan_save/c11d2a00ee8e44ad9049730bb9cbc05a/Twin_RL/data/insert_block/ablation_data/twin_sft/outdomain/id_60 \
  --output /media/zhourui/data/twin-rl/task3_insert_block/ablation_data/twin_sft/indomain_60/indomain.pkl \
  --config /media/zhourui/Twin-RL/examples/experiments/task1_pick_banana_basket/config.py \
  --hand left \
  --stack-obs-num 2 \
  --action-def absolute \
  --octo-model-path /media/zhourui/Twin-Octo/checkpoints/octo-small \
  --fix-gripper \
  --fix-gripper-state 1.0
  # --delta-gripper \

