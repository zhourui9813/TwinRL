export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95 && \
export DISPLAY=""

echo "user2001:x:2001:2001:User 2001:/home/user2001:/bin/bash" | tee -a /etc/passwd
echo "group2001:x:2001:" | tee -a /etc/group

# Generate two four-digit random port numbers (1000–9999) and ensure they are non-repeating.
PORT_NUMBER=$(( (RANDOM % 9000) + 1000 ))
BROADCAST_PORT=$(( (RANDOM % 9000) + 1000 ))
while [[ "$BROADCAST_PORT" -eq "$PORT_NUMBER" ]]; do
  BROADCAST_PORT=$(( (RANDOM % 9000) + 1000 ))
done

export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EXP_NAME=task4_insert_circle
DEMO_PATH=/media/zhourui/data/twin-rl/task3_insert_block/real30_new/train_conrft/real30_mew_filter.pkl

NOW_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
CKPT_ROOT=/media/zhourui/Twin-RL/conrft/examples/experiments/task4_insert_circle/checkpoints/real30_new/q_loss
CHECKPOINT_PATH="${CKPT_ROOT}/${NOW_TIME}_delta_ee_full/"

python -m examples.train_offline "$@" \
    --exp_name=$EXP_NAME \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --q_weight=0.1 \
    --bc_weight=1.0 \
    --demo_path=$DEMO_PATH \
    --pretrain_steps=10000 \
    --debug=False \
    --learner \
    --port_number=$PORT_NUMBER \
    --broadcast_port=$BROADCAST_PORT



