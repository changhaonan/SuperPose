# set path
export SUPERPOSE_PATH="/home/robot-learning/Projects/SuperPose"

# init r2d2
# conda activate r2d2

# change workspace
cd ${SUPERPOSE_PATH}/r2d2

# run feature server
python ${SUPERPOSE_PATH}/r2d2/run_server.py --port=9090 --model=models/r2d2_WASF_N16.pt --top-k=500