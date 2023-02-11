# set path
export SUPERPOSE_PATH="/home/robot-learning/Projects/SuperPose"
export FEATURE_TYPR="r2d2"
export EXERIMENT_NAME="Scene"

# switch on feature type
if [ ${FEATURE_TYPR} = "r2d2" ]; then
    # change workspace
    cd ${SUPERPOSE_PATH}/r2d2
    # run feature server
    python ${SUPERPOSE_PATH}/r2d2/run_r2d2_server.py --port=9090 --model=models/r2d2_WASF_N16.pt --top-k=500
elif [ ${FEATURE_TYPR} = "orb" ]; then
     # change workspace
    cd ${SUPERPOSE_PATH}/cv_server
    # run feature server
    python ${SUPERPOSE_PATH}/cv_server/run_orb_server.py --port=9090 --top-k=500
    # python ${SUPERPOSE_PATH}/cv_server/run_orb_server.py --port=9090 --top-k=500 --save-dir=${SUPERPOSE_PATH}/debug/${EXERIMENT_NAME}
fi