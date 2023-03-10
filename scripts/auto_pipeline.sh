export ROOT_PATH=/home/robot-learning/Projects/SuperPose
export DATA_SET=cracker_box
export DATA_ID=4
export ICG_PATH=$ROOT_PATH/ICG/examples/generator_example

# Run one pose prior first
echo "Running one pose prior"
python ${ROOT_PATH}/one_pose_prior.py +experiment=one_pose_prior.yaml input.eval_dirs=[${ROOT_PATH}/data/${DATA_SET}/${DATA_SET}_${DATA_ID}] video_mode=image

# Run the star to icg
echo "Running star to icg"
python ${ROOT_PATH}/utils/star_to_icg.py --tracker_name ${DATA_SET} --model_dir ${ROOT_PATH}/data/${DATA_SET}/${DATA_SET}_1 --eval_dir ${ROOT_PATH}/data/${DATA_SET}/${DATA_SET}_${DATA_ID} --icg_dir ${ICG_PATH} --enable_depth

# Run icg
echo "Running icg"
${ROOT_PATH}/build/examples/run_generated_tracker ${ICG_PATH}/config.yaml

echo "Finished!"