# SuperPose (SuperTrack)

The idea of this project is to combine pure RGB-based pose estimator OnePose with pure geoemetry based method ICG as well as the robust 2D Tracker.

The system is divided into two module:

1. Scanning module: The input is a set of RGBD sequence or a RGB-CAD model. And the output is a Sparse Feature bindled with a Sparse View Model. We also provide visualizer for this phase.

2. The tracking model is using the Model comes from the previous step. RTS is used to provide a preliminary mask. And OnePose is used to generate a starting Pose. ICG to used update and conduct the track.

The demo version is going to connected using python. If everything is good, will consider to connect using C++ & CUDA to accelerate more.

## Plan

Feb.1st to Feb.4th
- [x] Align the MeshModel with the SparseModel.
- [x] Check How OnePose is working on KF frames. Is it still able to provide proper Estimation. (It is performing OK...)
- [ ] Finish the Scan phase. Generate model from Video Sequence, existing CAD Model or NeRF model.

Feb.5th to Feb.11th

- [x] Test ICG algorithm.
- [x] Test ICG in real-world. Combine OnePose with ICG.
    - [x] Add a one-pose detector. Detector integration.
    - [x] The current problem is that how can we combine them.
    - [x] Create a video recorder to record the video.
    - [ ] Build a hybrid pipeline for video.
        - [ ] The current idea is to add a feature upon it.
        - [ ] An easy way to think is that we can record some keyinformation and read them when running icg.
- [ ] Add build tools from NeRF.
- [x] Add Network-based detector.
- [ ] Download a prepapre for Benchmark dataset & Prepare Tools for them. YCB-video, BOI, BOP challenge.
- [ ] Run OnePose and ICG seperately on benchmark.
- [ ] The current idea is to do a feature-based pure CPU method. ICG plus.
- [x] Change SuperTrack to zmq socket.
- [x] Run BundleTrack with r2d2.
- [x] Replace feature matching with bundletrack method.
    - [x] Get feature generation finished.
    - [x] Create a feature Viewer. [Wed]
    - [x] Create Sparse feature view of the object. [Wed]
        - [x] Understand what model should generate.
        - [x] Render different aspects of the CAD model. [Thurs]
        - [x] Augment the normal shader with RGB shader. [Thurs]
        - [x] Extract Keypoint feature. [Thurs]
        - [x] Save Keypoint feature into the model.
        - [x] Create a sparse feature object. (Each view should have a feature.)
        - [x] Build connection.
    - [x] Build the feature matching. [Sun]
        - [x] Build the feature matching pipeline and compair the result.
        - [x] Merge the system with pfb.
            - [x] Make the image sparse model very sparse. (Just contain multiple images with poses.)
            - [x] I can try to improve the closestview and compute the rot-angle.
    - [ ] Combination
        - [ ] Directly do PNP.
            - [x] Build PNP problem and solve it.
            - [x] Render it into feature viewer.
            - [x] The optimization and PNP seems can not work together?
            - [x] Put it into the refiner step.
            - [x] I should try to put it at the refiner step.
            - [x] Doesn't seems to be working.
                - [x] Fix the PNP bug.
            - [x] Integrate OnePose into the system.
            - [ ] SFM from CAD. This a small module. But we need to integrate it to make be able to run in test scene. Finish the demo this week.
        - [ ] Integrate into loss
            - [ ] Compute Jacobian. 
            - [ ] Test the system run. 
            - [ ] Further test the system.

Feb.17th
- [ ] Finish the matching process.
- [ ] Debug ICG on YCB-V dataset to avoid the .txt requirement.

Feb.16th-Feb.18th
- [ ] Add the feature lost.

Feb.19th to Feb.25th
- [ ] Run result on the selected dataset.
- [ ] Improve the method with adaptive structure.
- [ ] Write the paper.
- [ ] Polish the method

Feb.26th to March.1st
- [ ] Write the paper

# Debug

# Declare

Our code based is based on the implementation of OnePose, ICG and Pytracking.