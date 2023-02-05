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
- [ ] Test ICG in real-world. Combine OnePose with ICG.
    - [ ] Add a one-pose detector. Detector integration.
    - [ ] The current problem is that how can we combine them.
    - [x] Create a video recorder to record the video.
    - [ ] Build a hybrid pipeline for video.
        - [ ] The current idea is to add a feature upon it.
        - [ ] An easy way to think is that we can record some keyinformation and read them when running icg.
- [ ] Download a prepapre for Benchmark dataset & Prepare Tools for them. YCB-video, BOI, BOP challenge.
- [ ] Run OnePose and ICG seperately on benchmark.

Feb.12th to Feb.18th
- [ ] Combine ICG, OnePose and Two tracker.
- [ ] Test them on benchmark.
- [ ] Collect our own dataset.

Feb.19th to Feb.25th
- [ ] Run on the DIY dataset.
- [ ] Write the paper.
- [ ] Polish the method

Feb.26th to March.1st
- [ ] Write the paper

# Debug

# Declare

Our code based is based on the implementation of OnePose, ICG and Pytracking.