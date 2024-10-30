# Multi class activity classification in videos using Motion History Image generation
### Introduction
Human action recognition has been a topic of interest across multiple fields ranging from security to entertainment systems. Tracking the motion and identifying the action being performed on a real time basis is necessary for critical security systems. In entertainment, especially gaming, the need for immediate responses for actions and gestures are paramount for the success of that system. We show that Motion History image has been a well established framework to capture the temporal and activity information in multi dimensional detail enabling various usecases including classification. We utilize MHI to produce sample data to train a classifier and demonstrate its effectiveness for action classification across six different activities in a single multi-action video. We analyze the classifier performance and cases where MHI struggles to capture the true activity and discuss mechanisms and future work to overcome those limitations.

### Citation
If you find this project useful in your research or work, please consider citing it:
```
@article{gopal2024multiclass,
  title={Multi class activity classification in videos using Motion History Image generation},
  author={Gopal, Senthilkumar},
  journal={arXiv preprint arXiv:2410.09902},
  year={2024}
}
```
## Project Checklist 
* `mhi.py` - Primary source file
* [Multi Action Video with Prediction Labels](https://youtu.be/2q4zOnSYKSA)

## Installation Details
1. Use the conda env setup using `cv_proj.yml`
2. `matplotlib=3.0.3` needs to be installed in the environment.

## Setup Details
1. All the dataset files are already added to the project folder. [Reference](http://www.nada.kth.se/cvap/actions/)
2. Run the file *mhi.py* to perform all the steps
3. `LOAD_TRAIN_DATA` - Use this flag to switch on/off loading the dataset from the `/datasets` 

## Execution steps
Following are the steps that are executed as part of the *file: mhi.py*

1. `generate_report_images` - Generates the binary images, MHI, MEI for the report/presentation
2. `execute_classifier_based_recognition('KNN')` - Executes training, validation, testing for KNN classifier
3. `execute_classifier_based_recognition('MLP')` - Executes training, validation, testing for MLP classifier 
4. `predict_multi_action` - Uses the MLP classifier to predict the various actions in the sample video
5. `execute_jogging_only_prediction(mlp_recog)` - Executes the MLP classifier for a sample where the prediction was incorrect for jogging.
6. `execute_incorrect_multi_action_prediction` - Executes the MLP classifier for a sample where the prediction was incorrect for jogging.

## Runtime Expectations
The expected runtime for *mhi.py*
1. Execution with saved dataset - Approx 4 Minutes
2. Execution with Full training - Approx 3 Minutes

## Project Tree Descriptions

### Input Folders
1. `/input_videos` -  Contains the `_d1` type of video files useful for training.
2. `/my_videos` - Videos that were captured for testing the ML classifier
3. `/input_files` - Text files provided/created for storing frame references
4. `/datasets` - generated datasets of MHI/MEI and labels for easier loading and training

### Output Folders
1. `/output` - Primary output folder containing the confusion matrix, sample video with labels and sample frames 
2. `/report` - Images generated for the report
3. `/report/binary` - Binary images used in the report for different videos
4. `/report/mhi` - MHI images used in the report for different videos
5. `/report/mei` - MEI images used in the report for different videos

### Other files and their usage
1. `utility.py` - Utility functions for one of testing and generation

### Resources / Reference Links
1. [Image_moment](https://en.wikipedia.org/wiki/Image_moment)
2. [KNN Learning](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_knn/py_knn_understanding/py_knn_understanding.html)
3. [Scikit KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
4. [KNN Good Intro](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/)
5. [Dataset Train vs. Test](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)
6. [Hu Moments](https://www.pyimagesearch.com/2014/10/27/opencv-shape-descriptor-hu-moments-example/)
7. [MHI](http://web.cse.ohio-state.edu/~davis.1719/CVL/Research/MHI/mhi.html)
8. [MLP Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier)
