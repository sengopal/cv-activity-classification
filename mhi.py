# Activity Classification using MHI

import os
import re
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

# Controls for Training and Loading saved data
SAVE_TRAIN_DATA = False
LOAD_TRAIN_DATA = True

IMG_DIR = "input_files"
VID_DIR = "input_videos"
MY_VID_DIR = "my_videos"
OUT_DIR = "output"

# ref: http://www.nada.kth.se/cvap/actions/
ACTIONS = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']

# Global constants
THETA = 15
TRAIN_TAU = 300

SUFFIX_LIST = ["_d1"]


# Ref: https://www.kaggle.com/ahmethamzaemra/mlpclassifier-example and Piazza Suggestion
# Ref: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
# TODO: Refactor to OO with interface and methods
class MHIClassifer():
    def __init__(self, type='KNN'):
        self.type = type
        if self.type == 'MLP':
            #  solver='lbfgs' - . For small datasets, however, ‘lbfgs’ can converge faster and perform better.
            self.classifer = MLPClassifier(activation='relu', hidden_layer_sizes=(200), max_iter=1000, solver='lbfgs', alpha=0.001, random_state=21, tol=0.000000001)
        else:
            # Default
            self.classifier = cv2.ml.KNearest_create()

    def train(self, training_data, y_labels):
        if self.type == 'KNN':
            self.classifier.train(training_data, cv2.ml.ROW_SAMPLE, y_labels)
        elif self.type == 'MLP':
            self.classifer.fit(training_data, y_labels)

    def predict(self, prediction_data):
        if self.type == 'KNN':
            retval, res, neighbours, dist = self.classifier.findNearest(prediction_data, 1)
            # print("results: ", str(res))
            # print("neighbours: ", str(neighbours))
            # print('shape: ', res.shape)
            return retval
        elif self.type == 'MLP':
            return self.classifer.predict(prediction_data)[0]


class MHIRecognition():
    def __init__(self, classifier=MHIClassifer('MLP')):
        self.classifier = classifier
        self.scale_mom_list_mei = []
        self.scale_mom_list_mhi = []
        self.y_labels = []

    def build_training_data(self, file_name, action_label, action_seq):
        """
            action_label --> One of the labels defined in ACTIONS
            action_seq --> List of tuples of frames to be used from 00sequences.txt provided in the data
        """
        mei_list, mhi_list = self.compute_mei_mhi(action_seq, file_name, tau=TRAIN_TAU, theta=THETA)
        # Use Hu moments to characterize the MHI. Use both the unscaled and scaled central moments upq and vpq

        for mhi in mhi_list:
            cen_mom, scale_mom = self.calculate_humoments(mhi)
            self.scale_mom_list_mhi.append(scale_mom)

            # setting index since the classifier complains that this has to be a numpy array
            self.y_labels.append(ACTIONS.index(action_label))

        for mei in mei_list:
            cen_mom, scale_mom = self.calculate_humoments(mei)
            self.scale_mom_list_mei.append(scale_mom)

    def compute_mei_mhi(self, action_seq, file_name, tau, theta=THETA, mhi_save_name=None):
        mhi_list = []
        mei_list = []
        for frame in action_seq:
            # print('Frame_Seq: ', frame)
            # Create a method to obtain a binary motion signal to analyze over time.
            binary_image = self.create_binary_img(frame, file_name, theta=theta)

            # Given the sequence Bt we can construct the MHIs. The Motion History Image M at time t
            mhi = self.compute_mhi(binary_image, tau=tau)
            mhi_list.append(mhi)

            # Computing MEI as well to be added as part of the training data to provide more segregation
            mei = (255 * mhi > 0).astype(np.uint8)

            if mhi_save_name is not None:
                cv2.imwrite("output/report/mhi/{}_mhi_{}.png".format(mhi_save_name, frame), mhi)
                mei_copy = cv2.normalize(mei, mei.copy(), 0.0, 255.0, cv2.NORM_MINMAX)
                cv2.imwrite("output/report/mei/{}_mei_{}.png".format(mhi_save_name, frame), mei_copy)

            mei_list.append(mei)

        return mei_list, mhi_list

    def train_classifier(self):
        self.y_labels = np.array(self.y_labels).astype(np.int)
        # print('self.y_labels: ', self.y_labels)

        self.scale_mom_list_mhi = np.array(self.scale_mom_list_mhi).astype(np.float32)
        # print('self.scale_mom_list: ', self.scale_mom_list)
        self.scale_mom_list_mei = np.array(self.scale_mom_list_mei).astype(np.float32)

        # since we are using Row sample in the classifier, hstack places the arrays next to each other to form a large row
        training_data = np.hstack((self.scale_mom_list_mhi, self.scale_mom_list_mei))

        self.classifier.train(training_data, self.y_labels)

    def predict_action(self, file_name, action_seq, tau=TRAIN_TAU, mhi_save_name=None):
        """
        :param file_name: test video file_name
        :param action_seq: frame seq to be captured for prediction
        :return: action label of prediction
        """

        mei_list, mhi_list = self.compute_mei_mhi(action_seq, file_name, tau, theta=THETA, mhi_save_name=mhi_save_name)

        # Use Hu moments to characterize the MHI. Use both the unscaled and scaled central moments upq and vpq
        scale_mom_list_mhi = []
        scale_mom_list_mei = []
        for mhi in mhi_list:
            cen_mom, scale_mom = self.calculate_humoments(mhi)
            scale_mom_list_mhi.append(scale_mom)

        for mei in mei_list:
            cen_mom, scale_mom = self.calculate_humoments(mei)
            scale_mom_list_mei.append(scale_mom)

        scale_mom_list_mhi = np.array(scale_mom_list_mhi).astype(np.float32)
        scale_mom_list_mei = np.array(scale_mom_list_mei).astype(np.float32)

        prediction_data = np.hstack((scale_mom_list_mhi, scale_mom_list_mei))

        retval = self.classifier.predict(prediction_data)
        return retval

    def create_binary_img(self, frames, file_name, theta=15, bin_save_name=None):
        # Extract the person from the background and use the area the person occupies as a signal.
        # To create an MHI you first need to compute the frame difference sequence

        frames_for_bin_image = {10, 20, 30}

        # Need to extract the sequence frames using Video reader
        # Code copied from PS3/experiment.py
        image_gen = video_frame_generator(file_name)

        curr_image = image_gen.__next__()
        frame_num = 1
        binary_image_list = []
        # print('frames: ',frames)
        while curr_image is not None:
            # print("Processing frame {}".format(frame_num))

            next_image = image_gen.__next__()

            if (frames[0] <= frame_num <= frames[1]) and (next_image is not None):
                # Frame in sequence
                curr_image_processed = self.process_image(curr_image.copy())

                # to compute the frame difference sequence
                next_image_processed = self.process_image(next_image.copy())
                # show_img(next_image_processed)

                binary_image = np.zeros(shape=curr_image_processed.shape, dtype=np.uint8)
                # binary_image[np.abs(next_image_processed - curr_image_processed) >= theta] = 1
                binary_image = np.abs(cv2.subtract(next_image_processed, curr_image_processed)) >= theta
                # TypeError: Expected cv::UMat for argument 'src'. cv2.morphologyEx expects the image to be 1 or 0
                binary_image = binary_image.astype(np.uint8)
                # print('Binary Check:', np.any(binary_image))

                # You might want to “clean up” the binary images using a morphological OPEN operator to remove noise.
                # Ref: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
                binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

                if np.sum(binary_image) > 0:
                    binary_image_list.append(binary_image)

                if bin_save_name is not None and frame_num in frames_for_bin_image:
                    binary_image_copy = cv2.normalize(binary_image, binary_image.copy(), 0.0, 255.0, cv2.NORM_MINMAX)
                    cv2.imwrite("output/report/binary/{}_{}_binimg.png".format(bin_save_name, frame_num), binary_image_copy)
                    cv2.imwrite("output/report/binary/{}_{}_frameimg.png".format(bin_save_name, frame_num), curr_image)

            # Setting the curr_image as next_image for the process to continue
            curr_image = next_image
            frame_num += 1

        return binary_image_list

    def process_image(self, input_image):
        image_process = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        image_process = cv2.GaussianBlur(image_process, ksize=(3, 3), sigmaX=0.0, sigmaY=0.0)
        # image_process = cv2.medianBlur(image_process, 1)
        # image_process = cv2.bilateralFilter(image_process, 9, 75, 75)
        return image_process

    def compute_mhi(self, binary_image_list, tau):
        # Building the MHI using the structure provided in the lectures
        #
        mhi = np.zeros(binary_image_list[0].shape, dtype=np.float)

        for binary_image in binary_image_list:
            # All motion instances are set to Tau
            mhi[binary_image > 0] = tau
            # as time passes the value becomes greyer until it reaches 0
            mhi[mhi > 0] -= 1
            mhi[mhi < 0] = 0

        mhi = mhi.astype(np.uint8)
        mhi = cv2.normalize(mhi, mhi, 0.0, 255.0, cv2.NORM_MINMAX)
        # mhi = cv2.medianBlur(mhi, 5)
        # show_img(mhi)
        # print(mhi)
        return mhi

    def calculate_humoments(self, mhi):
        # using reference: https://www.pyimagesearch.com/2014/10/27/opencv-shape-descriptor-hu-moments-example/
        # Using all the moments for <features> E {20, 11, 02, 30, 21, 12, 03, 22

        features = [(2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), (2, 2)]

        # Calculating the moment M00
        # Area (for binary images) or sum of grey level (for greytone images): {\displaystyle M_{00}} {\displaystyle M_{00}}
        # Reference - https://en.wikipedia.org/wiki/Image_moment
        # M(00) = Sum of non-zero pixels → Area of the non-zero portion of the image - Lecture on Image moments
        M00 = np.sum(mhi)

        # Using the negative value to restructure the unknown value as a valid ndarray

        # ref: https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.reshape.html - Specifically the -1 section
        # https://stackoverflow.com/questions/39071630/numpy-reshape-confusion-with-negative-shape-values
        x = np.arange(mhi.shape[1]).reshape(-1, 1)
        y = np.arange(mhi.shape[0]).reshape(-1, 1).T

        # print(x.shape, y.shape)
        # Calculating Mij as per the formula provided  - Regular np.Matrix multiplication does not work due to their shapes, hence
        # rearranging to get the correct order for dot product
        M10 = np.sum(np.dot(mhi, x))
        M01 = np.sum(np.dot(y, mhi))

        # finding the centroids / average x and y
        x_mean = M10 / M00
        y_mean = M01 / M00

        # print('centroids: {} {}'.format(x_mean, y_mean))
        cent_moments = []
        scale_inv_moments = []

        # # Reference - https://en.wikipedia.org/wiki/Image_moment
        mu_00 = M00
        for i, (p, q) in enumerate(features):
            x_diff = x - x_mean
            y_diff = y - y_mean
            # print('{} {}', x_diff.shape, y_diff.shape)

            temp = np.dot((x_diff ** p), (y_diff ** q))
            mu_pq = np.sum(np.dot(temp, mhi))
            cent_moments.append(mu_pq)

            v_pq = cent_moments[i] / mu_00 ** (1 + (p + q) / 2)
            scale_inv_moments.append(v_pq)

        return cent_moments, scale_inv_moments

    def save_training_data(self):
        np.save('datasets/scale_mom_list_mhi_' + self.classifier.type + '.npy', self.scale_mom_list_mhi)
        np.save('datasets/scale_mom_list_mei_' + self.classifier.type + '.npy', self.scale_mom_list_mei)
        np.save('datasets/y_labels_' + self.classifier.type + '.npy', self.y_labels)

    def load_saved_training_data(self):
        self.scale_mom_list_mhi = np.load('datasets/scale_mom_list_mhi_' + self.classifier.type + '.npy')
        self.scale_mom_list_mei = np.load('datasets/scale_mom_list_mei_' + self.classifier.type + '.npy')
        self.y_labels = np.load('datasets/y_labels_' + self.classifier.type + '.npy')


## VIDEO Utility - Copied from PS3
def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None


# Utility to display the provided image
def show_img(res):
    cv2.imshow('res', res)
    cv2.waitKey(0)


## FILE PARSING Utilities - START
def get_action(file_name):
    regex = re.compile("_")
    a = regex.split(file_name)
    return str.strip(a[1])


def extract_seq(s):
    a = re.compile(",").split(s)
    seqs = []
    for seq in a:
        # print(seq)
        b = re.compile("-").split(seq)
        seqs.append((int(b[0]), int(b[1])))
    return seqs


def parse_data(file_name):
    # Ref: https://stackabuse.com/read-a-file-line-by-line-in-python/
    filepath = os.path.join(IMG_DIR, file_name)
    regex = re.compile("\t")
    # data = {'person01_handwaving_d1' : {action:'handwaving', seq: [(1,100)]}
    data = {}
    with open(filepath) as fp:
        for line in fp:
            # print(line)
            array = regex.split(line)
            file_name = str.strip(array[0])
            action = get_action(file_name)
            dict_value = {'action': action, 'sequence': extract_seq(array[2])}
            data[file_name] = dict_value
    # print(data)
    return data


def get_file_identifier(suffix, i, action):
    file_id = "person0" if i < 10 else "person"
    file_id = file_id + str(i) + "_" + action + suffix
    return file_id


## FILE PARSING Utilities - END

# Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
# Ref: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
# Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, file_name, title_prefix=None):
    # cm = (cm * 100 / cm.sum()).astype(np.uint) / 100.0
    # cm = ((cm).astype(np.float32) / cm.sum())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title_prefix + ' - Normalized Confusion Matrix', fontweight="bold")
    plt.ylabel('Actual Label')
    plt.xlabel('Prediction Label')
    tm = np.arange(len(ACTIONS))
    plt.xticks(tm, ACTIONS, rotation='45')
    plt.yticks(tm, ACTIONS)
    plt.colorbar()
    plt.tight_layout()

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.savefig(os.path.join(OUT_DIR, file_name))


def execute_training(recog, training_range):
    if LOAD_TRAIN_DATA:
        print('Loading training data')
        recog.load_saved_training_data()
    else:
        for suffix in SUFFIX_LIST:
            for i in training_range:
                for action in ACTIONS:
                    file_identifier = get_file_identifier(suffix, i, action)

                    if file_identifier in sequence_data:
                        # print('Processing: ', file_identifier)
                        action_data = sequence_data[file_identifier]
                        # print('action_data:', action_data)
                        video_file_name = os.path.join(VID_DIR, action, file_identifier + "_uncomp.avi")
                        try:
                            recog.build_training_data(file_name=video_file_name, action_label=action, action_seq=action_data['sequence'])
                        except Exception as inst:
                            print('ERROR', inst)
                            continue
                    else:
                        print('Skipping: ', file_identifier)

        if SAVE_TRAIN_DATA:
            recog.save_training_data()

    recog.train_classifier()


def execute_prediction(recog, data_range, conf_mat_file_name):
    cnfmt = np.zeros((6, 6))
    results = []
    for suffix in SUFFIX_LIST:
        for i in data_range:
            for action_label in ACTIONS:
                file_identifier = get_file_identifier(suffix, i, action_label)

                if file_identifier in sequence_data:
                    # print('Processing: ', file_identifier)
                    test_file_name = os.path.join(VID_DIR, action_label, file_identifier + "_uncomp.avi")
                    try:
                        result = recog.predict_action(test_file_name, action_seq=sequence_data[file_identifier]['sequence'])
                        # print(' {}: {} {} {}'.format(file_identifier, ACTIONS.index(action_label), result, (ACTIONS.index(action_label) == result)))
                        results.append((ACTIONS.index(action_label) == result))
                    except Exception as inst:
                        print('ERROR', inst)
                        result = 0.0
                    yte = ACTIONS.index(action_label)
                    cnfmt[yte, int(result)] += 1
                else:
                    print('Skipping: ', file_identifier)

            # print(get_file_identifier('', i, '_RESULT') + ': ', sum(results))

    # Ref: https://stackoverflow.com/questions/12765833/counting-the-number-of-true-booleans-in-a-python-list
    accuracy = (sum(results) * 100) / (len(ACTIONS) * len(data_range))
    # https://stackoverflow.com/questions/28343745/how-do-i-print-a-sign-using-string-formatting
    print('{} - Accuracy {:.2f}%%'.format(recog.classifier.type, accuracy))

    plot_confusion_matrix(cnfmt, conf_mat_file_name, title_prefix=recog.classifier.type)


# Code from PS3/experiment.py - Utilities for creating a video file
def mp4_video_writer(filename, frame_size, fps=20):
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # To fix the error OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 12 and format 'avi / AVI (Audio Video Interleaved)'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


# Copied from ps2/experiment.py
def write_text(img, text):
    # Using the suggested structure in piazza and slack and refering to https://gist.github.com/aplz/fd34707deffb208f367808aade7e5d5c
    font_scale = 1
    font = cv2.FONT_HERSHEY_DUPLEX

    # set the rectangle background to white
    rectangle_bgr = (255, 255, 255)
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=2)[0]
    # set the text start position
    (text_offset_x, text_offset_y) = (200, 100)
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x - 10, text_offset_y - 30), (text_offset_x + text_width + 10, text_offset_y + text_height - 5))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, thickness=cv2.FILLED)
    cv2.rectangle(img, box_coords[0], box_coords[1], color=None, thickness=1)
    cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=2)


def generate_output_video(file_identifier, predict_result):
    file_name = os.path.join(MY_VID_DIR, file_identifier + "_uncomp.avi")
    image_gen = video_frame_generator(file_name)

    frame_nums_report = [190, 400, 575, 925, 1215, 1370]

    curr_image = image_gen.__next__()
    frame_num = 1
    h, w, d = curr_image.shape

    out_path = "output/" + file_identifier + "_prediction.avi"
    video_out = mp4_video_writer(out_path, (w, h), fps=25)

    while curr_image is not None:
        # print("Processing frame {}".format(frame_num))

        text = 'Waiting for prediction ....'
        if frame_num in predict_result:
            # print('frame_num:', frame_num, predict_result[frame_num])
            text = 'Predicted Action: ' + ACTIONS[predict_result[frame_num]].upper()

        write_text(curr_image, text)
        # show_img(curr_image)
        video_out.write(curr_image)

        if frame_num in frame_nums_report:
            cv2.imwrite("output/multi_action_predict_{}.png".format(frame_num), curr_image)

        curr_image = image_gen.__next__()
        frame_num += 1

    video_out.release()


def flatten_result(predict_result):
    """
    Converts the dict model of range to a flatten frame_num indexed value to be easily interlaced during
    video writing
    input: {(1560, 1710): 1, (1190, 1250): 2, (180, 275): 3, (1304, 1436): 1, (730, 800): 0, (555, 690): 4, (835, 1025): 0, (370, 500): 5}
    output: {1: 1, 2: 1, 3: 1, 8: 2, 9: 2}

    :param predict_result:
    :return:
    """
    # Ref: https://docs.quantifiedcode.com/python-anti-patterns/readability/not_using_items_to_iterate_over_a_dictionary.html
    flat_result = {}
    for key, val in predict_result.items():
        # print(key, ":", val)
        for i in range(key[0], key[1] + 1, 1):
            # print(i)
            flat_result[i] = val

    # print(flat_result)
    return flat_result


def predict_multi_action(recog, sample_seq_data):
    file_identifier = 'sample08_multi_action'

    if file_identifier in sample_seq_data:
        action_seq_list = sample_seq_data[file_identifier]['sequence']
        test_file_name = os.path.join(MY_VID_DIR, file_identifier + ".avi")
        predict_result = {}
        for action_seq in action_seq_list:
            # print('Processing: ', action_seq)
            try:
                result = recog.predict_action(test_file_name, action_seq=[action_seq], tau=350, mhi_save_name=file_identifier)
                predict_result[action_seq] = result
            except Exception as inst:
                print('ERROR', inst)

        # Expected result - {(1560, 1710): 1, (1190, 1250): 2, (180, 275): 3, (1304, 1436): 1, (730, 800): 0, (555, 690): 4, (835, 1025): 0, (370, 500): 5}
        print(predict_result)

        generate_output_video(file_identifier, flatten_result(predict_result))

    else:
        print('Skipping: ', file_identifier)


def generate_report_images(sequence_data):
    for action in ACTIONS:
        recog_classifier = MHIRecognition()

        file_identifier = get_file_identifier("_d1", 15, action)
        action_data = sequence_data[file_identifier]
        video_file_name = os.path.join(VID_DIR, action, file_identifier + "_uncomp.avi")

        # # Generating individual binary image frames for only one action sequence
        recog_classifier.create_binary_img((1, 50), video_file_name, theta=5, bin_save_name=file_identifier)

        file_identifier = get_file_identifier("_d1", 20, action)
        action_data = sequence_data[file_identifier]
        video_file_name = os.path.join(VID_DIR, action, file_identifier + "_uncomp.avi")
        action_seq = action_data['sequence']

        # using small values for TAU for generating a good representational report image
        recog_classifier.compute_mei_mhi(action_seq, video_file_name, tau=30, theta=5, mhi_save_name=file_identifier)


def get_dataset():
    # Training/Validation/Testing - Split
    training_range = [1, 2, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 23, 24]
    validation_range = [4, 8, 20, 25]
    test_range = [3, 7, 19, 22]

    assert (len([x for x in validation_range if x in training_range]) == 0)
    assert (len([x for x in test_range if x in training_range]) == 0)

    return training_range, validation_range, test_range


def execute_classifier_based_recognition(name):
    print('********* {} based Recognition *********'.format(name))

    recog = MHIRecognition(classifier=MHIClassifer(type=name))

    training_range, validation_range, test_range = get_dataset()

    print('********* TRAINING *********')
    execute_training(recog, training_range)

    # Validation
    print('********* VALIDATION *********')
    execute_prediction(recog, validation_range, 'confusion_matrix_validation_{}.png'.format(name.lower()))

    print('********* TEST *********')
    execute_prediction(recog, test_range, 'confusion_matrix_test_{}.png'.format(name.lower()))

    print('********* {} based Recognition Completed*********'.format(name))
    return recog


def execute_jogging_only_prediction(mlp_recog):
    file_list = ['sample01_jogging', 'sample02_jogging']

    for file_identifier in file_list:
        if file_identifier in sample_seq_data:
            # print('Processing: ', file_identifier)
            action_seq_list = sample_seq_data[file_identifier]['sequence']
            test_file_name = os.path.join(MY_VID_DIR, file_identifier + ".avi")
            for action_seq in action_seq_list:
                result = mlp_recog.predict_action(test_file_name, action_seq=[action_seq], tau=350)
                print(' Prediction Accuracy : Actual - {} Predicted {}'.format('jogging', ACTIONS[result]))
        else:
            print('Skipping: ', file_identifier)


def execute_incorrect_multi_action_prediction(mlp_recog):
    actual_action_labels = [3, 4, 5, 0, 0, 1, 1, 2, 2]
    file_identifier = 'sample03_multi_action'

    if file_identifier in sample_seq_data:
        # print('Processing: ', file_identifier)
        action_seq_list = sample_seq_data[file_identifier]['sequence']
        test_file_name = os.path.join(MY_VID_DIR, file_identifier + ".avi")
        for i, action_seq in enumerate(action_seq_list):
            result = mlp_recog.predict_action(test_file_name, action_seq=[action_seq], tau=100, mhi_save_name=file_identifier)
            print(' Prediction Accuracy : Actual - {} Predicted {}'.format(ACTIONS[actual_action_labels[i]], ACTIONS[result]))
    else:
        print('Skipping: ', file_identifier)


if __name__ == '__main__':
    start_time = time.time()
    print('********* Beginning MHI *********', start_time)
    # Action sequence per file is defined separately from 00sequences.txt
    sequence_data = parse_data('00sequences_strip.txt')

    # Generate MHI, MEI images for varying actions and different Taus
    generate_report_images(sequence_data)

    # Executing for KNN
    execute_classifier_based_recognition('KNN')

    # Executing using MLP and storing the classifier for multi-action videos recognition
    mlp_recog = execute_classifier_based_recognition('MLP')

    # Parsing data for my generated videos
    sample_seq_data = parse_data('my_videos_sequence.txt')

    print('********* MY SAMPLE - MULTI ACTION *********')
    predict_multi_action(mlp_recog, sample_seq_data)

    # Report Specific - For generating incorrect recog results for ANALYSIS

    # Incorrect Results - Due to mis-structured Tau
    print('********* INCORRECT RECOG. SAMPLE - JOGGING ONLY *********')
    execute_jogging_only_prediction(mlp_recog)

    # Incorrect Results - Due to shadows and artifacts
    print('********* INCORRECT RECOG. SAMPLE - MULTI ACTIION *********')
    execute_incorrect_multi_action_prediction(mlp_recog)

    print("--- %.2f Minutes ---" % ((time.time() - start_time) / 60.0))
