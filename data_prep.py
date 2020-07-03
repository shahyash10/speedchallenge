import os
import cv2
import numpy as np


class DataPreProcessing:
    def __init__(self):
        try:
            os.mkdir("data/train_images")
            os.mkdir("data/test_images")
        except:
            pass

    def optical_flow(self, im_c, im_n):
        gray_c = cv2.cvtColor(im_c, cv2.COLOR_BGR2GRAY)
        gray_n = cv2.cvtColor(im_n, cv2.COLOR_BGR2GRAY)
        rgb_flow = np.zeros_like(im_c)
        flow = cv2.calcOpticalFlowFarneback(
            gray_c, gray_n, None, 0.5, 1, 15, 2, 5, 1.3, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        rgb_flow[:, :, 0] = ang * (180 / np.pi / 2)
        rgb_flow[:, :, 2] = (mag * 15).astype(int)
        rgb_flow[:, :, 1] = 255

        return rgb_flow

    def train_preprocess(self, train_vid="data/train.mp4"):

        cap = cv2.VideoCapture(train_vid)
        _, prev = cap.read()
        i = 0
        while True:
            ret, curr = cap.read()
            if not ret:  # EOF
                break
            frame = self.optical_flow(curr, prev)
            # resize the frame
            frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
            cv2.imwrite("data/train_images/" + str(i) + ".png", frame)
            i += 1
            prev = curr

    def test_preprocess(self, test_vid="data/test.mp4"):

        cap = cv2.VideoCapture(test_vid)
        _, prev = cap.read()
        i = 0
        while True:
            ret, curr = cap.read()
            if not ret:  # EOF
                break
            frame = self.optical_flow(curr, prev)
            # resize the frame
            frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
            cv2.imwrite("data/test_images/" + str(i) + ".png", frame)
            i += 1
            prev = curr


data_prep = DataPreProcessing()
data_prep.train_preprocess()
data_prep.test_preprocess()
