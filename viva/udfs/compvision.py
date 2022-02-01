import os
import sys
import cv2
import pickle
import torch
import numpy as np
import pandas as pd
from typing import Iterator
from time import sleep, perf_counter
from viva.utils.config import ConfigManager
config = ConfigManager()

from viva.sparkmodels import InferenceResults, RawFrameData
from viva.udfs.inference import batch, batch_size

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (
    StructField, ArrayType, StructType, BinaryType, IntegerType, StringType,
    FloatType
)

h = config.get_value('ingest', 'height')
w = config.get_value('ingest', 'width')

@pandas_udf('float')
def image_similarity(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # Initialize detector
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    detector = cv2.AKAZE_create()

    # Dimensions to resize to
    resize_dims = (200, 200)

    # Load image to compare against
    # TODO: IMPORTANT TO KNOW!!!
    # Currently assumes this image is local, but it could be fetched from
    # remote storage, or loaded as is done with all frames (i.e., as a bytearray
    # read from a DataFrame)
    target_img = cv2.imread('./data/similarity_img.png')

    # Resize to fixed dimensions
    target_img = cv2.resize(target_img, resize_dims)

    # Pass target_img through detector
    _, target_des = detector.detectAndCompute(target_img, None)

    for content_series in content_series_iter:
        # Reverse content series
        all_framebytes = content_series[0]
        all_width = content_series[1]
        all_height = content_series[2]
        content_series_rev = tuple([(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)])

        all_res = []
        for c in content_series_rev:
            inp_bytes = c[0]
            inp_width = c[1]
            inp_height = c[2]
            comparing_img = np.array(bytearray(inp_bytes)).reshape(inp_height, inp_width, 3)

            # Resize to fixed dimensions
            comparing_img = cv2.resize(comparing_img, resize_dims)

            _, comparing_des = detector.detectAndCompute(comparing_img, None)
            ret = 1e3 # Large value when no features are detected
            if comparing_des is not None:
                matches = bf.match(target_des, comparing_des)
                dist = [m.distance for m in matches]
                ret = sum(dist) / len(dist)
            all_res.append(ret)
        comp_pd = pd.Series(all_res)
        yield comp_pd

@pandas_udf(InferenceResults)
def image_brightness(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
    # Dimensions to resize to
    resize_dims = (200, 200)

    # Averages above this threshold are classified as day (based on empirical measurements)
    threshold = 115

    for content_series in content_series_iter:
        # Reverse content series
        all_framebytes = content_series[0]
        all_width = content_series[1]
        all_height = content_series[2]
        content_series_rev = tuple([(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)])

        all_res = []
        for c in content_series_rev:
            inp_bytes = c[0]
            inp_width = c[1]
            inp_height = c[2]
            img = np.array(bytearray(inp_bytes)).reshape(inp_height, inp_width, 3)

            # Resize to fixed dimensions
            img = cv2.resize(img, resize_dims)

            # Convert image to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # Add up all the pixel values in the V channel
            sum_brightness = np.sum(hsv[:,:,2])
            area = resize_dims[0]*float(resize_dims[1])  # pixels

            # find the average brigtness
            avg = sum_brightness/area

            label = 'night'
            cls = 0
            if (avg > threshold):
                label = 'day'
                cls = 1

            next_map = {
                'xmin'  : [None],
                'ymin'  : [None],
                'xmax'  : [None],
                'ymax'  : [None],
                'label' : [label],
                'cls'   : [cls],
                'score' : [1.0]
            }
            all_res.append(next_map)

        bright_pd = pd.DataFrame(all_res)
        yield bright_pd

@pandas_udf(InferenceResults)
def svm_classification(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
    # Dimensions to resize to
    resize_dims = (200, 200)

    # Load pre-trained model
    # TODO: IMPORTANT TO KNOW!!!
    # Currently assumes this image is local, but it could be fetched from
    # remote storage, or loaded as is done with all frames (i.e., as a bytearray
    # read from a DataFrame)
    model = pickle.load(open('./data/svm_day_night.sav', 'rb'))

    for content_series in content_series_iter:
        # Reverse content series
        all_framebytes = content_series[0]
        all_width = content_series[1]
        all_height = content_series[2]
        content_series_rev = tuple([(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)])

        all_res = []
        for c in content_series_rev:
            inp_bytes = c[0]
            inp_width = c[1]
            inp_height = c[2]
            img = np.array(bytearray(inp_bytes)).reshape(inp_height, inp_width, 3)

            # Resize to fixed dimensions
            img = cv2.resize(img, resize_dims)

            # Flatten input
            img = img.flatten()

            # Get prediction
            cls = model.predict(img.reshape(1,-1))
            cls = int(cls)

            label = None
            if cls == 0:
                label = 'night'
            else:
                label = 'day'

            next_map = {
                'xmin'  : [None],
                'ymin'  : [None],
                'xmax'  : [None],
                'ymax'  : [None],
                'label' : [label],
                'cls'   : [cls],
                'score' : [1.0]
            }
            all_res.append(next_map)

        svm_pd = pd.DataFrame(all_res)
        yield svm_pd

@pandas_udf(InferenceResults)
def motion_detect(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
    # Len(contours) above this are classified as mosition (based on empirical measurements)
    threshold = 2

    for content_series in content_series_iter:
        # Reverse content series
        all_framebytes = content_series[0]
        all_width = content_series[1]
        all_height = content_series[2]
        content_series_rev = tuple([(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)])

        all_res = []
        previous_frame = None
        for c in content_series_rev:
            inp_bytes = c[0]
            inp_width = c[1]
            inp_height = c[2]
            img = np.array(bytearray(inp_bytes)).reshape(inp_height, inp_width, 3)

            img_rgb = img

            # Grayscale and blur
            prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

            label = 'no_motion'
            cls = 0
            if previous_frame is None:
                # First frame; there is no previous one yet
                previous_frame = prepared_frame
            else:
                # Calculate difference and update previous frame
                diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
                previous_frame = prepared_frame

                # Dilate the image to make differences more seeable; suitable for contour detection
                kernel = np.ones((5, 5))
                diff_frame = cv2.dilate(diff_frame, kernel, 1)

                # Only take different areas that are different enough (>20 / 255)
                thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

                # Find contours
                contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
                if (len(contours) > threshold):
                    label = 'motion'
                    cls = 1

            next_map = {
                'xmin'  : [None],
                'ymin'  : [None],
                'xmax'  : [None],
                'ymax'  : [None],
                'label' : [label],
                'cls'   : [cls],
                'score' : [1.0]
            }
            all_res.append(next_map)

        motion_pd = pd.DataFrame(all_res)
        yield motion_pd

@pandas_udf('float')
def executor_overhead(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    t_sleep = 1
    for content_series in content_series_iter:
        # Reverse content series for creating batches
        s = perf_counter()
        all_framebytes = content_series[0]
        sleep(t_sleep)
        e = perf_counter()
        t = e-s
        predictions_pd = pd.Series([t]*len(all_framebytes))
        yield predictions_pd

@pandas_udf('float')
def simple_transfer(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
    device = torch.device('cuda' if use_cuda else 'cpu')
    for content_series in content_series_iter:
        num_frames = len(content_series[0])
        size = (num_frames, h, w, 3)
        s = perf_counter()
        tensor = torch.randint(0, 255, size, dtype=torch.uint8)
        e = perf_counter()
        t = e-s
        tensor = tensor.to(device)
        predictions_pd = pd.Series([t]*num_frames)
        yield predictions_pd

@pandas_udf(RawFrameData)
def data_generator(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
    all_f = []
    for content_series in content_series_iter:
        num_frames = len(content_series[0])
        for i in range(0, num_frames):
            size = (h, w, 3)
            tensor = torch.randint(0, 255, size, dtype=torch.uint8).numpy().tobytes()
            curr_map = {
                'id': [i],
                'framebytes': [tensor],
                'height': [h],
                'width': [w]
            }
            all_f.append(curr_map)
        video_pd = pd.DataFrame(all_f)
        yield video_pd

@pandas_udf(InferenceResults)
def complex_transfer(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
    use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
    device = torch.device('cuda' if use_cuda else 'cpu')
    for content_series in content_series_iter:
        # Reverse content series for creating batches
        all_framebytes = content_series[0]
        all_width = content_series[1]
        all_height = content_series[2]
        content_series_rev = tuple([(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)])

        # Create batches for this iteration
        batches = list(batch(content_series_rev, batch_size))

        for i,b in enumerate(batches):
            for bb in b:
                inp_bytes = bb[0]
                inp_width = bb[1]
                inp_height = bb[2]
                img_torch = torch.frombuffer(bytearray(inp_bytes), dtype=torch.uint8).reshape(inp_height, inp_width, 3)
                img_torch = img_torch.to(device)

            next_return = []
            for i in range(0, len(b)):
                next_map = {
                    'xmin'  : [0],
                    'ymin'  : [0],
                    'xmax'  : [0],
                    'ymax'  : [0],
                    'label' : ['nolabel'],
                    'cls'   : [0],
                    'score' : [0],
                }
                next_return.append(next_map)

            predictions_pd = pd.DataFrame(next_return)
            yield predictions_pd
