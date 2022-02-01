"""
Based on: https://docs.databricks.com/_static/notebooks/deep-learning/dist-img-infer-2-pandas-udf.html
and: https://docs.databricks.com/_static/notebooks/deep-learning/pytorch-images.html
"""

import os
import io
import sys
import json
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import pandas as pd
import torch
torch.warnings.filterwarnings('ignore')
from PIL import Image
from itertools import islice
from typing import Iterator, Tuple

from viva.sparkmodels import InferenceResults, TrackResults
from viva.utils.config import viva_setup, ConfigManager
config = ConfigManager()

from viva.utils.profiling import perf_count
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import (
    StructField,
    ArrayType,
    StructType,
    BinaryType,
    IntegerType,
    StringType,
    FloatType
)

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo, NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

import cv2
from sklearn.metrics.pairwise import cosine_similarity

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.datasets.folder import default_loader

from tensorflow.keras.applications.imagenet_utils import decode_predictions

from facenet_pytorch import MTCNN

sys.path.append(os.path.join(os.path.dirname(__file__), 'CameraTraps'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_sort/deep_sort/deep/reid'))
from detection import run_tf_detector_batch as animal_detect

from deepface.commons import functions
from deepface.extendedmodels import Age

from viva.utils.tf_helpers import split_tf_model

# Constants
batch_size = 16 # Constant for now, but should likely become more dynamic

# Create batches
def batch(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

# Image dataset class for producing batches of transformed images
class ImageDataset(Dataset):
    def __init__(self, paths, preprocess_type='ImageNet'):
        self.paths = paths
        self.preprocess_type = preprocess_type

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        next_input = self.paths[index]
        inp_bytes = next_input[0]
        inp_width = next_input[1]
        inp_height = next_input[2]
        img_np = np.array(bytearray(inp_bytes)).reshape(inp_height, inp_width, 3)
        image = Image.fromarray(img_np)
        if self.preprocess_type == 'ImageNet':
            image = self._imagenet_preprocess(image)
            return image

    def _imagenet_preprocess(self, image):
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return transform(image)

def imagenet_model_udf(model_fn):
    """
    # Wraps an ImageNet model into a Pandas UDF that makes predictions.
    """
    use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
    device = torch.device('cuda' if use_cuda else 'cpu')

    @pandas_udf(InferenceResults)
    def predict(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        model = model_fn()
        model.eval()
        model.to(device)
        for content_series in content_series_iter:
            # Reverse content series for creating batches
            all_framebytes = content_series[0]
            all_width = content_series[1]
            all_height = content_series[2]
            content_series_rev = [(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)]

            dataset = ImageDataset(content_series_rev,
                                   preprocess_type='ImageNet')
            loader = DataLoader(dataset, batch_size=batch_size)
            with torch.no_grad():
                for image_batch in loader:
                    predictions_raw = model(image_batch.to(device)).cpu()
                    predictions = predictions_raw.numpy()
                    # To convert to probability
                    probabilities_raw = torch.nn.functional.softmax(predictions_raw, dim=1)
                    probabilities, _ = torch.topk(probabilities_raw, 1) # Only need the score

                    # Convert predictions to dictionary that matches InferenceResults
                    # Each prediction is of the form (class, description, score)
                    # We don't use the score here, rather we use the probabilities extracted above from softmax
                    next_return = []
                    decoded_predictions = decode_predictions(predictions, top=1)
                    for dp,prob in zip(decoded_predictions,probabilities):
                        next_decoded = dp[0]
                        next_label = next_decoded[1]
                        next_score = prob.item()
                        next_map = {
                            'xmin'  : [None],
                            'ymin'  : [None],
                            'xmax'  : [None],
                            'ymax'  : [None],
                            'label' : [next_label],
                            'cls'   : [None],
                            'score' : [next_score]
                        }
                        next_return.append(next_map)
                    predictions_pd = pd.DataFrame(next_return)
                    yield predictions_pd

    return predict

def qclassification_model_udf(model_fn):
    """
    # Wraps an ImageNet model into a Pandas UDF that makes predictions.
    """
    use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
    device = torch.device('cuda' if use_cuda else 'cpu')

    @pandas_udf(InferenceResults)
    def predict(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        model, preprocess, weights = model_fn()
        model.eval()
        model.to(device)
        for content_series in content_series_iter:
            # Reverse content series for creating batches
            all_framebytes = content_series[0]
            all_width = content_series[1]
            all_height = content_series[2]
            content_series_rev = tuple([(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)])

            # Create batches for this iteration
            batches = list(batch(content_series_rev, batch_size))

            for i,b in enumerate(batches):
                next_return = []
                for bb in b:
                    inp_bytes = bb[0]
                    inp_width = bb[1]
                    inp_height = bb[2]
                    img_np = np.array(bytearray(inp_bytes)).reshape(inp_height, inp_width, 3)
                    img = Image.fromarray(img_np)
                    preproc_batch = preprocess(img).unsqueeze(0)
                    prediction = model(preproc_batch).squeeze(0).softmax(0)
                    class_id = int(prediction.argmax().item())
                    score = prediction[class_id].item()
                    category_name = weights.meta["categories"][class_id]

                    next_map = {
                        'xmin'  : [None],
                        'ymin'  : [None],
                        'xmax'  : [None],
                        'ymax'  : [None],
                        'label' : [category_name],
                        'score' : [score],
                        'cls'   : [class_id],
                    }
                    next_return.append(next_map)
                predictions_pd = pd.DataFrame(next_return)
                yield predictions_pd

    return predict

def yolo_model_udf(model_fn):
    """
    Define the function for model inference using YOLOv5.
    """
    @pandas_udf(InferenceResults)
    def predict(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        model = model_fn()

        for content_series in content_series_iter:
            # Reverse content series for creating batches
            all_framebytes = content_series[0]
            all_width = content_series[1]
            all_height = content_series[2]
            content_series_rev = tuple([(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)])

            # Create batches for this iteration
            batches = list(batch(content_series_rev, batch_size))

            for i,b in enumerate(batches):
                next_inp_list = []
                for bb in b:
                    inp_bytes = bb[0]
                    inp_width = bb[1]
                    inp_height = bb[2]
                    img = np.array(bytearray(inp_bytes)).reshape(inp_height, inp_width, 3)
                    next_inp_list.append(img)

                results = model(next_inp_list)
                results_pd = results.pandas().xyxy

                # Convert predictions to dictionary that matches InferenceResults
                # This is a matter of extracting a list per column per output
                next_return = []
                for dp in results_pd:
                    next_map = {
                        'xmin'  : dp['xmin'].to_list(),
                        'ymin'  : dp['ymin'].to_list(),
                        'xmax'  : dp['xmax'].to_list(),
                        'ymax'  : dp['ymax'].to_list(),
                        'label' : dp['name'].to_list(),
                        'cls'   : dp['class'].to_list(),
                        'score' : dp['confidence'].to_list(),
                    }
                    next_return.append(next_map)

                predictions_pd = pd.DataFrame(next_return)
                yield predictions_pd

    return predict

def img2vec_model_udf(model_fn):
    """
    Define the function for image to vector using img2vec (ResNet-18)
    """

    @pandas_udf(BinaryType())
    def predict(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        model = model_fn()

        for content_series in content_series_iter:
            # Reverse content series for creating batches
            all_framebytes = content_series[0]
            all_width = content_series[1]
            all_height = content_series[2]
            content_series_rev = tuple([(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)])

            # Create batches for this iteration
            batches = list(batch(content_series_rev, batch_size))

            for i,b in enumerate(batches):
                next_inp_list = []
                for bb in b:
                    inp_bytes = bb[0]
                    inp_width = bb[1]
                    inp_height = bb[2]
                    img = np.array(bytearray(inp_bytes)).reshape(inp_height, inp_width, 3)
                    img = Image.fromarray(img, 'RGB')
                    next_inp_list.append(img)

                results = model.get_vec(next_inp_list)

                # Convert each vector to a bytes array
                next_return = [r.tobytes() for r in results]

                predictions_pd = pd.Series(next_return)
                yield predictions_pd

    return predict

def kmeans_model_udf(model_fn):
    """
    Define the function for KMeans clustering
    """

    @pandas_udf(InferenceResults)
    def predict(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        centroids, labels = model_fn()
        vector_size = centroids.shape[1]

        for content_series in content_series_iter:
            curr_results = []
            for c in content_series:
                next_vec = np.frombuffer(bytearray(c), dtype=np.float32).reshape(vector_size,)

                # Find the cluster with the max cosine similarity
                max_ind = -1
                max_sim = -1
                for i in range(centroids.shape[0]):
                    next_centroid = centroids[i,:]
                    sim = cosine_similarity(next_vec.reshape((1, -1)), next_centroid.reshape((1, -1)))[0][0]
                    if sim > max_sim:
                        max_sim = sim
                        max_ind = i

                # Get the associated label array
                label = labels[max_ind]
                score = [max_sim] * len(label)

                # Build the result
                next_map = {
                    'xmin'  : [None],
                    'ymin'  : [None],
                    'xmax'  : [None],
                    'ymax'  : [None],
                    'label' : label,
                    'cls'   : [None],
                    'score' : score,
                }
                curr_results.append(next_map)

            predictions_pd = pd.DataFrame(curr_results)
            yield predictions_pd

    return predict

def action_model_udf(model_fn):
    """
    Define the function for model inference using an action detector (3D-ResNet).
    """
    use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
    device = torch.device('cuda' if use_cuda else 'cpu')

    @pandas_udf(InferenceResults)
    def predict(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        model = model_fn()

        # Load labels
        # TODO: IMPORTANT TO KNOW!!!
        # Currently assumes labels are local, but these can be fetched from remote
        # storage (or something)
        json_filename = os.path.join(config.get_value('storage', 'input'), 'kinetics_classnames.json')
        with open(json_filename, "r") as f:
            kinetics_classnames = json.load(f)

        kinetics_id_to_classname = {}
        for k, v in kinetics_classnames.items():
            kinetics_id_to_classname[v] = str(k).replace('"', "")

        # Define transform
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 40 # This parameter needs to be tuned

        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )

        for content_series in content_series_iter:
            # Reverse content series for creating batches
            all_framebytes = content_series[0]
            all_width = content_series[1]
            all_height = content_series[2]
            content_series_rev = tuple([(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)])

            # Create batches for this iteration
            batches = list(batch(content_series_rev, batch_size))

            for i,b in enumerate(batches):
                # Prepare next batch to be fed in
                all_tensors = []
                for bb in b:
                    inp_bytes = bb[0]
                    inp_width = bb[1]
                    inp_height = bb[2]
                    img_torch = torch.frombuffer(bytearray(inp_bytes), dtype=torch.uint8).reshape(inp_height, inp_width, 3)
                    img_torch = img_torch.permute(2, 0, 1)
                    all_tensors.append(img_torch)

                video_data = {
                    'video': torch.stack(all_tensors, dim=1)
                }
                video_data = transform(video_data)
                inputs = video_data["video"]
                inputs = inputs.to(device, non_blocking=True)

                preds = model(inputs[None, ...])

                # Get the predicted class
                post_act = torch.nn.Softmax(dim=1)
                preds = post_act(preds)
                pred_class = preds.topk(k=1).indices[0]
                pred_label = kinetics_id_to_classname[int(pred_class)]
                pred_score = preds.topk(k=1).values[0].item()

                next_map = {
                    'xmin'  : [None],
                    'ymin'  : [None],
                    'xmax'  : [None],
                    'ymax'  : [None],
                    'label' : [pred_label],
                    'cls'   : [int(pred_class)],
                    'score' : [pred_score]
                }
                # Assign the same output to all inputs since the output length
                # must be the same as the length of the input
                next_return = [next_map for i in range(len(b))]

                predictions_pd = pd.DataFrame(next_return)
                yield predictions_pd

    return predict

def emotion_model_udf(model_fn):
    """
    Define the function for model inference using emotion detection (FER)
    """

    @pandas_udf(InferenceResults)
    def predict(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        model = model_fn()

        class_map = {
            "angry"    : 0,
            "disgust"  : 1,
            "fear"     : 2,
            "happy"    : 3,
            "sad"      : 4,
            "surprise" : 5,
            "neutral"  : 6
        }

        for content_series in content_series_iter:
            # Reverse content series for creating batches
            all_framebytes = content_series[0]
            all_width = content_series[1]
            all_height = content_series[2]
            content_series_rev = tuple([(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)])

            # Create batches for this iteration
            batches = list(batch(content_series_rev, batch_size))

            for i,b in enumerate(batches):
                curr_results = []
                for bb in b:
                    inp_bytes = bb[0]
                    inp_width = bb[1]
                    inp_height = bb[2]
                    img = np.array(bytearray(inp_bytes)).reshape(inp_height, inp_width, 3)
                    prediction = model.detect_emotions(img)

                    # Convert predictions to dictionary that matches InferenceResults
                    # A tad tricky to format: dict is defined before being built up
                    next_map = {
                        'xmin'  : [],
                        'ymin'  : [],
                        'xmax'  : [],
                        'ymax'  : [],
                        'label' : [],
                        'cls'   : [],
                        'score' : []
                    }
                    for p in prediction:
                        box = p['box']
                        all_emotions = p['emotions']
                        top_emotion = max(all_emotions, key=all_emotions.get)
                        top_emotion_class = class_map.get(top_emotion, len(class_map))
                        score = all_emotions[top_emotion]

                        next_map['xmin'].append(box[0])
                        next_map['ymin'].append(box[1])
                        next_map['xmax'].append(box[0] + box[2])
                        next_map['ymax'].append(box[1] + box[3])
                        next_map['label'].append(top_emotion)
                        next_map['cls'].append(top_emotion_class)
                        next_map['score'].append(score)

                    curr_results.append(next_map)

                predictions_pd = pd.DataFrame(curr_results)
                yield predictions_pd

    return predict

def facenet_model_udf(model_fn):
    """
    Define the function for model inference using facenet
    """
    use_cuda = torch.cuda.is_available() and config.get_value('execution', 'gpu')
    device = torch.device('cuda' if use_cuda else 'cpu')

    @pandas_udf(InferenceResults)
    def predict(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        # Define the function for model inference using face recognition (MTCNN + IncRes)
        model, mtcnn = model_fn()

        # Read in labels from an npy.
        # TODO: IMPORTANT TO KNOW!!!
        # Currently assumes labels are local, but these can be fetched from remote
        # storage (or something)
        labels = np.load(os.path.join(config.get_value('storage', 'input'), 'rcmalli_vggface_labels_v2.npy'))

        for content_series in content_series_iter:
            # Reverse content series for creating batches
            all_framebytes = content_series[0]
            all_width = content_series[1]
            all_height = content_series[2]
            content_series_rev = tuple([(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)])

            # Create batches for this iteration
            batches = list(batch(content_series_rev, batch_size))

            for i,b in enumerate(batches):
                curr_results = []
                next_inp_list = []
                for bb in b:
                    inp_bytes = bb[0]
                    inp_width = bb[1]
                    inp_height = bb[2]
                    img = np.array(bytearray(inp_bytes)).reshape(inp_height, inp_width, 3)
                    next_inp_list.append(img)
                    # img_bbox = mtcnn.detect(img)[0]
                    # img_cropped = mtcnn(img)
                img_bboxes, _ = mtcnn.detect(next_inp_list)
                img_crops = mtcnn(next_inp_list)
                for idx, img_cropped in enumerate(img_crops):

                    next_map = {
                        'xmin'  : [],
                        'ymin'  : [],
                        'xmax'  : [],
                        'ymax'  : [],
                        'label' : [],
                        'cls' : [],
                        'score' : []
                    }

                    # If there are no faces, we skip
                    if img_cropped is not None:
                        prediction = model(img_cropped.to(device, non_blocking=True))
                        # To convert to probability
                        probabilities_raw = torch.nn.functional.softmax(prediction, dim=1)
                        probabilities, _ = torch.topk(probabilities_raw, 1) # Only need the score

                        # Get the output and labels
                        max_vals = torch.max(prediction, dim=1)
                        for i in range(len(max_vals.values)):
                            bbox = img_bboxes[idx][i]
                            xmin, ymin, xmax, ymax = bbox
                            ind = max_vals.indices[i].item()
                            label = labels[ind].item()
                            # Get the probability from the softmax as the score
                            score = probabilities[i].item()

                            next_map['xmin'].append(xmin)
                            next_map['ymin'].append(ymin)
                            next_map['xmax'].append(xmax)
                            next_map['ymax'].append(ymax)
                            next_map['label'].append(label)
                            next_map['cls'].append(ind)
                            next_map['score'].append(score)
                    else:
                        for k in next_map.keys():
                            next_map[k].append(None)

                    curr_results.append(next_map)
                predictions_pd = pd.DataFrame(curr_results)
                yield predictions_pd

    return predict

#TODO: IMPORTANT!!!! This UDF needs to be updated to support raw binary image inputs
def animal_model_udf(model_fn):
    """
    Define the function for model inference using animal detect (MegaDetector)
    """

    @pandas_udf(InferenceResults)
    def predict(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        model_path = model_fn()

        label_map = {
            '1': 'animal',
            '2': 'person',
            '3': 'vehicle',
            '4': 'group'
        }

        for content_series in content_series_iter:
            # Create batches for this iteration
            batches = list(batch(content_series, batch_size))

            for i,b in enumerate(batches):
                results = animal_detect.load_and_run_detector_batch(model_file=model_path, image_file_names=list(b))

                curr_results=[]
                for r in results:
                    next_map = {
                        'xmin'  : [],
                        'ymin'  : [],
                        'xmax'  : [],
                        'ymax'  : [],
                        'label' : [],
                        'cls' : [],
                        'score' : []
                    }
                    for d in r['detections']:
                        xmin = d['bbox'][0]
                        ymin = d['bbox'][1]
                        xmax = xmin + d['bbox'][2]
                        ymax = ymin + d['bbox'][3]
                        label = label_map.get(d['category'], str(d['category']))
                        score = d['conf']

                        next_map['xmin'].append(xmin)
                        next_map['ymin'].append(ymin)
                        next_map['xmax'].append(xmax)
                        next_map['ymax'].append(ymax)
                        next_map['label'].append(label)
                        next_map['cls'].append(d['category'])
                        next_map['score'].append(score)

                    curr_results.append(next_map)
                predictions_pd = pd.DataFrame(curr_results)
                yield predictions_pd

    return predict

def tracking_model_udf(model_fn):
    """
    Define the function for model inference using YOLOv5.
    """

    @pandas_udf(TrackResults)
    def predict(content_series_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        deepsort = model_fn()

        def xyxy2xywh(x):
            # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
            y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
            y[:, 2] = x[:, 2] - x[:, 0]  # width
            y[:, 3] = x[:, 3] - x[:, 1]  # height
            return y

        for content_series in content_series_iter:
            # Create batches for this iteration
            curr_results = []
            for idx, row in content_series[3].iterrows():
                # if idx % 10 == 0:
                #     print('Object tracker %d of %d' % (idx+1, len(content_series[3])))

                inp_bytes  = content_series[0][idx]
                inp_width  = content_series[1][idx]
                inp_height = content_series[2][idx]
                img = np.array(bytearray(inp_bytes)).reshape(inp_height, inp_width, 3)

                nparr = np.zeros((len(row.label), 6))
                nparr[:, 0] = row.xmin
                nparr[:, 1] = row.ymin
                nparr[:, 2] = row.xmax
                nparr[:, 3] = row.ymax
                nparr[:, 4] = row.score
                nparr[:, 5] = row.cls
                det = torch.from_numpy(nparr)
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                cls = det[:, 5]
                results = deepsort.update(xywhs, confs, cls, img)
                next_map = {
                    'xmin'  : [],
                    'ymin'  : [],
                    'xmax'  : [],
                    'ymax'  : [],
                    'label' : [],
                    'track' : [],
                    'cls'  :  [],
                    'score' : []
                }
                if len(results) > 0:
                    tid = results[:, 4]
                    # not used, just for reference
                    # xmin = results[:, 0]
                    # ymin = results[:, 1]
                    # xmax = results[:, 2]
                    # ymax = results[:, 3]
                    # label = results[:, 5]
                    next_map = {
                        'xmin'  : list(row.xmin),
                        'ymin'  : list(row.ymin),
                        'xmax'  : list(row.xmax),
                        'ymax'  : list(row.ymax),
                        'label' : list(row.label),
                        'track' : tid.tolist(),
                        'cls'   : list(row.cls),
                        'score' : list(row.score)
                    }
                # append results even if empty
                curr_results.append(next_map)

            predictions_pd = pd.DataFrame(curr_results)
            yield predictions_pd

    return predict

# Deepface Constants
df_gender_labels = ['Woman', 'Man']
df_emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
df_race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

# Takes a prefix model as input, returns a column (pd.Series) that stores the embedding.
def deepface_prefix_model_udf(model):
    @pandas_udf(BinaryType())
    def predict(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for content_series in content_series_iter:
            all_framebytes = content_series[0]
            all_width = content_series[1]
            all_height = content_series[2]

            s_zip = [(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)]
            img_np_series = [ np.array(bytearray(x[0])).reshape(x[2], x[1], 3) for x in s_zip]
            img_region_prep_series = [ functions.preprocess_face(img = x, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = 'opencv', return_region = True) for x in img_np_series]

            img_prep_series = [x[0] for x in img_region_prep_series]
            img_np_series_final = np.squeeze(np.array(img_prep_series), axis=1)

            embeds =  model.predict(img_np_series_final, batch_size=batch_size)
            #flatten the embeddings:
            embeds_flat = [x.tobytes() for x in embeds]
            yield pd.Series(embeds_flat)

    return predict

# Takes as input full model, layer_id to split the model into pre,suffix
def deepface_suffix_model_udf(model, model_type, layer_id):
    @pandas_udf(InferenceResults)
    def predict(si_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:

        # first, split the model to get the suffix, starting from layer_id: (prefix has layers [0,layer_id-1])
        model_pre, model_suffix = split_tf_model(model, layer_id-1)
        pre_embed_dim = model_pre.layers[-1].output_shape[1:]

        for content_series in si_iter:
            all_embeds = [np.frombuffer(bytearray(x), dtype=np.float32).reshape(pre_embed_dim) for x in content_series]
            all_embeds = np.array(all_embeds)

            all_preds = model_suffix.predict(all_embeds, batch_size=batch_size)

            if 'Age' in model_type:
                all_preds = [int(Age.findApparentAge(x)) for x in all_preds]
                preds_labels = [str(x) for x in all_preds]
            elif 'Gender' in model_type:
                all_preds = [ np.argmax(x) for x in all_preds]
                preds_labels = [df_gender_labels[x] for x in all_preds]
            elif 'Race' in model_type:
                all_preds = [ np.argmax(x) for x in all_preds]
                preds_labels = [df_race_labels[x] for x in all_preds]

            preds_df = [{
                'xmin': [None], 
                'xmax': [None], 
                'ymin': [None], 
                'ymax': [None], 
                'label': [preds_labels[i]], 
                'cls': [all_preds[i]], 
                'score': [1.0]
            } for i in range(len(all_preds))]


            yield pd.DataFrame(preds_df)

    return predict

def deepface_model_udf(model, model_type):
    @pandas_udf(InferenceResults)
    def predict(si_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        for content_series in si_iter:
            #Preprocess:
            all_framebytes = content_series[0]
            all_width = content_series[1]
            all_height = content_series[2]

            s_zip = [(f,w,h) for f,w,h in zip(all_framebytes, all_width, all_height)]
            img_np_series = [ np.array(bytearray(x[0])).reshape(x[2], x[1], 3) for x in s_zip]
            img_region_prep_series = [ functions.preprocess_face(img = x, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = 'opencv', return_region = True) for x in img_np_series]

            img_prep_series = [x[0] for x in img_region_prep_series]
            img_np_series_final = np.squeeze(np.array(img_prep_series), axis=1)

            # Running the model
            preds = model.predict(img_np_series_final, batch_size=batch_size)

            # Postprocessing the predictions:
            if 'Age' in model_type:
                preds = [int(Age.findApparentAge(x)) for x in preds]
                preds_labels = [str(x) for x in preds]
            elif 'Gender' in model_type:
                preds = [ np.argmax(x) for x in preds]
                preds_labels = [df_gender_labels[x] for x in preds]
            elif 'Race' in model_type:
                preds = [ np.argmax(x) for x in preds ]
                preds_labels = [df_race_labels[x] for x in preds]

            preds_df = [{
                'xmin': [img_region_prep_series[i][1][0]],
                'xmax': [img_region_prep_series[i][1][2]],
                'ymin': [img_region_prep_series[i][1][1]],
                'ymax': [img_region_prep_series[i][1][3]],
                'label': [preds_labels[i]],
                'cls': [preds[i]],
                'score': [1.0]
            } for i in range(len(preds))]
            yield pd.DataFrame(preds_df)

    return predict
