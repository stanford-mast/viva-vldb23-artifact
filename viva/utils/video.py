# some generic video processing functions

from typing import List, Any
import cv2
import numpy as np

from viva.utils.config import ConfigManager

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = (
            'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A',
            '92CC17', '3DDB86', '1A9334', '00D4BB', '2C99A8', '00C2FF',
            '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF',
            'FF95C8', 'FF37C7'
        )
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im):
        self.im = im
        self.lw = max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)

def collapse_frames(data):
    frames = []
    count = 0
    cur_row = data[count]
    nxt_row = cur_row
    while count < len(data):
        cur_row = data[count]
        boxes = []
        while cur_row.id == nxt_row.id:
            boxes.append({
                'xmin': nxt_row.xmin,
                'xmax': nxt_row.xmax,
                'ymin': nxt_row.ymin,
                'ymax': nxt_row.ymax,
                'label': nxt_row.label,
                'track': nxt_row.track if 'track' in nxt_row else '',
                'cls': nxt_row.cls,
                'score': nxt_row.score,
            })
            count += 1
            if count == len(data):
                break
            nxt_row = data[count]
        frames.append((cur_row.framebytes, cur_row.width, cur_row.height, boxes))

    return frames

def write_boxes(frame, boxes):
    """
    draw bounding boxes and label on frame
    return drawn frame
    """
    ant = Annotator(frame)
    for box in boxes:
        # Can check for any of them to be None for skipping
        if box['xmin'] is None:
            continue
        bbox = (box['xmin'], box['ymin'], box['xmax'], box['ymax'])
        label = box['label']
        track = 'None' if box['track'] is None else box['track']
        score = box['score']
        c = 0 if box['cls'] is None else box['cls']
        tlabel= f'{track} {label} {score:.2f}'
        ant.box_label(bbox, tlabel, color=colors(c))

    return ant.result()

def write_video(data: List[Any], fname: str) -> None:
    """
    write a list of frames to fname file
    """
    config = ConfigManager()
    width = config.get_value('ingest', 'width')
    height = config.get_value('ingest', 'height')
    fps = config.get_value('ingest', 'fps')
    out_res = (width, height)
    frames = collapse_frames(data)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_tracked = cv2.VideoWriter(fname, fourcc, fps, out_res)

    for fb, width, height, boxes in frames:
        frame = np.array(bytearray(fb)).reshape(height, width, 3)
        #frame = cv2.imread(fid)
        box_frame = write_boxes(frame, boxes)
        video_tracked.write(box_frame)

    video_tracked.release()
