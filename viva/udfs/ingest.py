import os
from typing import Tuple, List, Iterator
from glob import glob

import cv2
import ffmpeg
import pandas as pd
from pyspark.sql import Row
from pyspark.sql.functions import pandas_udf, udf
from pyspark.sql.types import StringType, ArrayType, Row

from viva.sparkmodels import VideoMetaData, FrameData, RawFrameData, ChunkVideo

def _ffmpeg_helper(fname: str, outname: str,
                   input_args: dict, output_args: dict) -> None:
    """
    calls ffmpeg doesn't return but fails if there was an error
    """
    cmd = ffmpeg.input(fname, **input_args).output(outname, **output_args)
    try:
        cmd.run(overwrite_output=True, capture_stdout=True,
                capture_stderr=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e

@pandas_udf(returnType=VideoMetaData)
def probe(uri: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
    """
    use ffmpeg probe to extract metadata from video
    """
    def _probe(uri: str) -> Tuple:
        probe = ffmpeg.probe(uri)
        video_stream = next(
            (
                stream
                for stream in probe['streams']
                if stream['codec_type'] == 'video'
            ),
            None,
        )
        if video_stream is None:
            return None

        width= int(video_stream['width'])
        height = int(video_stream['height'])
        nb_frames = int(video_stream['nb_frames'])
        duration = float(video_stream['duration'])
        fps = int(eval(video_stream['avg_frame_rate']))
        bit_rate = int(video_stream['bit_rate'])

        return [width, height, nb_frames, duration, fps, bit_rate]

    for batch in uri:
        out_df = pd.DataFrame(columns=[x.name for x in VideoMetaData.fields])
        for idx, u in enumerate(batch):
            out_df.loc[idx] = _probe(u)
        yield out_df

@pandas_udf(returnType=ChunkVideo)
def chunk(uri: pd.Series, segment_time_s: pd.Series, outdir: pd.Series) -> pd.DataFrame:
    """
    splits an input uri into equally sized videos of segment_time length.
    no encoding or additional processing happens so this is fast
    """

    #TODO if we want chunks larger than 60s segment_time arg will change

    outuris = []
    for idx, (u, s, o) in enumerate(zip(uri, segment_time_s, outdir)):
        if not os.path.exists(os.path.abspath(o)):
            os.makedirs(os.path.abspath(o))

        fn = os.path.basename(u)
        outname_base = f'{fn.split(".")[0]}'
        outname = outname_base + '_%d.mp4'
        outuri = os.path.abspath(os.path.join(o, outname))
        input_args = {
            'hwaccel': 'auto',
            'hwaccel_output_format': 'auto',
        }

        output_args = {
            'map': '0',
            'c': 'copy',
            'f': 'segment',
            'segment_time': f'00:00:{s:02}',
            'reset_timestamps': '1' #TODO not sure what this does
        }

        _ffmpeg_helper(u, outuri, input_args, output_args)
        all_outuris = glob(os.path.abspath(os.path.join(o, outname_base + '*mp4')))
        all_ids = [int(i.split('_')[-1].split('.')[0]) for i in all_outuris]
        next_map = {
            'id': all_ids,
            'uri': all_outuris
        }
        outuris.append(next_map)

    return pd.DataFrame(outuris)

@pandas_udf(returnType=StringType())
def encode(uri: pd.Series, width: pd.Series, height: pd.Series, fps: pd.Series,
           outdir: pd.Series) -> pd.Series:
    """
    """

    gpu = False
    encoder = 'h264_nvenc' if gpu else 'libx264'

    outuris = pd.Series(object)
    for idx, (u, w, h, f, o) in enumerate(zip(uri, width, height, fps, outdir)):
        if not os.path.exists(os.path.abspath(o)):
            os.makedirs(os.path.abspath(o))

        fn = os.path.basename(u)
        outname = f'{fn.split(".")[0]}-{f}-{w}x{h}.mp4'
        outuri = os.path.abspath(os.path.join(o, outname))
        if not os.path.exists(outuri):
            input_args = {
                'hwaccel': 'auto',
                'hwaccel_output_format': 'auto',
            }
            #TODO i think this is an nvidia gpu arg
            # 'vf': f'scale_npp={res}',

            res = f'{w}:{h}'
            output_args = {
                'map': '0',
                'vf': f'scale={res}',
                'r': f'{f}',
                'c:v': f'{encoder}',
            }

            _ffmpeg_helper(u, outuri, input_args, output_args)
        outuris.loc[idx] = outuri

    return outuris

@pandas_udf(RawFrameData)
def framedecode(iterator: Iterator[Tuple[pd.Series, ...]]) -> Iterator[pd.DataFrame]:
    all_results = []
    for curr_iter in iterator:
        uri = curr_iter[0]
        total_frames = curr_iter[1]
        start_chunk = curr_iter[2]
        end_chunk = curr_iter[3]
        start_window = curr_iter[4]
        end_window = curr_iter[5]

        all_results = []
        for u, tf, sc, ec, sw, ew in zip(uri, total_frames, start_chunk,
                end_chunk, start_window, end_window):
            # Frame range across all frames in original unchunked video
            sg = int(tf * sw)
            eg = int(tf * ew)

            # Figure out whether to decode any frames from this chunk
            start_frame = -1
            end_frame = -1
            if (eg >= sc) and (sg <= ec):
                if sg >= sc:
                    start_frame = sg - sc
                else:
                    start_frame = 0

                if eg <= ec:
                    end_frame = eg - sc
                else:
                    end_frame = ec - sc
            else:
                # Append empty map
                curr_map = {'id': [None],
                            'framebytes': [None],
                            'height': [None],
                            'width': [None]}
                all_results.append(curr_map)
                continue

            video = cv2.VideoCapture(u)
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            success = True
            frame_id = start_frame
            curr_frames = []
            width = -1
            height = -1
            while success and (frame_id <= end_frame):
                success, frame = video.read()

                if width < 0 and height < 0:
                    height = frame.shape[0]
                    width = frame.shape[1]

                curr_frames.append(frame.tobytes())
                frame_id += 1
            video.release()

            all_frame_id = [i for i in range(start_frame + sc, end_frame + sc + 1)]
            curr_map = {'id': all_frame_id,
                        'framebytes': curr_frames,
                        'height': [height] * len(all_frame_id),
                        'width': [width] * len(all_frame_id)}
            all_results.append(curr_map)

        video_pd = pd.DataFrame(all_results)
        yield video_pd

@udf(returnType=ArrayType(FrameData))
def framewrite(uri: str) -> List[Tuple]:
    """
    input: video uris
    output: frame uris
    """

    #TODO rewrite as pandas udf (faster)

    dn = os.path.dirname(uri)
    fn = os.path.basename(uri).split('.')[0]
    fname_dir = f'{fn}_frames'
    dir_path = os.path.abspath(os.path.join(dn, fname_dir))

    # Create directory if it does not exist
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    outuri = f'{fn}_%05d.png'
    full_outuri = os.path.abspath(os.path.join(dir_path, outuri))

    input_args = {
        'hwaccel': 'auto',
        'hwaccel_output_format': 'auto',
    }

    output_args = {'f': 'image2'}
    _ffmpeg_helper(uri, full_outuri, input_args, output_args)

    # TODO not great here clean this up
    frames = sorted(glob(f'{dir_path}/*.png'))
    all_out = [(uri, frameid) for uri, frameid in zip(frames, range(0, len(frames)))]

    return all_out
