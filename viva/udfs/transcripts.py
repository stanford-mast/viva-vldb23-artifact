import os
import sys
import pandas as pd
from typing import Iterator

from viva.sparkmodels import TranscriptResults

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

# Helper function to convert time windows to frames
def convert_timewindow_to_frames(start, end, fps):
    """
    Returns a string with the start/end time: start,end 
    """
    # Start
    start_split = start.split(':')
    start_hours = int(start_split[0]) * 3600
    start_minutes = int(start_split[1]) * 60
    start_sec_ms_split = start_split[2].split(',')
    start_seconds = int(start_sec_ms_split[0])
    start_ms = int(start_sec_ms_split[1]) * 1e-3

    total_start_seconds = start_hours + start_minutes + start_seconds + start_ms
    starting_frame = int(total_start_seconds * fps)

    # End
    end_split = end.split(':')
    end_hours = int(end_split[0])
    end_minutes = int(end_split[1])
    end_sec_ms_split = end_split[2].split(',')
    end_seconds = int(end_sec_ms_split[0])
    end_ms = int(end_sec_ms_split[1]) * 1e-3

    total_end_seconds = end_hours + end_minutes + end_seconds + end_ms
    ending_frame = int(total_end_seconds * fps)

    return starting_frame, ending_frame

def search_transcript_udf(fps: int):
    @pandas_udf(TranscriptResults)
    def search(content_series_iter: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
        for content_series in content_series_iter:
            curr_results = []
            for c in content_series:
                """
                The input is a path to a transcript
                Currently assumes transcript is in srt format

                Format is as follows:
                1
                00:00:02,468 --> 00:00:04,203
                HELLO AND WELCOME
                """
                # dict is defined before being built up
                next_map = {'transcriptframeid'  : [],
                            'transcripturi'  : [],
                            'transcriptsegment' : []}
                fd = open(c, 'r')
                current_timewindow = ''
                for line in fd:
                    line_split = line.split()
                    if len(line_split) == 1 and line_split[0].isnumeric():
                        current_timewindow = '' # Reset
                    elif '-->' in line:
                        current_timewindow = line.strip()
                    else:
                        # Only add if current_timewindow is not empty
                        line_lower = line.lower().strip()
                        if current_timewindow != '' and line_lower:
                            current_timewindow_split = current_timewindow.split()
                            time_start = current_timewindow_split[0]
                            time_end = current_timewindow_split[2]
                            start_frame, end_frame = convert_timewindow_to_frames(time_start, time_end, fps)

                            for i in range(start_frame, end_frame + 1):
                                next_map['transcriptframeid'].append(i)
                                next_map['transcripturi'].append(c)
                                next_map['transcriptsegment'].append(line_lower)

                fd.close()
                curr_results.append(next_map)

            predictions_pd = pd.DataFrame(curr_results)
            yield predictions_pd

    return search
