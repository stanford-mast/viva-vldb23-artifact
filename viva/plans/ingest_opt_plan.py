import os
import sys
import pyspark.sql.functions as F
from pyspark.sql.functions import explode, col, lag
from pyspark.sql.window import Window

from viva.nodes.ingest_nodes import EncodeNode, VideoProbeNode, FrameDecodeNode, ChunkNode
from viva.utils.config import ConfigManager

config = ConfigManager()
outdir = config.get_value('storage', 'output')
width = config.get_value('ingest', 'width')
height = config.get_value('ingest', 'height')
fps = config.get_value('ingest', 'fps')
chunk_size = config.get_value('ingest', 'chunk_size_s')
start_fraction = config.get_value('ingest', 'start_fraction')
end_fraction = config.get_value('ingest', 'end_fraction')

def chunk_select(df):
    df = df.select(col('chunked.*'))
    df = df.selectExpr('inline(arrays_zip(id,uri))')
    return df

def probe_select(df):
    df = df.select(col('id'), col('encoded').alias('uri'), col('videodata.nb_frames').alias('chunk_frames'))

    # Generate a cumulative sum to compute start and end global frame IDs
    ws = Window.partitionBy().orderBy('id').rowsBetween(-sys.maxsize, 0)
    df = df.withColumn('cumsum', F.sum(df.chunk_frames).over(ws))

    wl = Window.partitionBy().orderBy('id')
    df = df.withColumn("start_chunk", lag("cumsum", 1, 0).over(wl))

    df = df.withColumn("end_chunk", (df.cumsum - 1))

    # This is the number of total frames
    df_m = df.agg(F.max(col('end_chunk')).alias('total_frames'))

    df = df.join(df_m).select(col('id'), col('uri'), col('start_chunk'), col('end_chunk'), col('total_frames'))
    return df

def frame_select(df):
    df = df.select(col('uri'), col('framedata.*'))
    df = df.selectExpr('%s as %s' % ('uri', 'uri'), \
                       'inline(arrays_zip(id,width,height,framebytes))')

    # Drop the rows where no frames were decoded from (i.e., id == null)
    df = df.filter(col('id').isNotNull())
    return df

Plan = []
# Need to chunk before encoding since keyframes will get messed up after encoding at a different FPS
cn = ChunkNode(['uri'], [chunk_size, 'tmp'])
cn.add_filter(chunk_select)
Plan.append(cn)

Plan.append(EncodeNode(['uri'], [width, height, fps, outdir]))
next_input = [Plan[len(Plan)-1].out_column]

vp = VideoProbeNode(next_input)
vp.add_filter(probe_select)
Plan.append(vp)

fr = FrameDecodeNode(['uri', 'total_frames', 'start_chunk', 'end_chunk'], [start_fraction, end_fraction])
fr.add_filter(frame_select)
Plan.append(fr)
