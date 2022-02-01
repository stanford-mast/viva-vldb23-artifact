import os
from typing import List
from pyspark.sql.functions import col, lit

from viva.nodes.node import Node
from viva.udfs.ingest import encode, framewrite, framedecode, chunk, probe
from viva.udfs.transcripts import search_transcript_udf

class EncodeNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'encoded'
        operator = encode
        super().__init__(in_columns, out_column, operator, in_literals)

class FrameDecodeNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'framedata'
        operator = framedecode
        super().__init__(in_columns, out_column, operator, in_literals)

class FrameWriteNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'framedata'
        operator = framewrite
        super().__init__(in_columns, out_column, operator, in_literals)

class VideoProbeNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'videodata'
        operator = probe 
        super().__init__(in_columns, out_column, operator, in_literals)

class ChunkNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'chunked'
        operator = chunk
        super().__init__(in_columns, out_column, operator, in_literals)

class IngestTranscriptNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'transcriptdata'
        # in_literals are used as single values, not to be treated as entire columns
        fps = in_literals[0]
        operator = search_transcript_udf(fps)
        super().__init__(in_columns, out_column, operator, [])
