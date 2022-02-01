from typing import List

from viva.nodes.node import Node
from viva.transcripts.search_transcript import TranscriptSearch 

class IngestTranscriptNode(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str] = []):
        out_column = 'transcriptdata'
        # in_literals are used as single values, not to be treated as entire columns
        fps = in_literals[0]
        operator = TranscriptSearch(fps).transcript_udf
        super().__init__(in_columns, out_column, operator, [])
