"""
Data models using spark types
"""
from pyspark.sql.types import (
    StructField, StructType, BinaryType, IntegerType, LongType, StringType,
    FloatType, ArrayType
)

VideoMetaData = StructType([
    StructField('width', IntegerType(), True),
    StructField('height', IntegerType(), True),
    StructField('nb_frames', IntegerType(), True),
    StructField('duration', FloatType(), True),
    StructField('fps', IntegerType(), True),
    StructField('bit_rate', IntegerType(), True)
])

IngestVideo = StructType([
    StructField('uri', StringType(), False),
    StructField('id', IntegerType(), False),
])

ChunkVideo = StructType([
    StructField('uri', ArrayType(StringType(), True)),
    StructField('id', ArrayType(IntegerType(), True)),
])

FrameData = StructType([
    StructField('frameuri', StringType(), True),
    StructField('frameid', IntegerType(), True)
])

RawFrameData = StructType([
    StructField('width', ArrayType(IntegerType(), True)),
    StructField('height', ArrayType(IntegerType(), True)),
    StructField('framebytes', ArrayType(BinaryType(), True)),
    StructField('id', ArrayType(IntegerType(), True)),
])

# Inference results schema
InferenceResults = StructType([
    StructField('xmin', ArrayType(FloatType(), True)),
    StructField('ymin', ArrayType(FloatType(), True)),
    StructField('xmax', ArrayType(FloatType(), True)),
    StructField('ymax', ArrayType(FloatType(), True)),
    StructField('label', ArrayType(StringType(), True)),
    StructField('cls', ArrayType(IntegerType(), True)),
    StructField('score', ArrayType(FloatType(), True))
])

# Track results schema
TrackResults = StructType([
    StructField('xmin', ArrayType(FloatType(), True)),
    StructField('ymin', ArrayType(FloatType(), True)),
    StructField('xmax', ArrayType(FloatType(), True)),
    StructField('ymax', ArrayType(FloatType(), True)),
    StructField('label', ArrayType(StringType(), True)),
    StructField('cls', ArrayType(IntegerType(), True)),
    StructField('score', ArrayType(FloatType(), True)),
    StructField('track', ArrayType(IntegerType(), True))
])
# Transcript results schema
TranscriptResults = StructType([
    StructField('transcriptframeid', ArrayType(IntegerType(), True)),
    StructField('transcripturi', ArrayType(StringType(), True)),
    StructField('transcriptsegment', ArrayType(StringType(), True)),
])
