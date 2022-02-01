import os
from typing import List
import itertools

from google.cloud import storage
from pyspark.sql import Row

from viva.nodes.node import Node

class WalkRows(Node):
    def __init__(self, in_columns: List[str], in_literals):
        super().__init__(in_columns, None, None, in_literals)

    def custom_op(self, df):
        # can only be called in createdataframe
       paths = self.in_columns
       extensions = self.in_literals
       idx = itertools.count(0)
       return [
           Row(str(os.path.abspath(os.path.join(pth, name))), next(idx))
           for path in paths
           for pth, subdirs, files in os.walk(path)
           for name in sorted(files) if name.endswith(tuple(extensions))
       ]

class RemoteBucketFetch(Node):
    def __init__(self, in_columns: List[str], in_literals: List[str]):
        super().__init__(in_columns, None)
        self.helper = WalkRows(in_columns, in_literals)

    def custom_op(self):
        # can only be called in createdataframe
        client = storage.Client()
        buckets = self.in_columns
        for bucket_name in buckets:
            bucket = client.get_bucket(bucket_name)
            for blob in bucket.list_blobs():
                destname = os.path.join(bucket_name, blob.name)
                blob.download_to_filename(destname)

        return self.helper.custom_op(None)
