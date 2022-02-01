from pyspark.sql.functions import (
    col, ltrim, count, struct, abs, lag, when, signum, collect_set,
    array_contains, lit, row_number
)
from pyspark.sql.window import Window

#===== Start Common =====#
def transcript_search(df, keyword):
    cl = 'transcriptsegment'
    df = df.filter(col(cl).contains(keyword))
    return df

def objects_or(df, objects):
    cl = 'label'
    return df.filter(col(cl).isin(objects))

def object_filter(df, obj):
    cl = 'label'
    return df.filter(col(cl) == obj)

def similarity_filter(df, threshold):
    df = df.filter(col('similarity') < threshold)
    return df

def drop_frames(df, fraction, do_random=True):
    """
    drop configurable number of frames to simulate accuracy hit
    """
    retain = float(1 - fraction)
    if do_random:
        df = df.sample(withReplacement=False, fraction=retain, seed=1234).drop()
    else:
        w = Window.partitionBy().orderBy(col("id"))
        df = df.withColumn("rn", row_number().over(w)).filter(col("rn") % int(1/retain) == 0).drop(*["rn"])
    return df

def drop_confidence(df, confidence_change):
    df = df.withColumn('score', col('score') - lit(confidence_change))
    return df

#===== End Common =====#

#===== Start Amsterdam Dock =====#
def add_field_to_struct(df, node):
    field = 'uri'

    s_fields = df.schema[node].dataType.names
    df = df.withColumn(
        node,
        struct(*([col(node)[c].alias(c) for c in s_fields] + [col(field).alias(field)]))
    )
    return df

def calculate_movement(df):
    pts = 10
    df = df.withColumn('xcent', (df.xmin + df.xmax) / 2)\
            .withColumn('ycent', (df.ymin + df.ymax) / 2)
    w = Window.partitionBy().orderBy(df.track)
    df = df.withColumn('dx', df.xcent - lag(df.xcent, pts, 0).over(w))\
            .withColumn('dy', df.ycent - lag(df.ycent, pts, 0).over(w))

    return df

def remove_stationary(df):
    return df.filter((abs(df.dx) > 1) & (abs(df.dy) > 1))

def find_directions(df):
    move = 20
    df = df.withColumn('xdir', when((df.dx > move) & (signum(df.dx) == 1), 'right').otherwise('left'))\
            .withColumn('ydir', when((df.dy > move) & (signum(df.dy) == 1), 'up').otherwise('down'))
    return df

def left_turns(df):
    return df.filter((df.xdir == 'left') & (df.ydir == 'up'))
#===== End Amsterdam Dock =====#

#===== Start Angry Bernie =====#
def two_people(df):
    cl = 'label'
    df = df.filter(col(cl) == 'person')
    dfg = df.groupBy(col('id'))\
            .agg(count(col('id')).alias('count'))
    dfg = dfg.filter(col('count') == 2)
    df = df.join(dfg, df.id == dfg.id, 'leftsemi')

    return df

def faces_and(df):
    cl = 'label'
    # Bernie could also be detected as La_Mona_Jiménez
    faces = ['Angelo_Scola', 'Jake_Tapper']
    #faces = ['La_Mona_Jiménez', 'Jake_Tapper']
    # faces = ['Marco_Rubio', 'Jake_Tapper']
    fl = f'{cl}_list'
    dfg = df.groupBy(col('id'))\
            .agg(collect_set(ltrim(cl)).alias(fl))
    for f in faces:
        dfg = dfg.filter(array_contains(col(fl), f))
    df = df.join(dfg, df.id == dfg.id, 'leftsemi')

    return df
#===== End Angry Bernie =====#

#===== Start Dunk =====#
def faces_lebron_expand(df):
    cl = 'label'
    person = 'LeBron_James'
    window_expansion_thresh = 30 # In frames for now
    df_person = df.filter(ltrim(col(cl)) == person)
    if df_person.count() > 0:
        # Get a min/max id for person
        min_person = df_person.agg({"id": "min"}).collect()[0]['min(id)']
        max_person = df_person.agg({"id": "max"}).collect()[0]['max(id)']

        # Set window to be slighly expanded
        min_person -= window_expansion_thresh
        max_person += window_expansion_thresh

        # Filter original dataframe to be within the expanded window
        return df.select("*").where((df.id >= min_person) & (df.id <= max_person))
    else:
        return df_person
#===== End Dunk =====#

#===== Start DeepFace =====#
def age_old_filter(df):
    df = df.select("*").where( df.cls > 19 )
    return df
#===== End DeepFace =====#
