from pyspark.sql.functions import col

def explode_preds(df, node):
    zipf = ','.join(df.schema[node].dataType.names)
    df = df.select('uri', 'id', 'width', 'height', 'framebytes', f'{node}.*')
    df = df.selectExpr(
        '%s as %s' % ('uri', 'uri'),
        '%s as %s' % ('id', 'id'),
        '%s as %s' % ('width', 'width'),
        '%s as %s' % ('height', 'height'),
        '%s as %s' % ('framebytes', 'framebytes'),
        f'inline(arrays_zip({zipf}))'
    )
    return df

def quality_filter(df, score = 0.6):
    return df.filter(col('score') > score)

def proxy_quality_filter(df, thresh):
    df_needs_general = df.filter(col('score') < thresh)
    df_passes_proxy = df.filter(col('score') >= thresh)

    return (df_needs_general, df_passes_proxy)
