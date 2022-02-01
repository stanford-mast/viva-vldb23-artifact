from tensorflow.keras.models import Sequential

def split_tf_model(model, layer_id):
    model_pre = Sequential()
    model_post = Sequential()

    for li in range(0, min(layer_id+1, len(model.layers))):
        model_pre.add( model.get_layer(index=li) )

    for li in range(layer_id+1, len(model.layers) ):
        model_post.add( model.get_layer(index=li) )

    #print("IN split_tf_model: #LAYERS in full model: %i, pre: %i, post: %i"%( len(model.layers), len(model_pre.layers), len(model_post.layers) ), model.layers[layer_id].output_shape, model_pre.layers[-1].output_shape, model_post.layers[0].input_shape )

    return model_pre, model_post
