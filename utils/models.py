import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

def getKinsLikeNetwork(input_shape, units, layers_number, hidden_layers_activation="relu"):
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    input_layer = layers.Input(shape=input_shape)

    dense = layers.Dense(
        units=units, 
        activation=hidden_layers_activation,
        kernel_initializer=initializer
    )(input_layer)

    for i in range(1, layers_number):
        dense = layers.Dense(
            units=units - 3*i, 
            activation=hidden_layers_activation,
            kernel_initializer=initializer
        )(dense)
        # dense = layers.Dropout(rate=0.4, seed=42)(dense)


    dense = layers.Dense(
        units=1, 
        activation="sigmoid",
        kernel_initializer=initializer
    )(dense)


    base_model = keras.Model(inputs=input_layer, outputs=dense)

    return base_model

def getBaseModel(input_shape, units, layers_number, hidden_layers_activation="relu"):
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    input_layer = layers.Input(shape=input_shape)

    dense = layers.Dense(
        units=units, 
        activation=hidden_layers_activation,
        kernel_initializer=initializer
    )(input_layer)

    for i in range(1, layers_number):
        dense = layers.Dense(
            units=units - 4*i, 
            activation=hidden_layers_activation,
            kernel_initializer=initializer
        )(dense)

    dense = layers.Dense(
        units=1, 
        activation="sigmoid",
        kernel_initializer=initializer
    )(dense)

    base_model = keras.Model(inputs=input_layer, outputs=dense)

    return base_model

def getSymmetricModel(input_shape, base_model, use_triplets):
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
        

    processed_a = base_model(input_a)
    processed_b = base_model(input_b)
        

    input_models = [processed_a, processed_b]
    inputs = [input_a, input_b]

    if use_triplets:
        input_c = layers.Input(shape=input_shape)
        processed_c = base_model(input_c)
        input_models.append(processed_c)
        inputs.append(input_c)

    lambda_layer = layers.Lambda(
        lambda embeddings: tf.concat(embeddings, axis=1),
    )(input_models)

        
    model = keras.Model(inputs, lambda_layer)

    return model

def getQuadModel(input_shape, base_model):
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    input_c = layers.Input(shape=input_shape)    
    input_d = layers.Input(shape=input_shape)    

    processed_a = base_model(input_a)
    processed_b = base_model(input_b)
    processed_c = base_model(input_c)
    processed_d = base_model(input_d)   

    input_models = [processed_a, processed_b, processed_c, processed_d]
    inputs = [input_a, input_b, input_c, input_d]       
        

    lambda_layer = layers.Lambda(
        lambda embeddings: tf.concat(embeddings, axis=1),
    )(input_models)

        
    model = keras.Model(inputs, lambda_layer)

    return model
