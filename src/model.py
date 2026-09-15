# creating the model a frozen Xception based model
from tensorflow import keras
from tensorflow.keras.applications.xception import Xception
import config

def make_model(input_size=config.INPUT_SIZE,
               learning_rate=config.LEARNING_RATE,
               size_inner=config.SIZE_INNER,
               dropout=config.DROPRATE,
               num_classes=10):

    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )
    base_model.trainable = False # This

    inputs = keras.Input(shape=(input_size, input_size, 3)) # this is the input to the model,the doorway where a photo enters.
    base = base_model(inputs, training=False) # we are passing the inputs to the base model and training=False because we are not training the base model, if training=True then the base model will start trainig on 20 million data it already has plus our new data
    vectors = keras.layers.GlobalAveragePooling2D()(base)# cnn layers have a global average pooling layer and we have reduced it to a vector
    inner = keras.layers.Dense(size_inner, activation="relu")(vectors)#100 neurons learning to combine those 2048 numbers into clothing-shaped ideas
    drop = keras.layers.Dropout(dropout)(inner)#prevents the network leaning on any one neuron, this is to prevent overfitting, we don not wnat model to memorize anything
    outputs = keras.layers.Dense(num_classes)(drop) # this is the output layer

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),# adam decides how the adjust the number after each mistake
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),#scoring rule for how wrong an answer was.
        metrics=["accuracy"]
    )
    return model