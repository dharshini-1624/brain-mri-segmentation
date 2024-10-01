import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def conv_block(x, filters, kernel_size=(3, 3), padding="same", activation="relu"):
    conv = Conv2D(filters, kernel_size, padding=padding, activation=activation)(x)
    conv = Conv2D(filters, kernel_size, padding=padding, activation=activation)(conv)
    return conv

def unet_plus_plus(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)
    
   
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_block(pool4, 1024)

    
    up4 = UpSampling2D(size=(2, 2))(conv5)
    merge4 = concatenate([conv4, up4], axis=3)
    conv6 = conv_block(merge4, 512)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    merge3 = concatenate([conv3, up3], axis=3)
    conv7 = conv_block(merge3, 256)

    up2 = UpSampling2D(size=(2, 2))(conv7)
    merge2 = concatenate([conv2, up2], axis=3)
    conv8 = conv_block(merge2, 128)

    up1 = UpSampling2D(size=(2, 2))(conv8)
    merge1 = concatenate([conv1, up1], axis=3)
    conv9 = conv_block(merge1, 64)

    output = Conv2D(1, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs, output)
    return model


model_unet_plus_plus = unet_plus_plus()
model_unet_plus_plus.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


model_unet_plus_plus.save('my_best_model.h5')
