import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def conv_block(x, filters, kernel_size=(3, 3), padding="same", activation="relu"):
    conv = Conv2D(filters, kernel_size, padding=padding, activation=activation)(x)
    conv = Conv2D(filters, kernel_size, padding=padding, activation=activation)(conv)
    return conv

def attention_block(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, (1, 1), strides=(2, 2), padding="same")(x)
    phi_g = Conv2D(inter_channel, (1, 1), padding="same")(g)
    
    add_xg = tf.keras.layers.Add()([theta_x, phi_g])
    relu_xg = tf.keras.layers.Activation('relu')(add_xg)
    psi = Conv2D(1, (1, 1), padding="same")(relu_xg)
    sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)
    upsample_psi = UpSampling2D(size=(2, 2))(sigmoid_xg)
    upsample_psi = tf.keras.layers.Multiply()([x, upsample_psi])
    
    return upsample_psi

def attention_unet(input_shape=(256, 256, 1)):
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

   
    g4 = UpSampling2D(size=(2, 2))(conv5)
    att4 = attention_block(conv4, g4, 512)
    merge4 = concatenate([att4, g4], axis=3)
    conv6 = conv_block(merge4, 512)

    g3 = UpSampling2D(size=(2, 2))(conv6)
    att3 = attention_block(conv3, g3, 256)
    merge3 = concatenate([att3, g3], axis=3)
    conv7 = conv_block(merge3, 256)

    g2 = UpSampling2D(size=(2, 2))(conv7)
    att2 = attention_block(conv2, g2, 128)
    merge2 = concatenate([att2, g2], axis=3)
    conv8 = conv_block(merge2, 128)

    g1 = UpSampling2D(size=(2, 2))(conv8)
    att1 = attention_block(conv1, g1, 64)
    merge1 = concatenate([att1, g1], axis=3)
    conv9 = conv_block(merge1, 64)

    output = Conv2D(1, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs, output)
    return model


model_attention_unet = attention_unet()
model_attention_unet.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model_attention_unet.save('attention_unet_model.h5')
