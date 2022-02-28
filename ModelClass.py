import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorboard


class ResBlock(keras.Model):
    def __init__(self, filters, downsample):
        super().__init__()
        if downsample:
            self.conv1 = layers.Conv2D(filters, 3, 2, padding='same')
            self.shortcut = keras.Sequential([
                layers.Conv2D(filters, 1, 2),
                layers.BatchNormalization()
            ])
        else:
            self.conv1 = layers.Conv2D(filters, 3, 1, padding='same')
            self.shortcut = keras.Sequential()

        self.conv2 = layers.Conv2D(filters, 3, 1, padding='same')

    def call(self, input):
        shortcut = self.shortcut(input)

        input = self.conv1(input)
        input = layers.BatchNormalization()(input)
        input = layers.ReLU()(input)

        input = self.conv2(input)
        input = layers.BatchNormalization()(input)
        input = layers.ReLU()(input)

        input = input + shortcut
        return layers.ReLU()(input)


class ResNet18(tf.keras.Model):
    def __init__(self, outputs=1000):
        super().__init__()
        self.layer0 = tf.keras.Sequential([
            layers.Conv2D(64, 7, 2, padding='same'),
            layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU()
        ], name='layer0')

        self.layer1 = tf.keras.Sequential([
            ResBlock(64, downsample=False),
            ResBlock(64, downsample=False)
        ], name='layer1')

        self.layer2 = tf.keras.Sequential([
            ResBlock(128, downsample=True),
            ResBlock(128, downsample=False)
        ], name='layer2')

        self.layer3 = tf.keras.Sequential([
            ResBlock(256, downsample=True),
            ResBlock(256, downsample=False)
        ], name='layer3')

        self.layer4 = tf.keras.Sequential([
            ResBlock(512, downsample=True),
            ResBlock(512, downsample=False)
        ], name='layer4')

        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(outputs, activation='softmax')

    def call(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = self.fc(input)

        return input


def build_FCN(image_size, name='Model'):
    """Build a 3D convolutional neural network model
    There are many ways to do this."""

    image = tf.keras.Input(shape=image_size, name='image')

    kernelsize = 3
    filters = 32

    # Convolution
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernelsize, padding='same')(image)
    # f = tf.keras.layers.MaxPool3D(pool_size=3)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Convolve again
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation='relu', padding='same')(x)
    # x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    mask = tf.keras.layers.GlobalMaxPooling2D()(x)
    mask = tf.keras.layers.Activation('softmax', name='mask')(mask)
    # TODO implement mask output
    # Flatten, Dense, Output
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(units=50, activation="relu")(x)
    # melanoma = tf.keras.layers.Dense(units=1, activation='sigmoid', name='melanoma_label')(x)
    # keratosis = tf.keras.layers.Dense(units=1, activation='sigmoid', name='keratosis_label')(x)

    # Define the model.
    model = tf.keras.Model(inputs=[image], outputs=[mask], name=name) #melanoma, keratosis], name=name)

    # Compile model
    model.compile(
        loss='categorical_crossentropy',  # tfa.losses.sigmoid_focal_crossentropy,
        optimizer='Adam',  # tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["accuracy"],
    )

    return model