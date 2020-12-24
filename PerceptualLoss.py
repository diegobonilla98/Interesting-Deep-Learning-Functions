import tensorflow.keras.backend as K
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
import tensorflow as tf

full_vgg = vgg19.VGG19(weights='imagenet')
full_vgg.trainable = False
for l in full_vgg.layers:
    l.trainable = False
model = Model(inputs=full_vgg.input, outputs=full_vgg.get_layer('block2_conv2').output)
model.trainable = False


@tf.function
def perceptual_loss(y_true, y_pred):
    y_pred = vgg19.preprocess_input(y_pred)
    y_true = vgg19.preprocess_input(y_true)
    y_pred_features = K.variable(model(y_pred) / 12.75)
    y_true_features = K.variable(model(y_true) / 12.75)
    return K.mean(K.square(y_true_features - y_pred_features))


