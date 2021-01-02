import tensorflow.keras.backend as K
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
import tensorflow as tf

# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# from tensorflow.keras.backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# set_session(session)
# #
full_vgg = vgg16.VGG16(input_shape=(256, 256, 3), weights='imagenet', include_top=False)
full_vgg.trainable = False
for l in full_vgg.layers:
    l.trainable = False
model = Model(inputs=full_vgg.input, outputs=full_vgg.get_layer('block2_conv2').output)
model.trainable = False
# #
# model.summary()
#
# image_org = cv2.imread('./images/green-crested-lizard-5741470_640.jpg')
# image = cv2.resize(image_org, (256, 256))
# image = np.expand_dims(image, axis=0).astype('float32')
# image = (image - 127.5)
#
# feat = model(image)
# feat = session.run(feat)
# #
# fig, ax = plt.subplots(nrows=11, ncols=11)
# for ind in range(11 * 11):
#     ax.ravel()[ind].imshow(np.uint8(feat[0, :, :, ind] / feat[0, :, :, ind].max() * 255))
#     ax.ravel()[ind].set_axis_off()
# plt.tight_layout(pad=0.2, h_pad=0.2)
# plt.show()
#
# alls = np.sum(feat[0, :, :, :], axis=-1)
# alls = np.uint8(alls / alls.max() * 255)
# alls = cv2.cvtColor(alls, cv2.COLOR_GRAY2BGR)
#
# plt.imshow(cv2.addWeighted(cv2.resize(image_org, (112, 112)), 0.5, alls, 0.5, 0.0)[:, :, ::-1])
# plt.show()
#
# out = full_vgg.predict(image)
# label = vgg16.decode_predictions(out)
# label = label[0][0]
# print('%s (%.2f%%)' % (label[1], label[2]*100))


def perceptual_loss(y_true, y_pred):
    # images between [-127.5, 127.5]
    y_pred_features = model(y_pred)
    y_true_features = model(y_true)
    return K.mean(K.square(y_true_features - y_pred_features))


