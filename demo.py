from model_all import Pose_GAN
from dataset_reader import DataLoader
import tensorflow as tf
from config import cfg
import os
import cv2
import datetime
import numpy as np
import scipy.misc
from dataset_reader_market import DataLoader2
# from dataset_reader import DataLoader


def transform(img):
    img = (img + 1) / 2.0
    return img[:, :, [2, 1, 0]]


dataloader = DataLoader2()  # todo: change back
model = Pose_GAN()
sess = tf.Session()

saver = tf.train.Saver(max_to_keep=2)
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(cfg.LOGDIR)

start_itr = 0
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored...")
    start_itr = int(ckpt.model_checkpoint_path.split('-')[1])
    print("starting from iteration", start_itr)

for itr in range(100):
    g1_feed, conditional_image, target_image, target_morphologicals = dataloader.next_batch(cfg.BATCH_SIZE_G2D,
                                                                                            trainorval='TRAIN')
    feed_dict = {model.g1_input: g1_feed, model.ia_input: conditional_image}

    final_output, g2_out, g1_out = sess.run([model.final_output, model.g2_output, model.g1_output], feed_dict=feed_dict)
    size = final_output.shape[0]

    # dir_name = cfg.RESULT_DIR + '/g2_iter_' + str(itr) + 'at' + str(datetime.datetime.now())
    dir_name = cfg.RESULT_DIR + '/market_g2_final_iter_0425_lambda10'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for i in range(size):
        j = i + itr * cfg.BATCH_SIZE
        print('writting result to', dir_name, '...')
        name = dir_name + '/sample' + str(j + 1) + 'finalout.jpg'
        scipy.misc.imsave(name, transform(final_output[i]))
        name = dir_name + '/sample' + str(j + 1) + 'g2out.jpg'
        scipy.misc.imsave(name, transform(g2_out[i]))
        name = dir_name + '/sample' + str(j + 1) + 'g1out.jpg'
        scipy.misc.imsave(name, transform(g1_out[i]))
        name_cond = dir_name + '/sample' + str(j + 1) + 'conditionalimg.jpg'
        scipy.misc.imsave(name_cond, transform(conditional_image[i, :, :, :]))
        name_target = dir_name + '/sample' + str(j + 1) + 'target.jpg'
        scipy.misc.imsave(name_target, transform(target_image[i, :, :, :]))
