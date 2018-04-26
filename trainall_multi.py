from model_all import Pose_GAN
from dataset_reader import DataLoader
import tensorflow as tf
from config import cfg
from six.moves import xrange
import os
import cv2
import datetime
import numpy as np
import scipy.misc


def transform(img):
    img = (img + 1) / 2.0
    return img[:, :, [2, 1, 0]]


def tower_loss(scope, g1_feed, conditional_image, target_image, target_morphologicals):
    """Calculate the total loss on a single tower running the CIFAR model"""
    return 0


def average_gradients(tower_gradient):
    return tf.add_n(tower_gradient)

if not os.path.exists(cfg.RESULT_DIR):
    os.makedirs(cfg.RESULT_DIR)

dataloader = DataLoader()
model = Pose_GAN()
g1_loss, g2_loss, d_loss = model.build_loss()
tf.summary.scalar("g1loss", g1_loss)

# train g1 using multi-gpu
with tf.Graph().as_default(), tf.device('/cpu:0'):
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0), trainable=False)
    # g1 optimizer
    train_g1 = tf.train.AdamOptimizer(learning_rate=0.00008, beta1=0.5, beta2=0.999)

    # # get one batch of data
    # g1_feed, conditional_image, target_image, target_morphologicals = dataloader.next_batch(cfg.BATCH_SIZE,
    #                                                                                         trainorval='TRAIN')

    # convert from np.ndarray to Tensor
    # g1_feed = tf.convert_to_tensor(g1_feed)
    # conditional_image = tf.convert_to_tensor(conditional_image)
    # target_image = tf.convert_to_tensor(target_image)
    # target_morphologicals = tf.convert_to_tensor(target_morphologicals)

    # construct a queue for data
    # batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([g1_feed, conditional_image, target_image,
    #                                                              target_morphologicals], capacity=2*cfg.BATCH_SIZE)

    # compute on each gpu and average
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(cfg.NUM_GPUS):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i) as scope:
                    # dequeue one batch for gpu
                    # define variable and ops
                    g1_feed_batch, conditional_image_batch, target_image_batch, target_morphologicals_batch = batch_queue.dequeue()
                    loss = tower_loss(scope, g1_feed_batch, conditional_image_batch, target_image_batch, target_morphologicals_batch)

                    # reuse variables for the next tower
                    tf.get_variable_scope().reuse_variables()

                    # todo: retain the summaries from the final tower

                    # Calculate grads on the current gpu
                    grads = train_g1.compute_gradients(loss)
                    tower_grads.append(grads)

    grads = average_gradients(tower_grads)

    # todo: add summaries to learning rate, histograms for gradients
    apply_gradient_op = train_g1.apply_gradients(grads, global_step=global_step)
    # todo: add summaries for trainable variables

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)

    for itr in xrange(cfg.MAXITERATION):
        start_time = time.time()
        sess.run(apply_gradient_op)
        if itr % 10 == 0:
            train_loss = sess.run(g1_loss)
            print("training loss is", train_loss, "itr", itr)

        if itr == cfg.MAXITERATION - 1 or itr % 10000 == 0:
            if itr == cfg.MAXITERATION - 1:
                print("Training of G1 done. At iteration ", itr)
