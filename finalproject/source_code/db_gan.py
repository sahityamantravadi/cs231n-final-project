from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
from glob import glob
import math
import matplotlib.pyplot as plt
import nibabel as nb
from nilearn import image
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
import timeit
import warnings
from data_io import Dataset_Pipeline, _get_data

EPS = 1e-12
ngf = 8 #number of generator filters in first conv layer
ndf = 16 #number of discriminator filters in first conv layer
seed = 123
lr = 0.00005 #initial learning rate for adam
beta1 = 0.5 #momentum term of adam"

def save_batch(pre, post, epoch, step):
    pre_out_base = '/home/smantra/finalproject/logs/debiasing_GAN/images/pre_epoch{}_step{}'.format(epoch, step)
    post_out_base = '/home/smantra/finalproject/logs/debiasing_GAN/images/post_epoch{}_step{}'.format(epoch, step)
    for i in range(4):
        pre_file = pre_out_base + '_img{}.nii.gz'.format(i + 1)
        post_file = post_out_base + '_img{}.nii.gz'.format(i + 1)
        pre_img = nb.Nifti1Image(pre[i,:,:,:], np.eye(4))
        nb.save(pre_img, pre_file)
        post_img = nb.Nifti1Image(post[i,:,:,:], np.eye(4))
        nb.save(post_img, post_file)
    print('Saved images for epoch {}, step {}'.format(epoch, step))

def lrelu(x, a=0.1):
    return tf.maximum(a*x, x)

def G_conv(batch_input, out_channels):
    return tf.layers.conv3d(batch_input, out_channels, kernel_size=4, strides=(2, 2, 2), padding="valid")

def D_conv(batch_input, out_channels):
    return tf.layers.conv3d(batch_input, out_channels, kernel_size=4, strides=(1, 1, 1), padding="same")

def D_max_pool(batch_input):
    return tf.layers.max_pooling3d(batch_input, 2, 2)

def G_conv_transpose(batch_input, out_channels):
    return tf.layers.conv3d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2, 2), padding="valid")

def batchnorm(batch_input):
    return tf.layers.batch_normalization(batch_input, epsilon=1e-5, momentum=0.1, training=True)

def generator(G_in, G_out_channels):
    with tf.variable_scope("generator"):
        layers = []
        G_in = tf.expand_dims(G_in, -1)
        # encoder_1
        with tf.variable_scope("encoder_1"):
            conv_out = G_conv(G_in, ngf)
            layers.append(conv_out)

        layer_nfilters = [
            ngf, # encoder_2
            ngf * 2, # encoder_3
            ngf * 4, # encoder_4
        ]

        for out_n in layer_nfilters:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.1)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = G_conv(G_in, out_n)
                output = batchnorm(convolved)
                layers.append(output)

        layer_specs = [
#            ngf * 4,   # decoder_4
#            ngf * 2,   # decoder_3
            ngf,       # decoder_2
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, out_channels in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
    #             if decoder_layer == 0:
    #                 # first decoder layer doesn't have skip connections
    #                 # since it is directly connected to the skip_layer
    #                 decoder_input = layers[-1]
    #             else:
    #                 decoder_input = tf.concat([layers[-1], layers[skip_layer]], axis=4)
                decoder_input = layers[-1]
                rectified = tf.nn.relu(decoder_input)

                decoder_output = G_conv_transpose(rectified, out_channels)

                output = batchnorm(decoder_output)
                
                layers.append(output)

        # decoder_1
#         with tf.variable_scope("decoder_1"):
#             #dec_1_input = tf.concat([layers[-1], layers[0]], axis=4)
#             dec_1_input = layers[-1]
#             rectified = tf.nn.relu(dec_1_input)
#             output = G_conv_transpose(rectified, G_out_channels)
#             output = tf.tanh(output)
#             layers.append(output)

        return layers[-1]

def site_discriminator(D_input):
    with tf.variable_scope("site_discriminator"):
        n_layers = 2
        layers = []

        #D_input = tf.concat([discrim_inputs, discrim_targets], axis=4)

        # layer_1:
#        D_input = tf.expand_dims(D_input, -1)
        with tf.variable_scope("layer_1"):
            convolved = D_conv(D_input, ndf)
            pooled = D_max_pool(convolved)
            rectified = lrelu(pooled, 0.1)
            layers.append(rectified)

        # layer_2:
        # layer_3:
        # layer_4:
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf/(2*(i+1))

                #stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = D_conv(layers[-1], out_channels)
                pooled = D_max_pool(convolved)
                normalized = batchnorm(pooled)
                rectified = lrelu(normalized, 0.1)
                layers.append(rectified)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = D_conv(rectified, out_channels=1)
            fc1 = tf.contrib.layers.flatten(convolved)
#            fc1 = tf.layers.dense(fc1, 300)
#            fc1 = lrelu(fc1)
            # Output layer, class prediction
            out = tf.layers.dense(fc1, 17)
            layers.append(out)
    #         pred_classes = tf.argmax(out, axis=1)
    #         layers.append(pred_classes)
        return layers[-1]

def qc_discriminator(D_input):
    with tf.variable_scope("qc_discriminator"):
        n_layers = 2
        layers = []

    #    D_input = tf.concat([discrim_inputs, discrim_targets], axis=4)
    #    D_input = tf.expand_dims(D_input, -1)

        # layer_1:
        with tf.variable_scope("layer_1"):
            convolved = D_conv(D_input, ndf)
            pooled = D_max_pool(convolved)
            rectified = lrelu(pooled, 0.1)
            layers.append(rectified)

        # layer_2:
        # layer_3:
        # layer_4:
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf/(2*(i+1))
                #stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = D_conv(layers[-1], out_channels)
                pooled = D_max_pool(convolved)
                normalized = batchnorm(pooled)
                rectified = lrelu(normalized, 0.1)
                layers.append(rectified)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = D_conv(rectified, out_channels=1)

            fc1 = tf.contrib.layers.flatten(convolved)
#            fc1 = tf.layers.dense(fc1, 300)
#            fc1 = lrelu(fc1)
            # Output layer, class prediction
            out = tf.layers.dense(fc1, 2)
            layers.append(out)
    #         pred_classes = tf.argmax(out, axis=1)
    #         layers.append(pred_classes)
        return layers[-1]
    
features = tf.placeholder(np.float32, [4, 106, 128, 110])
qc_labels = tf.placeholder(np.int32, [4])
site_labels = tf.placeholder(np.int32, [4])

with tf.variable_scope("generator"):
    debiased_channels = int(qc_labels.get_shape()[-1])
    debiased = generator(features, debiased_channels)

with tf.variable_scope("qc_discriminator"):
    qc_out = qc_discriminator(debiased)
    x_in = tf.identity(qc_out)

with tf.variable_scope("site_discriminator"):
    site_out = site_discriminator(debiased)

with tf.name_scope("predictions"):
    qc_preds = tf.cast(tf.argmax(qc_out, axis=1), tf.int32)
    site_preds = tf.cast(tf.argmax(site_out, axis=1), tf.int32)

with tf.name_scope("accuracy"):
    qc_acc = tf.reduce_mean(tf.cast(tf.equal(qc_labels, qc_preds), tf.float32))
    tf.summary.scalar('QC_accuracy', qc_acc)
    site_acc = tf.reduce_mean(tf.cast(tf.equal(site_labels, site_preds), tf.float32))
    tf.summary.scalar('Site_accuracy', site_acc)

qc_D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'qc_discriminator')
site_D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'site_discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 

site_D_solver = tf.train.AdamOptimizer(lr, beta1)
qc_D_solver = tf.train.AdamOptimizer(lr, beta1)
G_solver = tf.train.AdamOptimizer(lr, beta1)

with tf.name_scope("G_loss"):
    G_loss = tf.abs(tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=site_out, labels=site_labels)) - tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=qc_out, labels=qc_labels)))
    G_loss_scalar = tf.summary.scalar('G_loss', G_loss)

with tf.name_scope("QC_D_loss"):
    qc_D_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=qc_out, labels=qc_labels))
    qc_D_loss_scalar = tf.summary.scalar('QC_D_loss', qc_D_loss)

with tf.name_scope("Site_D_loss"):
    site_D_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=site_out, labels=site_labels))
    site_D_loss_scalar = tf.summary.scalar('Site_D_loss', site_D_loss)

site_D_train_step = site_D_solver.minimize(site_D_loss, var_list=site_D_vars)
qc_D_train_step = qc_D_solver.minimize(qc_D_loss, var_list=qc_D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
qc_D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'qc_discriminator')
site_D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'site_discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')

def run_a_gan(sess, G_train_step, G_loss,\
              qc_D_train_step, qc_D_loss,\
              site_D_train_step, site_D_loss,\
              G_extra_step, qc_D_extra_step, site_D_extra_step,\
              num_epoch, train_writer, test_writer):
    """Train a GAN for a certain number of epochs.
    """
    log_dir = "logs"
    current_run_subdir = os.path.join(
        "run_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    model_dir = os.path.join(log_dir, 'dbGAN', '6012018')

    ds = Dataset_Pipeline(target_shape=(106, 128, 110),
                          n_epochs=10,
                          train_src_folder="/home/smantra/finalproject/data/",
                          train_cache_prefix="/home/smantra/finalproject/cache_train/",
                          eval_src_folder="/home/smantra/finalproject/eval/",
                          eval_cache_prefix="/home/smantra/finalproject/cache_eval/",
                          batch_size=4
                         )
    train_dataset = _get_data(batch_size=ds.batch_size,
                                  src_folder=ds.train_src_folder,
                                  n_epochs=10,
                                  cache_prefix=ds.train_cache_prefix,
                                  shuffle=True,
                                  target_shape=ds.target_shape,
                                 )
    val_dataset = _get_data(batch_size=ds.batch_size,
                              src_folder=ds.eval_src_folder,
                              n_epochs=10,
                              cache_prefix=ds.eval_cache_prefix,
                              shuffle=True,
                              target_shape=ds.target_shape,
                             )


    ds_it = train_dataset.make_one_shot_iterator()
    next_batch = ds_it.get_next()

    vds_it = val_dataset.make_one_shot_iterator()
    vds_batch = vds_it.get_next()
    
    ro = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    print("Starting training:")
    
    for epoch in range(num_epoch):
        for step in range(200):
            #run a batch of data through the network
            feats, (qc_labs, site_labs) = sess.run(next_batch, options=ro)
            num_debiased_channels = int(qc_labels.get_shape()[-1])
            feed_dict={features: feats,
                       qc_labels : qc_labs,
                       site_labels : site_labs,                       
                      }
            devices = ['/gpu:0', '/gpu:1']
            for d in devices:
                with tf.device(d):
                    _, G_loss_curr, G_loss_summary  = sess.run([G_train_step, G_loss, G_loss_scalar], feed_dict=feed_dict, options=ro)
                    _, qc_D_loss_curr, qc_D_loss_summary = sess.run([qc_D_train_step, qc_D_loss, qc_D_loss_scalar], feed_dict=feed_dict, options=ro)
                    _, site_D_loss_curr, site_D_loss_summary = sess.run([site_D_train_step, site_D_loss, site_D_loss_scalar], feed_dict=feed_dict, options=ro)
                    train_writer.add_summary(G_loss_summary, epoch*200 + step)
                    train_writer.add_summary(qc_D_loss_summary, epoch*200 + step)
                    train_writer.add_summary(site_D_loss_summary, epoch*200 + step)
                    
            if (step % 50 == 0):
                vfeats, (vqc_labs, vsite_labs) = sess.run(vds_batch, options=ro)
                vfeed_dict={features: vfeats,
                   qc_labels : vqc_labs,
                   site_labels : vsite_labs,                       
                }
                summary, qc_acc_curr, site_acc_curr, gen_out = sess.run([merged, qc_acc, site_acc, debiased], feed_dict=vfeed_dict, options=ro)
                test_writer.add_summary(summary, epoch*200 + step)
                
                print('Epoch: {}, Step: {}, qc_D: {:.4}, site_D: {:.4}, G:{:.4}'.format(epoch,step,qc_D_loss_curr,site_D_loss_curr,G_loss_curr))
                print('Val Accuracy: QC {:.4}, Site {:.4}'.format(qc_acc_curr, site_acc_curr))
                save_batch(vfeats, gen_out, epoch, step)
            elif (step % 10 == 0):
                vfeats, (vqc_labs, vsite_labs) = sess.run(vds_batch, options=ro)
                vfeed_dict={features: vfeats,
                   qc_labels : vqc_labs,
                   site_labels : vsite_labs,                       
                }
                summary, qc_acc_curr, site_acc_curr = sess.run([merged, qc_acc, site_acc], feed_dict=vfeed_dict, options=ro)
                test_writer.add_summary(summary, epoch*200 + step)
                
                print('Epoch: {}, Step: {}, qc_D: {:.4}, site_D: {:.4}, G:{:.4}'.format(epoch,step,qc_D_loss_curr,site_D_loss_curr,G_loss_curr))
                print('Val Accuracy: QC {:.4}, Site {:.4}'.format(qc_acc_curr, site_acc_curr))
                    
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

with get_session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/home/smantra/finalproject/logs/debiasing_GAN/train', sess.graph)
    test_writer = tf.summary.FileWriter('/home/smantra/finalproject/logs/debiasing_GAN/test')

    sess.run(tf.global_variables_initializer(), options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
    run_a_gan(sess, G_train_step, G_loss,\
              qc_D_train_step, qc_D_loss,\
              site_D_train_step, site_D_loss,\
              G_extra_step, qc_D_extra_step, site_D_extra_step,\
              10, train_writer, test_writer)
