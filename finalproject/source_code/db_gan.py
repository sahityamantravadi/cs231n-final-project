from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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

batch_size=4
EPS = 1e-12
ngf = 32 #number of generator filters in first conv layer
ndf = 32 #number of discriminator filters in first conv layer
seed = 123
lr = 0.0005 #initial learning rate for adam
beta1 = 0.5 #momentum term of adam"
dropout_g = 0.3
dropout_d = 0.3
QC_weight = 3.0
Site_weight = 2.0

#train_dir = '/home/smantra/finalproject/logs/debiasing_GAN/train'
#test_dir = '/home/smantra/finalproject/logs/debiasing_GAN/test'
#label_file = '/home/smantra/finalproject/logs/debiasing_GAN/images/image_labels.txt'

test_dir = '/home/smantra/finalproject/logs/debiasing_GAN/test_2conv'
train_dir = '/home/smantra/finalproject/logs/debiasing_GAN/train_2conv'
label_file = '/home/smantra/finalproject/logs/debiasing_GAN/images/2conv_image_labels.txt'


def save_batch(pre, qc_lab, site_lab, post, qc_pred, site_pred, epoch, step, lab_file):
    im_base = 'epoch{}_step{}'.format(epoch, step)
    pre_out_base = '/home/smantra/finalproject/logs/debiasing_GAN/images/pre_{}'.format(im_base)
    post_out_base = '/home/smantra/finalproject/logs/debiasing_GAN/images/post_{}'.format(im_base)
    for i in range(batch_size):
        pre_file = pre_out_base + '_img{}.nii.gz'.format(i + 1)
        post_file = post_out_base + '_img{}.nii.gz'.format(i + 1)
        pre_img = nb.Nifti1Image(pre[i,:,:,:], np.eye(4))
        nb.save(pre_img, pre_file)
        post_img = nb.Nifti1Image(post[i,:,:,:], np.eye(4))
        nb.save(post_img, post_file)
        
        with open(lab_file, "a") as lf:
            imnum_base = '{}_img{}'.format(im_base, i)
            lf.write(imnum_base + '\t' + str(qc_lab[i]) + '\t' + str(site_lab[i]) + '\t' + str(qc_pred[i]) + '\t' + str(site_pred[i]) + '\n')
    print('Saved images for epoch {}, step {}'.format(epoch, step))

def lrelu(x, a=0.1):
    return tf.maximum(a*x, x)

def G_conv(batch_input, out_channels, val=True):
    pad = 'valid' if val else 'same'
    return tf.layers.conv3d(batch_input, out_channels, kernel_size=4, strides=(2, 2, 2), padding=pad)

def D_conv(batch_input, out_channels, val=False):
    pad = 'valid' if val else 'same'
    return tf.layers.conv3d(batch_input, out_channels, kernel_size=4, strides=1, padding=pad)

def D_max_pool(batch_input):
    return tf.layers.max_pooling3d(batch_input, 2, 2)

def G_conv_transpose(batch_input, out_channels, val=True):
    pad = 'valid' if val else 'same'
    return tf.layers.conv3d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2, 2), padding=pad)

def batchnorm(batch_input, is_training):
    return tf.layers.batch_normalization(batch_input, epsilon=1e-5, momentum=0.1, training=is_training)

def generator(G_in, training):
    with tf.variable_scope("generator"):
        layers = []
        G_in = tf.expand_dims(G_in, -1)
        layers.append(G_in)
        """
        # encoder_1
        with tf.variable_scope("encoder"):
            conv_out = G_conv(G_in, ngf)
            #rectified = lrelu(conv_out, 0.1)
            layers.append(conv_out)
        """
        with tf.variable_scope("conv"):
            output = D_conv(layers[-1], ngf/2)
            #rectified = tf.tanh(output)
            rectified = lrelu(output, 0.1)
            layers.append(rectified)
            """
            output1 = D_conv(layers[-1], ngf/4, False)
            #rectified1 = lrelu(output1, 0.1)
            rectified1 = tf.tanh(output1)
            normalized1 = tf.cond(training, lambda : batchnorm(rectified1, True), lambda : batchnorm(rectified1, False))
            layers.append(normalized1)
            """
            output2 = D_conv(layers[-1], 1, False)
            rectified2 = lrelu(output2, 0.1)
            #rectified2 = tf.tanh(output2)
            normalized2 = tf.cond(training, lambda : batchnorm(rectified2, True), lambda : batchnorm(rectified2, False))
            kept2 = tf.cond(training, lambda : tf.nn.dropout(normalized2, keep_prob=1 - dropout_g), lambda : normalized2)
            layers.append(kept2)
        """
        with tf.variable_scope("decoder"):
            output = G_conv_transpose(layers[-1], 1)
            output = tf.tanh(output)
            layers.append(output)
        """
        squeezed = tf.squeeze(layers[-1])
        layers.append(squeezed)
        print(layers[-1].get_shape())
        return layers[-1]

def site_discriminator(D_input, training):
    with tf.variable_scope("site_discriminator"):
        n_layers = 2
        layers = []

        # layer_1:
        D_input = tf.expand_dims(D_input, -1)
        with tf.variable_scope("layer_1"):
            convolved = D_conv(D_input, ndf)
            pooled = D_max_pool(convolved)
            rectified = lrelu(convolved, 0.1)
            layers.append(rectified)

        # layer_2:
        # layer_3:
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf/(2*(i+1))
                convolved = D_conv(layers[-1], out_channels)
                pooled = D_max_pool(convolved)
                #normalized = batchnorm(pooled, training)
                normalized = tf.cond(training, lambda : batchnorm(pooled, True), lambda : batchnorm(pooled, False))
                out = lrelu(normalized, 0.1)
                kept = tf.cond(training, lambda : tf.nn.dropout(out, keep_prob=1 - dropout_d), lambda : out)
                #if training:
                #    out = tf.nn.dropout(out, keep_prob=1 - dropout_d)
                layers.append(kept)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = D_conv(layers[-1], out_channels=1)
            pooled = D_max_pool(convolved)
            fc1 = tf.contrib.layers.flatten(pooled)
#            fc1 = tf.layers.dense(fc1, 100)
#            fc1 = lrelu(fc1)
            kept = tf.cond(training, lambda : tf.nn.dropout(fc1, 1 - dropout_d), lambda : fc1)
            #if training:
            #    fc1 = tf.nn.dropout(fc1, keep_prob=1 - dropout_d)
            out = tf.layers.dense(kept, 17)
            layers.append(out)
        return layers[-1]

def qc_discriminator(D_input, training):
    with tf.variable_scope("qc_discriminator"):
        n_layers = 2
        layers = []

        D_input = tf.expand_dims(D_input, -1)

        # layer_1:
        with tf.variable_scope("layer_1"):
            convolved = D_conv(D_input, ndf)
            pooled = D_max_pool(convolved)
            rectified = lrelu(convolved, 0.1)
            layers.append(rectified)

        # layer_2:
        # layer_3:
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf/(2*(i+1))
                convolved = D_conv(layers[-1], out_channels)
                pooled = D_max_pool(convolved)
                #normalized = batchnorm(pooled, training)
                normalized = tf.cond(training, lambda : batchnorm(pooled, True), lambda : batchnorm(pooled, False))
                out = lrelu(normalized, 0.1)
                kept = tf.cond(training, lambda : tf.nn.dropout(out, keep_prob=1 - dropout_d), lambda : out)
                #if training:
                #    out = tf.nn.dropout(out, keep_prob=1 - dropout_d)
                layers.append(kept)

        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = D_conv(layers[-1], out_channels=1)
            pooled = D_max_pool(convolved)
            fc1 = tf.contrib.layers.flatten(pooled)
#            fc1 = tf.layers.dense(fc1, 100)
#            fc1 = lrelu(fc1)
            kept = tf.cond(training, lambda : tf.nn.dropout(fc1, 1 - dropout_d), lambda : fc1)
            #if training:
            #    fc1 = tf.nn.dropout(fc1, keep_prob=1 - dropout_d)
            out = tf.layers.dense(kept, 2)
            layers.append(out)
        return layers[-1]
    
features = tf.placeholder(np.float32, [batch_size, 106, 128, 110])
qc_labels = tf.placeholder(np.int32, [batch_size])
site_labels = tf.placeholder(np.int32, [batch_size])
training = tf.placeholder(tf.bool, name='training')

with tf.variable_scope("generator"):
    debiased = generator(features, training)

with tf.variable_scope("qc_discriminator"):
    qc_out = qc_discriminator(debiased, training)

with tf.variable_scope("site_discriminator"):
    site_out = site_discriminator(debiased, training)

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
    G_loss = Site_weight * tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=site_out, labels=site_labels)) - QC_weight * tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=qc_out, labels=qc_labels))
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
    train_ds = Dataset_Pipeline(target_shape=(106, 128, 110),
                                n_epochs=num_epoch,
                                train_src_folder="/home/smantra/finalproject/data/",
                                train_cache_prefix="/home/smantra/finalproject/cache_train/",
                                eval_src_folder="/home/smantra/finalproject/eval/",
                                eval_cache_prefix="/home/smantra/finalproject/cache_eval/",
                                batch_size=batch_size
                               )
    val_ds = Dataset_Pipeline(target_shape=(106, 128, 110),
                                n_epochs=num_epoch,
                                train_src_folder="/home/smantra/finalproject/data/",
                                train_cache_prefix="/home/smantra/finalproject/cache_train/",
                                eval_src_folder="/home/smantra/finalproject/eval/",
                                eval_cache_prefix="/home/smantra/finalproject/cache_eval/",
                                batch_size=batch_size
                         )
    train_dataset = _get_data(batch_size=train_ds.batch_size,
                              src_folder=train_ds.train_src_folder,
                              n_epochs=num_epoch,
                              cache_prefix=train_ds.train_cache_prefix,
                              shuffle=True,
                              target_shape=train_ds.target_shape,
                              balance_dataset=True
                             )
    val_dataset = _get_data(batch_size=val_ds.batch_size,
                            src_folder=val_ds.eval_src_folder,
                            n_epochs=1,
                            cache_prefix=val_ds.eval_cache_prefix,
                            shuffle=True,
                            target_shape=val_ds.target_shape,
                            balance_dataset=False
                           )


    ds_it = train_dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, ds_it.initializer)
    sess.run(ds_it.initializer)
    next_batch = ds_it.get_next()

    vds_it = val_dataset.make_one_shot_iterator()
    vds_batch = vds_it.get_next()
    #vds_it = val_dataset.make_initializable_iterator()
    #tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, vds_it.initializer)
    #sess.run(vds_it.initializer)
    #vds_batch = vds_it.get_next()

    ro = tf.RunOptions(report_tensor_allocations_upon_oom=True)
    print("Starting training:")
    
    for epoch in range(num_epoch):
        for step in range(math.floor(801/batch_size)):
            #run a batch of data through the network
            feats, (qc_labs, site_labs) = sess.run(next_batch, options=ro)
            num_debiased_channels = int(qc_labels.get_shape()[-1])
            feed_dict={features: feats,
                       qc_labels : qc_labs,
                       site_labels : site_labs,
                       training : True
                      }
            devices = ['/gpu:0', '/gpu:1']
            with tf.device(devices[0]):
                _, G_loss_curr, G_loss_summary  = sess.run([G_train_step, G_loss, G_loss_scalar], feed_dict=feed_dict, options=ro)
                _, qc_D_loss_curr, qc_D_loss_summary = sess.run([qc_D_train_step, qc_D_loss, qc_D_loss_scalar], feed_dict=feed_dict, options=ro)
            with tf.device(devices[1]):
                _, site_D_loss_curr, site_D_loss_summary = sess.run([site_D_train_step, site_D_loss, site_D_loss_scalar], feed_dict=feed_dict, options=ro)
            train_writer.add_summary(G_loss_summary, epoch*200 + step)
            train_writer.add_summary(qc_D_loss_summary, epoch*200 + step)
            train_writer.add_summary(site_D_loss_summary, epoch*200 + step)
                    
            if (step % 50 == 0):
                vfeats, (vqc_labs, vsite_labs) = sess.run(vds_batch, options=ro)
                vfeed_dict={features: vfeats,
                   qc_labels : vqc_labs,
                   site_labels : vsite_labs,
                   training : False
                }
                summary, qc_acc_curr, site_acc_curr, gen_out, qc_p_curr, site_p_curr = sess.run([merged, qc_acc, site_acc, debiased, qc_preds, site_preds], feed_dict=vfeed_dict, options=ro)
                test_writer.add_summary(summary, epoch*200 + step)
                
                print('Epoch: {}, Step: {}, qc_D: {:.4}, site_D: {:.4}, G:{:.4}'.format(epoch,step,qc_D_loss_curr,site_D_loss_curr,G_loss_curr))
                print('Val Accuracy: QC {:.4}, Site {:.4}'.format(qc_acc_curr, site_acc_curr))
                print('QC Labels: {}, QC Predictions: {}'.format(vqc_labs, qc_p_curr))
                print('Site Labels: {}, Site Predictions: {}'.format(vsite_labs, site_p_curr))

                save_batch(vfeats, vqc_labs, vsite_labs, gen_out, qc_p_curr, site_p_curr, epoch, step, label_file)
            elif (step % 10 == 0):
                vfeats, (vqc_labs, vsite_labs) = sess.run(vds_batch, options=ro)
                vfeed_dict={features: vfeats,
                   qc_labels : vqc_labs,
                   site_labels : vsite_labs,
                   training : False
                }
                summary, qc_acc_curr, site_acc_curr, qc_p_curr, site_p_curr = sess.run([merged, qc_acc, site_acc, qc_preds, site_preds], feed_dict=vfeed_dict, options=ro)
                test_writer.add_summary(summary, epoch*200 + step)
                
                print('Epoch: {}, Step: {}, qc_D: {:.4}, site_D: {:.4}, G:{:.4}'.format(epoch,step,qc_D_loss_curr,site_D_loss_curr,G_loss_curr))
                print('Val Accuracy: QC {:.4}, Site {:.4}'.format(qc_acc_curr, site_acc_curr))
                print('QC Labels: {}, QC Predictions: {}'.format(vqc_labs, qc_p_curr))
                print('Site Labels: {}, Site Predictions: {}'.format(vsite_labs, site_p_curr))

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

with get_session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(train_dir, sess.graph)
    test_writer = tf.summary.FileWriter(test_dir)

    sess.run(tf.global_variables_initializer(), options=tf.RunOptions(report_tensor_allocations_upon_oom=True))
    run_a_gan(sess, G_train_step, G_loss,\
              qc_D_train_step, qc_D_loss,\
              site_D_train_step, site_D_loss,\
              G_extra_step, qc_D_extra_step, site_D_extra_step,\
              10, train_writer, test_writer)
