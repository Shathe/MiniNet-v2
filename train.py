import tensorflow as tf
import random
import os
import argparse
import time
from Mininet import MiniNet2, MiniNet
from utils.utils import get_parameters
from Loader import Loader
import math
import cv2
random.seed(os.urandom(9))

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset path", default='Datasets/camvid')
parser.add_argument("--augmentation", help="Whether to perform Image augmentation", default=1)
parser.add_argument("--init_lr", help="Initial learning rate", default=8e-4)#2e-3
parser.add_argument("--min_lr", help="Initial learning rate", default=1e-8)
parser.add_argument("--batch_size", help="batch_size (lower it if you have memory issues)", default=8)
parser.add_argument("--n_classes", help="number of classes to classify", default=11)
parser.add_argument("--epochs", help="Number of epochs to train", default=1500)
parser.add_argument("--width", help="width", default=512)
parser.add_argument("--height", help="height", default=256)
parser.add_argument("--save_model", help="save_model", default=1)
parser.add_argument("--checkpoint_path", help="checkpoint path", default='./models/camvid/')
parser.add_argument("--train", help="if true, train, if not, test", default=1)
parser.add_argument("--mininet_version", help="select mininet version 1 or 2", default=1)
args = parser.parse_args()





# Hyperparameter learning
init_learning_rate = float(args.init_lr)
power_lr = 0.9
min_learning_rate = float(args.min_lr)
augmentation = bool(int(args.augmentation))
save_model = bool(int(args.save_model))
train_or_test = bool(int(args.train))
batch_size = int(args.batch_size)
total_epochs = int(args.epochs)

# Other hyperparameters
width = int(args.width)
n_classes = int(args.n_classes)
height = int(args.height)
channels = 3
checkpoint_path = args.checkpoint_path

if int(args.mininet_version) == 1:
    print('This network was designed for 512x256 resolution')
    labels_resize_factor = 1
else:
    print('This network was designed for 1024x512 resolution')
    labels_resize_factor = 2

labels_w = int(width / labels_resize_factor)
labels_h = int(height / labels_resize_factor)

augmenter = None
if augmentation:
    augmenter = 'segmentation'

# for testing, batch_size = 1
if not train_or_test:
    batch_size = 1



# selct device
n_gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)


# Initialize Data Loader
loader = Loader(dataFolderPath=args.dataset, n_classes=n_classes, problemType='segmentation', width=width,
                height=height, ignore_label=n_classes, median_frequency=0.00)

testing_samples = len(loader.image_test_list)
training_samples = len(loader.image_train_list)



# Tensorflow Placeholders (similar to varaible initializer)
training_flag = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')

input_x = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='input')

label = tf.placeholder(tf.float32, shape=[None, labels_h, labels_w, n_classes + 1],
                       name='output')  # the n_classes + 1 is for the ignore classes
mask_label = tf.placeholder(tf.float32, shape=[None, labels_h, labels_w], name='mask')

# Initialize neural network
if int(args.mininet_version) == 1:
    output = MiniNet(input_x, n_classes, training=training_flag)
else:
    output = MiniNet2(input_x, n_classes, is_training=training_flag, upsampling=1)


# Get shapes
shape_output = tf.shape(output)
label_shape = tf.shape(label)
mask_label_shape = tf.shape(mask_label)


# FLATTEN placeholders
predictions = tf.reshape(output, [shape_output[1] * shape_output[2] * shape_output[0], shape_output[3]])
labels = tf.reshape(label, [label_shape[2] * label_shape[1] * label_shape[0], label_shape[3]])
mask_labels = tf.reshape(mask_label, [mask_label_shape[1] * mask_label_shape[0] * mask_label_shape[2]])

# Last class is the ignore class
labels_ignore = labels[:, n_classes]
labels_real = labels[:, :n_classes]

# LOSS
cost = tf.losses.softmax_cross_entropy(labels_real, predictions, weights=mask_labels)


# Metrics
labels = tf.argmax(labels, 1)
predictions = tf.argmax(predictions, 1)

indices = tf.squeeze(tf.where(tf.less_equal(labels, n_classes - 1)))  # ignore all labels >= num_classes
labels = tf.cast(tf.gather(labels, indices), tf.int64)
predictions = tf.gather(predictions, indices)

correct_prediction = tf.cast(tf.equal(labels, predictions), tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

acc, acc_op = tf.metrics.accuracy(labels, predictions)
mean_acc, mean_acc_op = tf.metrics.mean_per_class_accuracy(labels, predictions, n_classes)
iou, conf_mat = tf.metrics.mean_iou(labels, predictions, n_classes)
conf_matrix_all = tf.confusion_matrix(labels, predictions, num_classes=n_classes)

# Different variables lits
restore_variables = tf.global_variables()  # [var for var in tf.global_variables() if 'up3' not in var.name]  # Change name
train_variables = tf.trainable_variables()  # [var for var in tf.trainable_variables() if 'up' in var.name or 'm8' in var.name]
stream_vars = [i for i in tf.local_variables() if
               'count' in i.name or 'confusion_matrix' in i.name or 'total' in i.name]

# Count parameters
get_parameters()

# For batch norm
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # Uso el optimizador de Adam y se quiere minimizar la funcion de coste
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)  # adamOptimizer does not need lr decay
    train = optimizer.minimize(cost, var_list=train_variables)  # VARIABLES TO OPTIMIZE



saver = tf.train.Saver(tf.global_variables())
restorer = tf.train.Saver(restore_variables)

if not os.path.exists(checkpoint_path + 'iou'):
    os.makedirs(checkpoint_path + 'iou')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # get checkpoint if there is one
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    ckpt2 = tf.train.get_checkpoint_state(checkpoint_path + 'iou')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Loading model...')
        restorer.restore(sess, ckpt2.model_checkpoint_path)
        print('Model loaded')

    if train_or_test:

        # Start variables

        global_step = 0
        best_val_loss = float('Inf')
        best_iou = float('-Inf')
        loss_acum_train = 0.
        # EPOCH  loop
        for epoch in range(total_epochs):
            # Calculate tvariables for the batch and inizialize others
            time_first = time.time()
            epoch_learning_rate = (init_learning_rate - min_learning_rate) * math.pow(1 - epoch / 1. / total_epochs,
                                                                                      power_lr) + min_learning_rate
            print ("epoch " + str(epoch + 1) + ", lr: " + str(epoch_learning_rate) + ", batch_size: " + str(batch_size))

            total_steps = int(training_samples / batch_size) + 1
            show_each_steps = int(total_steps / 5)
            # steps in every epoch
            for step in range(total_steps):
                # get training data
                batch_x, batch_y, batch_mask = loader.get_batch(size=batch_size, train=True,
                                                                augmenter=augmenter,
                                                                labels_resize_factor=labels_resize_factor)  # , augmenter='segmentation'

                train_feed_dict = {
                    input_x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    mask_label: batch_mask,
                    training_flag: True
                }
                _, loss = sess.run([train, cost], feed_dict=train_feed_dict)
                loss_acum_train += loss

                if (step + 1) % show_each_steps == 0:
                    print("Step:", step, "Loss:", loss_acum_train / show_each_steps)
                    loss_acum_train = 0.

            # TEST
            loss_acum = 0.0
            for i in xrange(0, testing_samples):
                x_test, y_test, mask_test = loader.get_batch(size=1, train=False,
                                                             labels_resize_factor=labels_resize_factor)
                test_feed_dict = {
                    input_x: x_test,
                    label: y_test,
                    mask_label: mask_test,
                    learning_rate: 0,
                    training_flag: False
                }
                acc_update, miou_update, mean_acc_update, val_loss = sess.run(
                    [acc_op, conf_mat, mean_acc_op, cost],
                    feed_dict=test_feed_dict)
                acc_total, miou_total, mean_acc_total = sess.run([acc, iou, mean_acc], feed_dict=test_feed_dict)

                loss_acum = loss_acum + val_loss

            print("TEST")
            print("Accuracy: " + str(acc_total))
            print("miou: " + str(miou_total))
            print("mean accuracy: " + str(mean_acc_total))
            print("loss: " + str(loss_acum / testing_samples))

            # save models
            if save_model and best_iou < miou_total:
                best_iou = miou_total
                saver.save(sess=sess, save_path=checkpoint_path + 'iou/model.ckpt')
            if save_model and best_val_loss > loss_acum / testing_samples:
                best_val_loss = loss_acum / testing_samples
                saver.save(sess=sess, save_path=checkpoint_path + 'model.ckpt')

            saver.save(sess=sess, save_path=checkpoint_path + 'modellast.ckpt')

            sess.run(tf.variables_initializer(stream_vars))
            loader.suffle_segmentation()  # sheffle trainign set
            # show tiem to finish training
            time_second = time.time()
            epochs_left = total_epochs - epoch - 1
            segundos_per_epoch = time_second - time_first
            print(str(segundos_per_epoch * epochs_left) + ' seconds to end the training. Hours: ' + str(
                segundos_per_epoch * epochs_left / 3600.0))



    else:

        # TEST
        loss_acum = 0.0
        matrix_confusion = None
        list = loader.image_test_list

        # visual image placeholder
        img_out = tf.argmax(
            tf.image.resize_bilinear(output, size=[tf.shape(output)[1], tf.shape(output)[2]], align_corners=True), 3)

        for i in xrange(0, testing_samples):
            x_test, y_test, mask_test = loader.get_batch(size=1, train=False, labels_resize_factor=labels_resize_factor)
            test_feed_dict = {
                input_x: x_test,
                label: y_test,
                mask_label: mask_test,
                learning_rate: 0,
                training_flag: False
            }
            image, acc_update, miou_update, mean_acc_update, val_loss = sess.run(
                [img_out, acc_op, conf_mat, mean_acc_op, cost],
                feed_dict=test_feed_dict)
            acc_total, miou_total, mean_acc_total, matrix_conf = sess.run([acc, iou, mean_acc, conf_matrix_all],
                                                                          feed_dict=test_feed_dict)

            output_dir = 'out_dir/'
            dataset_name = args.dataset[:-1].split('/')[-1]
            out_dir = os.path.join(output_dir, dataset_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            index = loader.index_test
            name_split = list[index - 1].split('/')
            name = name_split[-1].replace('.jpg', '.png').replace('.jpeg', '.png')
            image = image[0, :, :]
            cv2.imwrite(os.path.join(out_dir, name), image)

            if i == 0:
                matrix_confusion = matrix_conf
            else:
                matrix_confusion += matrix_conf

            loss_acum = loss_acum + val_loss

        print("TEST")
        print("Accuracy: " + str(acc_total))
        print("miou: " + str(miou_total))
        print("mean accuracy: " + str(mean_acc_total))
        print("loss: " + str(loss_acum / testing_samples))
        print('matrix_conf')
        print(matrix_confusion)