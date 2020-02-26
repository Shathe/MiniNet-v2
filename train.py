import tensorflow as tf
import random
import os
import argparse
import time
from Mininet import MiniNet2, MiniNet2_cpu
from utils.utils import get_parameters
from Loader import Loader
import math
import cv2
random.seed(os.urandom(9))


 
parser = argparse.ArgumentParser() 
parser.add_argument("--dataset", help="Dataset to train", default='./Datasets/camvid')
parser.add_argument("--init_lr", help="Initial learning rate", default=1e-3)
parser.add_argument("--min_lr", help="Final learning rate", default=1e-5)
parser.add_argument("--max_batch_size", help="batch_size", default=6)
parser.add_argument("--n_classes", help="number of classes to classify", default=11)
parser.add_argument("--ignore_label", help="class to ignore", default=11)
parser.add_argument("--epochs", help="Number of epochs to train", default=100)
parser.add_argument("--width", help="width size to load the rgb image", default=960)
parser.add_argument("--height", help="height size to load the rgb image", default=720)
parser.add_argument("--median_frequency", help="median_frequency weight for class imbalance", default=0.) 
parser.add_argument("--labels_resize_factor", help="downsample factor to apply to the label image before comparing to the CNN output", default=1)
parser.add_argument("--img_resize_factor", help="downsample factor to apply to the rgb image before feeding the CNN", default=2)
parser.add_argument("--output_resize_factor", help="resize factor to upsample the output of the CNN", default=4)
parser.add_argument("--save_model", help="Whether to save the model while training", default=1)
parser.add_argument("--checkpoint_path", help="checkpoint path", default='./weights/Mininetv2_cpu/camvid_480x360')
parser.add_argument("--train", help="if true, train, if not, test", default=0)
parser.add_argument("--cpu_version", help="Whether to use the cpu version", default=0)

args = parser.parse_args()


# Hyperparameter
median_frequency = float(args.median_frequency)
labels_resize_factor = int(args.labels_resize_factor)
img_resize_factor = int(args.img_resize_factor)
output_resize_factor = int(args.output_resize_factor)
init_learning_rate = float(args.init_lr)
min_learning_rate = float(args.min_lr)
save_model = bool(int(args.save_model))
train_or_test = bool(int(args.train))
cpu_version = bool(int(args.cpu_version))
max_batch_size = int(args.max_batch_size)
total_epochs = int(args.epochs)
width = int(args.width)
n_classes = int(args.n_classes)
ignore_label = int(args.ignore_label)
height = int(args.height)
checkpoint_path = args.checkpoint_path 


n_gpu = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)
labels_w = int(width / labels_resize_factor)
labels_h = int(height / labels_resize_factor)

loader = Loader(dataFolderPath=args.dataset, n_classes=n_classes, problemType='segmentation', width=width,
                height=height, ignore_label=ignore_label, median_frequency=median_frequency)

testing_samples = len(loader.image_test_list)
training_samples = len(loader.image_train_list)

# Placeholders
training_flag = tf.placeholder(tf.bool)
input_x = tf.placeholder(tf.float32, shape=[None, height, width, 3], name='input')
if img_resize_factor > 1:
        input_xx = tf.image.resize_bilinear(input_x, size=[input_x.shape[1] / img_resize_factor,
                                                           input_x.shape[2] / img_resize_factor],
                                            align_corners=True)
else:
    input_xx = input_x

batch_images = tf.reverse(input_x, axis=[-1])  # opencv rgb -bgr
label = tf.placeholder(tf.float32, shape=[None, labels_h, labels_w, n_classes + 1],
                       name='output')  # the n_classes + 1 is for the ignore classes
mask_label = tf.placeholder(tf.float32, shape=[None, labels_h, labels_w], name='mask')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
 
# Network
if cpu_version:
    output = MiniNet2_cpu(input_xx, n_classes, is_training=training_flag, upsampling=output_resize_factor)
else:
    output = MiniNet2(input_xx, n_classes, is_training=training_flag, upsampling=output_resize_factor)
img_out = tf.argmax(
    tf.image.resize_bilinear(output, size=[tf.shape(output)[1] , tf.shape(output)[2] ], align_corners=True), 3)

 
# Get shapes
shape_output = tf.shape(output)
label_shape = tf.shape(label)
mask_label_shape = tf.shape(mask_label)

predictions = tf.reshape(output, [shape_output[1] * shape_output[2] * shape_output[0], shape_output[3]])
labels = tf.reshape(label, [label_shape[2] * label_shape[1] * label_shape[0], label_shape[3]])
mask_labels = tf.reshape(mask_label, [mask_label_shape[1] * mask_label_shape[0] * mask_label_shape[2]])
# Last class is the ignore class
labels_ignore = labels[:, n_classes]
labels_real = labels[:, :n_classes]

cost = tf.losses.softmax_cross_entropy(labels_real, predictions, weights=mask_labels,  label_smoothing=0.0)#

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

# Different variables
restore_variables = tf.global_variables()
train_variables = tf.trainable_variables()
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

if not os.path.exists(os.path.join(checkpoint_path, 'iou')):
    os.makedirs(os.path.join(checkpoint_path, 'iou'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # get checkpoint if there is one
    print(checkpoint_path)
    ckpt2 = tf.train.get_checkpoint_state(os.path.join(checkpoint_path, 'iou'))
    if ckpt2 and tf.train.checkpoint_exists(ckpt2.model_checkpoint_path):
        print('Loading model...')
        restorer.restore(sess, ckpt2.model_checkpoint_path)
        print('Model loaded')

    if train_or_test:

        # Start variables
        batch_size = int(max_batch_size)
        global_step = 0
        best_val_loss = float('Inf')
        best_iou = float('-Inf')
        loss_acum_train = 0.
        # EPOCH  loop
        for epoch in range(total_epochs):
            # Calculate tvariables for the batch and inizialize others
            time_first = time.time()
            epoch_learning_rate = (init_learning_rate - min_learning_rate) * math.pow(1 - epoch / 1. / total_epochs,
                                                                                      0.9) + min_learning_rate
            print ("epoch " + str(epoch + 1) + ", lr: " + str(epoch_learning_rate) + ", batch_size: " + str(batch_size))

            total_steps = int(training_samples / batch_size) + 1
            show_each_steps = int(total_steps / 5)
            # steps in every epoch
            for step in range(total_steps):
                # get training data
                batch_x, batch_y, batch_mask = loader.get_batch(size=batch_size, train=True,
                                                                augmenter='segmentation',
                                                                labels_resize_factor=labels_resize_factor)  

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
                image, acc_update, miou_update, mean_acc_update, val_loss = sess.run(
                    [img_out, acc_op, conf_mat, mean_acc_op, cost],
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
            dataset_name = args.dataset
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