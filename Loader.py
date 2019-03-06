from __future__ import print_function
import os
import numpy as np
from keras.utils.np_utils import to_categorical
import glob
import cv2
import random
from augmenters import get_augmenter
import scipy
from utils.utils import preprocess

random.seed(os.urandom(10))

problemTypes = ['classification', 'segmentation']


class Loader:
    def __init__(self, dataFolderPath, width=224, height=224, channels=3, n_classes=21, problemType='classification',
                 ignore_label=None, median_frequency=0.00, image_weight=False):
        self.dataFolderPath = dataFolderPath
        self.height = height
        self.width = width
        self.dim = channels
        self.ignore_label = ignore_label  # label to ignore
        self.freq = np.zeros(n_classes)  # vector for calculating the class frequency
        self.index_train = 0  # indexes for iterating while training
        self.index_test = 0  # indexes for iterating while testing
        self.median_frequency_soft = median_frequency  # softener value for the median frequency balancing (if median_frequency==0, nothing is applied, if median_frequency==1, the common formula is applied)
        self.image_weight = image_weight  # for semantic segmentation. If true, a folder with weights for every iamge has to be provided (same folder level than labels and images)
        if ignore_label and ignore_label < n_classes:
            raise Exception(
                'please, change the labeling in order to put the ignore label value to the last value > nunm_classes')

        print('Reading files...')
        '''
		possible structures:
		classification: 
		dataset
				train
					class n
						image..
				test
					class n
						image..

		segmentation: 
		dataset
				train
					images
						image n..
					labels
						label n ..
					weights       [optional]
						weight n..
				test
					images
						image n..
					labels
						label n ..
					weights       [optional]
						weight n..
		'''
        # Load filepaths
        files = glob.glob(os.path.join(dataFolderPath, '*', '*', '*'))

        print('Structuring test and train files...')
        self.test_list = [file for file in files if '/test/' in file]
        self.train_list = [file for file in files if '/train/' in file]

        # Check problem type
        if problemType in problemTypes:
            self.problemType = problemType
        else:
            raise Exception('Not valid problemType')

        if problemType == 'classification':
            # The structure has to be dataset/train/class/image.png
            # Extract dictionary to map class -> label

            # Shuffle train
            s = np.arange(len(self.train_list))
            np.random.shuffle(s)
            self.train_list = np.array(self.train_list)[s]

            # Shuffle test
            s = np.arange(len(self.test_list))
            np.random.shuffle(s)
            self.test_list = np.array(self.test_list)[s]

            print('Loaded ' + str(len(self.train_list)) + ' training samples')
            print('Loaded ' + str(len(self.test_list)) + ' testing samples')
            classes_train = [file.split('/train/')[1].split('/')[0] for file in self.train_list]
            classes_test = [file.split('/test/')[1].split('/')[0] for file in self.test_list]
            classes = np.unique(np.concatenate((classes_train, classes_test)))
            self.classes = {}
            for label in range(len(classes)):
                self.classes[classes[label]] = label
            self.n_classes = len(classes)
            self.freq = np.zeros(self.n_classes)

            if self.median_frequency_soft != 0:
                self.median_freq = self.median_frequency_exp(soft=self.median_frequency_soft)
            else:
                self.median_freq = np.ones(self.n_classes)
                print(self.median_freq)
        elif problemType == 'segmentation':
            # The structure has to be dataset/train/images/image.png
            # The structure has to be dataset/train/labels/label.png
            # Separate image and label lists
            # Sort them to align labels and images

            self.image_train_list = [file for file in self.train_list if '/images/' in file]
            self.image_test_list = [file for file in self.test_list if '/images/' in file]
            self.label_train_list = [file for file in self.train_list if '/labels/' in file]
            self.label_test_list = [file for file in self.test_list if '/labels/' in file]

            if self.image_weight:
                self.weight_train_list = [file for file in self.train_list if '/weights/' in file]
                self.weight_test_list = [file for file in self.test_list if '/weights/' in file]
                self.weight_test_list.sort()
                self.weight_train_list.sort()

            self.label_test_list.sort()
            self.image_test_list.sort()
            self.label_train_list.sort()
            self.image_train_list.sort()

            # Shuffle train
            s = np.arange(len(self.image_train_list))
            np.random.shuffle(s)
            self.image_train_list = np.array(self.image_train_list)[s]
            self.label_train_list = np.array(self.label_train_list)[s]

            print('Loaded ' + str(len(self.image_train_list)) + ' training samples')
            print('Loaded ' + str(len(self.image_test_list)) + ' testing samples')
            self.n_classes = n_classes

            if self.median_frequency_soft != 0:
                self.median_freq = self.median_frequency_exp(soft=self.median_frequency_soft)

        print('Dataset contains ' + str(self.n_classes) + ' classes')

    # Returns a weighted mask from a binary mask
    def _from_binarymask_to_weighted_mask(self, labels, masks):
        '''
		the input [mask] is an array of N binary masks 0/1 of size [N, H, W ] where the 0 are pixeles to ignore from the labels [N, H, W ]
		and 1's means pixels to take into account.
		This function transofrm those 1's into a weight using the median frequency
		'''
        if self.median_frequency_soft == 0:
            return masks

        else:
            weights = self.median_freq
            for i in xrange(masks.shape[0]):
                # for every mask of the batch
                label_image = labels[i, :, :]
                mask_image = masks[i, :, :]
                dim_1 = mask_image.shape[0]
                dim_2 = mask_image.shape[1]
                label_image = np.reshape(label_image, (dim_2 * dim_1))
                mask_image = np.reshape(mask_image, (dim_2 * dim_1))

                for label_i in xrange(self.n_classes):
                    # multiply the mask so far, with the median frequency wieght of that label
                    mask_image[label_image == label_i] = mask_image[label_image == label_i] * weights[label_i]
                # unique, counts = np.unique(mask_image, return_counts=True)

                mask_image = np.reshape(mask_image, (dim_1, dim_2))
                masks[i, :, :] = mask_image

            return masks

    def _perform_augmentation_segmentation(self, img, label, mask_image, augmenter):
        seq_image_contrast, seq_image_translation, seq_label, seq_mask = get_augmenter(name=augmenter,
                                                                                       c_val=self.ignore_label)

        # apply some contrast  to de rgb image
        img = img.reshape(sum(((1,), img.shape), ()))
        img = seq_image_contrast.augment_images(img)
        img = img.reshape(img.shape[1:])

        # Apply shifts and rotations to the mask, labels and image

        # Reshapes for the AUGMENTER framework
        # the loops are due to the external library failures
        img = img.reshape(sum(((1,), img.shape), ()))
        img = seq_image_translation.augment_images(img)
        img = img.reshape(img.shape[1:])

        label = label.reshape(sum(((1,), label.shape), ()))
        label = seq_label.augment_images(label)
        label = label.reshape(label.shape[1:])

        mask_image = mask_image.reshape(sum(((1,), mask_image.shape), ()))
        mask_image = seq_mask.augment_images(mask_image)
        mask_image = mask_image.reshape(mask_image.shape[1:])

        return img, label, mask_image

    def suffle_segmentation(self):
        if self.problemType == 'segmentation':
            s = np.arange(len(self.image_train_list))
            np.random.shuffle(s)
            self.image_train_list = np.array(self.image_train_list)[s]
            self.label_train_list = np.array(self.label_train_list)[s]

    # Returns a random batch of segmentation images: X, Y, mask
    def _get_batch_segmentation(self, size=32, train=True, augmenter=None, labels_resize_factor=1):
        labels_w = int(self.width / labels_resize_factor)
        labels_h = int(self.height / labels_resize_factor)
        # init numpy arrays
        x = np.zeros([size, self.height, self.width, self.dim], dtype=np.float32)
        y = np.zeros([size, labels_h, labels_w], dtype=np.uint8)
        mask = np.ones([size, labels_h, labels_w], dtype=np.float32)

        if train:
            image_list = self.image_train_list
            label_list = self.label_train_list
            if self.image_weight: weight_list = self.weight_train_list
            # Get [size] random numbers
            indexes = [i % len(image_list) for i in range(self.index_train, self.index_train + size)]
            self.index_train = indexes[-1] + 1

        else:
            image_list = self.image_test_list
            label_list = self.label_test_list
            if self.image_weight: weight_list = self.weight_test_list

            # Get [size] random numbers
            indexes = [i % len(image_list) for i in range(self.index_test, self.index_test + size)]
            self.index_test = indexes[-1] + 1

        random_images = [image_list[number] for number in indexes]
        random_labels = [label_list[number] for number in indexes]
        if self.image_weight: random_weights = [weight_list[number] for number in indexes]

        # for every random image, get the image, label and mask.
        # the augmentation has to be done separately due to augmentation
        for index in range(size):
            if self.dim == 1:
                img = cv2.imread(random_images[index], 0)
            else:
                img = cv2.imread(random_images[index])

            '''
            # random grayscale
            if random.random() > 0.5:
                img2 = cv2.imread(random_images[index], 0)
                img[:,:,0] = img2
                img[:,:,1] = img2
                img[:,:,2] = img2
            '''


            label = cv2.imread(random_labels[index], 0)

            # Reshape images if its needed
            if img.shape[1] != self.width or img.shape[0] != self.height:
                img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

            if label.shape[1] != labels_w or label.shape[0] != labels_h:
                label = cv2.resize(label, (labels_w, labels_h), interpolation=cv2.INTER_NEAREST)

            mask_image = mask[index, :, :]

            if self.image_weight:
                mask_weight = np.load(random_weights[index])
                if mask_weight.shape[1] != self.width or mask_weight.shape[0] != self.height:
                    mask_weight = cv2.resize(mask_weight, (self.width, self.height), interpolation=cv2.INTER_AREA)
                # raise Exception('When using image weights, the dimensions of the weigths and the specify height and width must match')
                # print('WARNING: When using image weights, the dimensions of the weigths and the specify height and width must match')
                mask_image = np.squeeze(mask_weight)

            if train and augmenter:
                img, label, mask_image = self._perform_augmentation_segmentation(img, label, mask_image, augmenter)

            # modify the mask and the labels. Mask

            mask_ignore = label >= self.n_classes
            mask_image[mask_ignore] = 0  # The ignore pixels will have a value o 0 in the mask
            label[mask_ignore] = self.n_classes  # The ignore label will be n_classes

            if self.dim == 1:
                img = np.reshape(img, (img.shape[0], img.shape[1], self.dim))

            x[index, :, :, :] = img
            y[index, :, :] = label
            mask[index, :, :] = mask_image

        # Apply weights to the mask
        mask = self._from_binarymask_to_weighted_mask(y, mask)

        # the labeling to categorical (if 5 classes and value is 2:  2 -> [0,0,1,0,0])
        a, b, c = y.shape
        y = y.reshape((a * b * c))

        # Convert to categorical. Add one class for ignored pixels
        y = to_categorical(y, num_classes=self.n_classes + 1)
        y = y.reshape((a, b, c, self.n_classes + 1)).astype(np.uint8)

        # tf.keras.applications.imagenet_utils.preprocess_input(x, mode='tf')
        # x = tf.keras.applications.xception.preprocess_input(x)
        x = preprocess(x)
        return x, y, mask

    # Returns a random batch
    def _get_batch_rgb(self, size=32, train=True, augmenter=None):

        x = np.zeros([size, self.height, self.width, self.dim], dtype=np.float32)
        y = np.zeros([size], dtype=np.uint8)

        if train:
            file_list = self.train_list
            folder = '/train/'
            # Get [size] random numbers
            indexes = [i % len(file_list) for i in range(self.index_train, self.index_train + size)]
            self.index_train = indexes[-1] + 1

        else:
            file_list = self.test_list
            folder = '/test/'
            # Get [size] random numbers
            indexes = [i % len(file_list) for i in range(self.index_test, self.index_test + size)]
            self.index_test = indexes[-1] + 1

        random_files = [file_list[number] for number in indexes]
        classes = [self.classes[file.split(folder)[1].split('/')[0]] for file in random_files]

        for index in range(size):
            img = cv2.imread(random_files[index])
            if img is None:
                print(random_files[index])
                print(indexes[index])

            if img.shape[1] != self.width or img.shape[0] != self.height:
                img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

            x[index, :, :, :] = img
            y[index] = classes[index]
        # the labeling to categorical (if 5 classes and value is 2:  2 -> [0,0,1,0,0])
        y = to_categorical(y, num_classes=self.n_classes)
        # augmentation
        if augmenter:
            augmenter_seq = get_augmenter(name=augmenter)
            x = augmenter_seq.augment_images(x)

        # x = x.astype(np.float32)
        # tf.keras.applications.imagenet_utils.preprocess_input(x, mode='tf')
        # x = tf.keras.applications.xception.preprocess_input(x)
        x = x.astype(np.float32) / 255.0 - 0.5

        return x, y

    def _get_key_by_value(self, dictionary, value_searching):
        for key, value in dictionary.iteritems():
            if value == value_searching:
                return key

        return None

    # Returns a random batch
    def get_batch(self, size=32, train=True, augmenter=None, labels_resize_factor=1):
        '''
		Gets a batch of size [size]. If [train] the data will be training data, if not, test data.
		if augmenter is no None, image augmentation will be perform (see file augmenters.py)
		if images are bigger than max_size of smaller than min_size, images will be resized (forced)
		'''
        if self.problemType == 'classification':
            return self._get_batch_rgb(size=size, train=train, augmenter=augmenter)
        elif self.problemType == 'segmentation':
            return self._get_batch_segmentation(size=size, train=train, augmenter=augmenter, labels_resize_factor=labels_resize_factor)

    # Returns the median frequency for class imbalance. It can be soften with the soft value (<=1)
    def median_frequency_exp(self, soft=1):

        if self.problemType == 'classification':
            quantity = []
            for class_name in self.classes:
                path = os.path.join(self.dataFolderPath, 'train', class_name)
                class_freq = len(glob.glob(os.path.join(path, '*')))
                self.freq[self.classes[class_name]] = class_freq

        elif self.problemType == 'segmentation':
            for image_label_train in self.label_train_list:
                image = cv2.imread(image_label_train, 0)
                for label in xrange(self.n_classes):
                    self.freq[label] = self.freq[label] + sum(sum(image == label))

        # Common code
        zeros = self.freq == 0
        if sum(zeros) > 0:
            print('There are some classes which are not contained in the training samples')

        results = np.median(self.freq) / self.freq
        results[zeros] = 0  # for not inf values.
        results = np.power(results, soft)
        print(results)
        return results


if __name__ == "__main__":

    loader = Loader('./Datasets/camvid', problemType='segmentation', ignore_label=11, n_classes=11, width=480,
                    height=360, median_frequency=0.25, image_weight=True)
    # print(loader.median_frequency_exp())
    x, y, mask = loader.get_batch(size=2, augmenter='segmentation')

    for i in xrange(2):
        cv2.imshow('x', ((x[i, :, :, :] + 0.5) * 255).astype(np.uint8))
        cv2.imshow('y', (np.argmax(y, 3)[i, :, :]).astype(np.uint8))
        print(mask.shape)
        cv2.imshow('mask', (mask[i, :, :] * 70).astype(np.uint8))
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    x, y, mask = loader.get_batch(size=3, train=False)



