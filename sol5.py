import random

import numpy as np
import sol5_utils
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
from keras.layers import Input, Dense, Convolution2D, Activation, merge
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model

RGB_DIM = 3
GRAY_SCALE = 1
MAX_GRAY_I = 255
FILTER_BASE = np.array([1, 1])
SUBTRACT_VAL = 0.5


def read_image(filename, representation):
    '''
    Reads an image file and converts it into a given representation.
    :param filename: string containing the image filename to read.
    :param representation: representation code, either 1 or 2 defining whether the output should be a
    grayscale image (1) or an RGB image (2).
    :return: normalized image in float64 type.
    '''
    im = imread(filename)
    if im.ndim == RGB_DIM and representation == GRAY_SCALE:
        im = rgb2gray(im)
        return im
    im = im.astype(np.float64)
    im /= MAX_GRAY_I
    return im


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    '''
    Creates a Python’s generator object which outputs random tuples of the form
    (source_batch, target_batch), where each output variable is an array of shape
    (batch_size, 1, height, width), target_batch is made of clean images,
    and source_batch is their respective randomly corrupted version according to
    corruption_func(im).
    :param filenames:A list of filenames of clean images.
    :param batch_size:The size of the batch of images for each iteration of Stochastic Gradient Descent.
    :param corruption_func:A function receiving a numpy’s array representation of an image as a single argument,
    and returns a randomly corrupted version of the input image.
    :param crop_size: A tuple (height, width) specifying the crop size of the patches to extract.
    :return: random tuples of the form(source_batch, target_batch),target_batch is made of clean images,
    and source_batch is their respective randomly corrupted version according to
    corruption_func(im).
    '''
    h, w = crop_size
    exist_images = {}
    while True:
        source_batch = []
        target_batch = []
        for i in range(batch_size):
            r = np.random.randint(0, len(filenames))
            img_name = filenames[r]
            img = exist_images.get(img_name)
            if img is None:
                img = read_image(img_name, GRAY_SCALE)
                exist_images[img_name] = img
            c_img = corruption_func(img)
            r_row = random.randint(0, img.shape[0] - h - 1)
            r_col = random.randint(0, img.shape[1] - w - 1)
            patch = img[r_row:r_row + h, r_col:r_col + w]
            target_batch.append(patch.reshape((1, h, w)) - SUBTRACT_VAL)
            c_patch = c_img[r_row:r_row + h, r_col:r_col + w]
            source_batch.append(c_patch.reshape((1, h, w)) - SUBTRACT_VAL)
        source_batch = np.array(source_batch).reshape((batch_size, 1, h, w))
        target_batch = np.array(target_batch).reshape((batch_size, 1, h, w))

        yield source_batch, target_batch


def resblock(input_tensor, num_channels):
    '''
    Takes as input a symbolic input tensor and the number of channels for each of its
    convolutional layers, and returns the symbolic output tensor of the layer configuration that
    include: conv3X3 - ReLU - conv3X3 - addition(input).
    :param input_tensor: A symbolic input tensor.
    :param num_channels: Number of channels for each of the convolution layers.
    :return:symbolic output tensor of the layer configuration.
    '''
    l = Convolution2D(num_channels, 3, 3, border_mode='same')(input_tensor)
    l = Activation('relu')(l)
    l = Convolution2D(num_channels, 3, 3, border_mode='same')(l)
    output_tensor = merge([input_tensor, l], mode='sum')
    return output_tensor


def build_nn_model(height, width, num_channels, num_res_blocks):
    '''
    This function return an untrained model with input dimension the shape
    of (1, height, width).
    :param height: the height of the input.
    :param width: the width of the input.
    :param num_channels: the number of channels for each convolution layer in the model.
    :param num_res_blocks:the number of residual blocks that are part of the
    structure of the model.
    :return: untrained model.
    '''
    a = Input(shape=(1, height, width))
    b = Convolution2D(num_channels, 3, 3, border_mode='same')(a)
    b = Activation('relu')(b)
    c = resblock(b, num_channels)
    for i in range(num_res_blocks - 1):
        c = resblock(c, num_channels)
    c = merge([b, c], mode='sum')
    c = Convolution2D(1, 3, 3, border_mode='same')(c)
    model = Model(input=a, output=c)

    return model


def train_model(model, images, corruption_func, batch_size,
samples_per_epoch, num_epochs, num_valid_samples):
    '''
    Divide the images into a training set and validation set, using an 80-20 split,
    compile the given model and train it.
    :param model: a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files. You should assume these paths are complete, and
    should append anything to them.
    :param corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
    and returns a randomly corrupted version of the input image.
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param samples_per_epoch: The number of samples in each epoch.
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to test on after every epoch.
    '''
    # 80% of the input goe to training and the other to validation.
    split_pos = int(len(images) * 0.2)
    training_set = images[split_pos:]
    validation_set = images[:split_pos]

    crop_size = (model.input_shape[2], model.input_shape[3])
    train_dataset = load_dataset(training_set, batch_size, corruption_func, crop_size)
    valid_dataset = load_dataset(validation_set, batch_size, corruption_func, crop_size)

    adam_opt = Adam(beta_2=0.9)
    model.compile(loss='mean_squared_error', optimizer=adam_opt)
    model.fit_generator(train_dataset, samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
                        validation_data=valid_dataset, nb_val_samples=num_valid_samples)


def restore_image(corrupted_image, base_model):
    '''
    Creates a new model that fits the size of the input image and has the same weights as
    the given base model, then restores the given image.
    :param corrupted_image: a grayscale image of shape (height, width) and with values in the [0, 1] range of
    type float64, that is affected by a corruption generated.
    :param base_model:a neural network trained to restore small patches.
    :return: The reconstructed image.
    '''
    h, w = corrupted_image.shape
    a = Input(shape=(1, h, w))
    b = base_model(a)
    new_model = Model(input=a, output=b)
    corrupted_image = corrupted_image.reshape((1, h, w))-0.5
    result = new_model.predict(corrupted_image[np.newaxis, ...])[0]
    result = result.reshape((h, w)).astype(np.float64)+0.5
    return np.clip(result, 0, 1)


def add_gaussian_noise(image, min_sigma, max_sigma):
    '''
    Adding to every pixel of the input image a zero-mean gaussian random variable
    with standard deviation equal to sigma. Sigma is uniformly distributed between min_sigma and
    max_sigma.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param min_sigma: a non-negative scalar value representing the minimal variance of the
    gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma, representing
    the maximal variance of the gaussian distribution.
    :return: The corrupted image.
    '''
    sigma = random.uniform(min_sigma, max_sigma)
    mu = 0
    gaussian = np.random.normal(mu, sigma, image.shape)
    corrupted = image + gaussian
    corrupted = np.round((corrupted * MAX_GRAY_I), 0) / MAX_GRAY_I
    return np.clip(corrupted, 0, 1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    '''
    Return a trained denoising nn model.
    :param num_res_blocks: the number of residual blocks that are part of the
    structure of the model.
    :param quick_mode: for quick train of the model.
    :return: a trained denoising nn model.
    '''
    min_sigma = 0.0
    max_sigma = 0.2
    patches_size = 24
    channels = 48
    batch_size = 100
    samples_per_epoch = 10000
    num_epochs = 5
    num_valid_samples = 1000
    if quick_mode:
        batch_size = 10
        samples_per_epoch = 30
        num_epochs = 2
        num_valid_samples = 30

    images = sol5_utils.images_for_denoising()
    corruption_func = lambda x: add_gaussian_noise(x, min_sigma, max_sigma)
    model = build_nn_model(patches_size, patches_size, channels, num_res_blocks)
    train_model(model, images, corruption_func, batch_size, samples_per_epoch, num_epochs,
                num_valid_samples)
    return model


def add_motion_blur(image, kernel_size, angle):
    '''
    Simulates motion blur on the given image using a square kernel
    of size kernel_size with direction of motion blur of the given angle.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param kernel_size: an odd integer specifying the size of the kernel.
    :param angle: an angle in radians in the range [0, π).
    :return: corrupted image.
    '''
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    return convolve(image, kernel)


def random_motion_blur(image, list_of_kernel_sizes):
    '''
    Simulates motion blur on the given image using random parameters of level and direction of the blur.
    :param image: a grayscale image with values in the [0, 1] range of type float64.
    :param list_of_kernel_sizes: list of odd integers specifying the size of the kernel.
    :return:corrupted image.
    '''
    angle = random.uniform(0, np.pi)
    kernel_ind = random.randint(0, len(list_of_kernel_sizes)-1)
    return add_motion_blur(image, list_of_kernel_sizes[kernel_ind], angle)


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    '''
    Return a trained deblurring nn model.
    :param num_res_blocks: the number of residual blocks that are part of the
    structure of the model.
    :param quick_mode: for quick train of the model.
    :return: a trained deblurring nn model.
    '''
    patches_size = 16
    channels = 32
    batch_size = 100
    samples_per_epoch = 10000
    num_epochs = 10
    num_valid_samples = 1000
    if quick_mode:
        batch_size = 10
        samples_per_epoch = 30
        num_epochs = 2
        num_valid_samples = 30

    images = sol5_utils.images_for_deblurring()
    corruption_func = lambda x: random_motion_blur(x, [7])
    model = build_nn_model(patches_size, patches_size, channels, num_res_blocks)
    train_model(model, images, corruption_func, batch_size, samples_per_epoch, num_epochs,
                num_valid_samples)
    return model






def main():
    # model = learn_denoising_model(quick_mode=False)
    # model.save('model.h5')
    # model = load_model('deblur_models\modeld1.h5')
    # model = load_model('deno_models\model4.h5')
    model = load_model('model.h5')


    # x= model.history['loss']
    file2 = 'text_orig.png'
    file1 = 'gray_orig.png'
    file3 = 'test_dog.jpg'
    img1 = read_image(file1, 1)


    ma = 0.1
    mi = 0.11
    dis = add_gaussian_noise(img1, mi, ma)
    # dis = random_motion_blur(img1, [7])




    # plt.imshow(dis, cmap='gray')
    # plt.axis('off')
    # plt.show()

    new_img = restore_image(dis, model)
    # new_img2 = restore_image(dis, model2)

    # plt.imshow(new_img, cmap='gray')
    # plt.axis('off')
    # plt.show()

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(dis, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(new_img, cmap='gray')
    plt.axis('off')

    plt.show()
    print("test1")


if __name__ == '__main__':
    main()
