from PIL import Image
from net import VGG19
import numpy as np
from chainer import functions as F
from chainer import cuda
from lbfgs import LBFGS
import chainer
import six

input_image_size = (200, 200)
clip_rect = (40, 20, 210, 190)


def feature(net, x, layers=['3_1', '4_1', '5_1']):
    y = net(x)
    out = [F.reshape(y[layer], (y[layer].shape[0], -1)) for layer in layers]
    return out


def preprocess_image(image, image_size, clip_rect=None):
    if clip_rect is not None:
        image = image.crop(clip_rect)
    image = image.resize(image_size, Image.BILINEAR)
    x = np.asarray(image, dtype=np.float32)
    return VGG19.preprocess(x, input_type='RGB')


def postprocess_image(original_image, diff):
    diff = diff.transpose((0, 2, 3, 1))
    diff = diff.reshape(diff.shape[1:])[:,:,::-1]
    diff = (diff + 128).clip(0, 255).astype(np.uint8)
    diff_image = Image.fromarray(diff).resize(original_image.size, Image.BILINEAR)
    image = np.asarray(original_image, dtype=np.int32) + np.asarray(diff_image, dtype=np.int32) - 128
    return Image.fromarray(image.clip(0, 255).astype(np.uint8))


def adjust_color_distribution(x, mean, std):
    m = np.mean(x, axis=(2, 3), keepdims=True)
    s = np.std(x, axis=(2, 3), keepdims=True)
    return (x - m) / s * std + mean


def normalized_diff(s, t):
    xp = cuda.get_array_module(s)
    w = t - s
    norm = xp.asarray(np.linalg.norm(cuda.to_cpu(w), axis=1, keepdims=True))
    return w / norm


def total_variation(x):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.shape
    wh = xp.asarray([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [[1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]], dtype=np.float32)
    ww = xp.asarray([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [[1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]], dtype=np.float32)
    return (F.sum(F.convolution_2d(x, W=wh) ** 2) + F.sum(F.convolution_2d(x, W=ww) ** 2)) / np.prod(x.shape, dtype=np.float32)


def find_nearest(xs, t):
    return min(xs, key=lambda x: np.linalg.norm(x - t))


def save_output_image(image, size, out_img):
    image = image.transpose((0, 2, 3, 1))
    image = image.reshape(image.shape[1:])[:,:,::-1]
    image = (image+128).clip(0, 255).astype(np.uint8)
    diff_image = Image.fromarray(image).resize(size, Image.BILINEAR)
    image = np.asarray(diff_image, dtype=np.int32)
    Image.fromarray(image.clip(0, 255).astype(np.uint8)).save(out_img)


def update(net, optimizer, link, target_layers, tv_weight=0.001):
    layers = feature(net, link.x)
    total_loss = 0
    losses = []
    for layer, target in zip(layers, target_layers):
        loss = F.mean_squared_error(layer, target)
        losses.append(float(loss.data))
        total_loss += loss
    tv_loss = tv_weight * total_variation(link.x)
    losses.append(float(tv_loss.data))
    total_loss += tv_loss
    link.cleargrads()
    total_loss.backward()
    optimizer.update()
    return losses


class MyImage():
    def __init__(self, image_path, net, device_id):
        self.device_id = device_id
        self.input_clip_rect = None
        self.image_path = image_path
        self.original_image = Image.open(image_path).convert('RGB')
        if self.input_clip_rect is not None:
            self.original_image = self.original_image.crop(self.input_clip_rect)
        self.image = preprocess_image(self.original_image, input_image_size)
        if device_id >= 0:
            net.to_gpu(device_id)
        xp = net.xp
        self.x = xp.asarray(self.image)
        self.org_layers = feature(net, self.x)
        self.org_layers = [layer.data for layer in self.org_layers]
        self.image_mean = np.mean(self.image, axis=(2, 3), keepdims=True)
        self.image_std = np.std(self.image, axis=(2, 3), keepdims=True)

    def back_from_feature_space(self, layers, net):
        residuals = []
        xp = net.xp
        initial_x = xp.random.uniform(-10, 10, self.x.shape).astype(np.float32)
        print('Calculating residuals')
        link = chainer.Link(x=self.x.shape)
        if self.device_id >= 0:
            link.to_gpu(self.device_id)
        link.x.data[...] = initial_x
        optimizer = LBFGS(1, size=5)
        optimizer.setup(link)

        # for j in six.moves.range(600):
        #     print "Iteration", j
        #     losses = update(net, optimizer, link, layers, 100.0)
        #     if (j + 1) % 20 == 0:
        #         z = cuda.to_cpu(link.x.data)
        #         z = adjust_color_distribution(z, self.image_mean, self.image_std)
        #         residuals.append(z - self.image)

        link = chainer.Link(x=self.x.shape)
        if self.device_id >= 0:
            link.to_gpu(self.device_id)
        link.x.data[...] = initial_x

        optimizer = LBFGS(1, size=5)
        optimizer.setup(link)
        for j in six.moves.range(1000):
            print "Iteration", j
            update(net, optimizer, link, layers, 100.0)
        z = cuda.to_cpu(link.x.data)
        # z = adjust_color_distribution(z, self.image_mean, self.image_std)
        # z -= find_nearest(residuals, z - self.image)
        return z

    def to_vector(self):
        return [l for layer in self.org_layers for l in layer[0]]

