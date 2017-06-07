from net import VGG19
from chainer import serializers
from sklearn.externals import joblib
from df_image import MyImage
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser('PCA')
parser.add_argument('-input', type=str)
parser.add_argument('-output', type=str)
parser.add_argument('-model_dump', type=str)
args = parser.parse_args()


def save_output_image(image, size, out_img):
    image = image.transpose((0, 2, 3, 1))
    image = image.reshape(image.shape[1:])[:,:,::-1]
    image = (image+128).clip(0, 255).astype(np.uint8)
    diff_image = Image.fromarray(image).resize(size, Image.BILINEAR)
    image = np.asarray(diff_image, dtype=np.int32)
    Image.fromarray(image.clip(0, 255).astype(np.uint8)).save(out_img)

device_id = 0
net = VGG19()
serializers.load_npz('vgg19.model', net)

print "Loading PCA model"
pca = joblib.load(args.model_dump)
print "PCA model loaded"

input_image_path = args.input
my_image = MyImage(input_image_path, net, device_id)
print "Image to vector"
vec = my_image.to_vector()
print "Transforming vector (PCA transform)"
my_image_trans = pca.transform([vec])
print "Creating inverse transform"
my_image_back = pca.inverse_transform(my_image_trans)

print "Generating layers"
layers_back = []
fst_list = []
snd_list = []
trd_list = []
for fst in range(0, 640000):
    fst_list.append(my_image_back[0][fst])
for snd in range(640000, 960000):
    snd_list.append(my_image_back[0][snd])
for trd in range(960000, 1046528):
    trd_list.append(my_image_back[0][trd])
layers_back.append(net.xp.asarray(np.array([fst_list], dtype=np.float32)))
layers_back.append(net.xp.asarray(np.array([snd_list], dtype=np.float32)))
layers_back.append(net.xp.asarray(np.array([trd_list], dtype=np.float32)))
print "Layers generated"

z = my_image.back_from_feature_space(layers_back, net)

save_output_image(z, my_image.original_image.size, args.output)



