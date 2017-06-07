from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib
import os
from df_image import MyImage
from net import VGG19
from chainer import serializers
import argparse

parser = argparse.ArgumentParser('PCA')
parser.add_argument('batch_size', type=int)
parser.add_argument('n_components', type=int)
parser.add_argument('rootdir', type=str)
parser.add_argument('max_samples', type=str)
parser.add_argument('model_dump', type=str)
args = parser.parse_args()

net = VGG19()
rootdir = args.rootdir
serializers.load_npz('vgg19.model', net)

image_paths = []
for root, subdirs, files in os.walk(rootdir):
    for f in files:
        image_paths.append(root + '/' + f)

i = 0
batch_size = args.batch_size
pca = IncrementalPCA(n_components=args.n_components, batch_size=batch_size)
batch = []
max_samples = args.max_samples
print "Start training"
for image_path in image_paths:
    if i >= max_samples:
        break
    i += 1
    print "Progress", i, "/", max_samples
    my_image = MyImage(image_path, net, -1)
    my_image_vector = my_image.to_vector()
    del my_image
    batch.append(my_image_vector)
    if len(batch) == batch_size:
        print "Fitting batch"
        pca.partial_fit(batch)
        print "Fitting batch finished"
        batch = []

print "Training finished, dumping to file"
joblib.dump(pca, args.model_dump)
