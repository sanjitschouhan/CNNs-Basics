import os

try:
	import tensorflow
	print "tensorflow Found"
except ImportError:
	print "Intalling tensorflow"
	os.system('python -m pip install tensorflow')

try:
	import numpy
	print "numpy Found"
except ImportError:
	print "Intalling numpy"
	os.system('python -m pip install numpy')

try:
	import scipy
	print "scipy Found"
except ImportError:
	print "Intalling scipy"
	os.system('python -m pip install scipy')

try:
	import torch
	print "torch Found"
except ImportError:
	print "Intalling torch"
	os.system('python -m pip install torch')

try:
	import torchvision
	print "torchvision Found"
except ImportError:
	print "Intalling torchvision"
	os.system('python -m pip install torchvision')


