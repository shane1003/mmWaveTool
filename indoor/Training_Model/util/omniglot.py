"""
Loading and augmenting the Omniglot dataset.

To use these APIs, you should prepare a directory that
contains all of the alphabets from both images_background
and images_evaluation.
"""

import os
import random

from PIL import Image
import numpy as np
#测试过了没问题
def read_dataset(data_dir):
    dataset = []

    for file in os.listdir(data_dir):
        fileNameSplit = file.split(sep='_')
        BeamId_ = int(fileNameSplit[0])
        Row_ = int(fileNameSplit[1])  # row coordinate
        Column_= int(fileNameSplit[2])  # column coordinate
        RSS_ = 32.5-float(fileNameSplit[3])
        d_ = float(fileNameSplit[4].split(sep='.jpg')[0])
        image_ = np.array(Image.open(data_dir + file).convert('L')).astype('float32')
        image_.resize((40, 40,1))
        dataset.append([image_, RSS_, Row_, Column_, d_,BeamId_])

    random.shuffle(dataset)
    return dataset

