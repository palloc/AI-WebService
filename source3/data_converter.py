# coding: utf-8
import numpy as np
from PIL import Image
import glob
from chainer.datasets import tuple_dataset


def datasetGenerator(data_pathes, channels=1):

    all_data = []

    for data_path in data_pathes:
        path = data_path[0]
        label = data_path[1]
        image_list = glob.glob(path + "*")

        for image_name in image_list:
            all_data.append([image_name, label])

    all_data = np.random.permutation(all_data)
    
    if channels == 1:
        image_data = []
        label_data = []

        for data in all_data:
            image = Image.open(data[0])
            image_data.append(np.asarray([np.float32(image) / 255.0]))
            label_data.append(np.int32(data[1]))
            
        threshold = np.int32(len(image_data) / 10 * 8)
        train = tuple_dataset.TupleDataset(image_data[:threshold], label_data[:threshold])
        test = tuple_dataset.TupleDataset(image_data[threshold:], label_data[threshold:])

    else:
        image_data = []
        label_data = []

        for data in all_data:
            image = Image.open(data[0])
            r, g, b = img.split()
            r_image_data.append(np.asarray([np.float32(r) / 255.0]))
            g_image_data.append(np.asarray([np.float32(g) / 255.0]))
            b_image_data.append(np.asarray([np.float32(b) / 255.0]))

            image_data.append(np.asarray([r_image_data, g_image_data, b_image_data]))
            label_data.append(data[1])

        threshold = np.int32(len(image_data) / 10 * 8
)


        train = tuple_dataset.TupleDataset(image_data[:threshold], label_data[:threshold])
        test = tuple_dataset.TupleDataset(image_data[threshold:], label_data[threshold:])

    return train, test
