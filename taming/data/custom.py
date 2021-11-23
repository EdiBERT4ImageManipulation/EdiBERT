import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example


class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class CustomCompletionBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        example["masked_image"] = self.data_masked[i]["image"]
        #print(self.data[i]['file_path_'],self.data_masked[i]['file_path_'])
        return example


class CustomCompletionTrain(CustomCompletionBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        masked_paths = []
        for i in range(len(paths)):
            path_i = paths[i]
            masked_paths.append(path_i.replace('celeba_cropped','celeba_cropped_masked'))
        self.data_masked = ImagePaths(paths=masked_paths, size=size, random_crop=False)


class CustomCompletionTest(CustomCompletionBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        masked_paths = []
        for i in range(len(paths)):
            path_i = paths[i]
            masked_paths.append(path_i.replace('celeba_cropped','celeba_cropped_masked'))
        self.data_masked = ImagePaths(paths=masked_paths, size=size, random_crop=False)

class CustomRandomCompletionTrain(CustomCompletionBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        masked_paths = []
        for i in range(len(paths)):
            path_i = paths[i]
            masked_paths.append(path_i.replace('celeba_cropped','celeba_cropped_non_rect_masked'))
        self.data_masked = ImagePaths(paths=masked_paths, size=size, random_crop=False)


class CustomRandomCompletionTest(CustomCompletionBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        masked_paths = []
        for i in range(len(paths)):
            path_i = paths[i]
            masked_paths.append(path_i.replace('celeba_cropped','celeba_cropped_non_rect_masked'))
        self.data_masked = ImagePaths(paths=masked_paths, size=size, random_crop=False)
