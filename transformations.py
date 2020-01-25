import torch
from torchvision import transforms, utils
import numpy as np
import cv2

class Normalize():
    """ convert a color image to a gray scale and normalize the color range to [0, 1]
    """
    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)
        keypoints_copy = np.copy(keypoints)

        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_copy = image_copy/255.0

        #subtract mean and divide by standard deviation
        keypoints_copy = (keypoints_copy - 100)/50.0

        return {"image" : image_copy, "keypoints" : keypoints_copy}


class ToTensor():
    """Convert ndarrays in sample to tensor
    """
    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]

        # if image has no grayscale color channel, add one
        if (len(image.shape) == 2):
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(keypoints)}


class Rescale():
    """rescale the image in a sample to a given size
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample["image"], sample["keypoints"]

        h, w = image.shape[:2]

        if isinstance(self.output_size , int):
            if h > w:
                new_h , new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h

        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))
        keypoints * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': keypoints}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}
