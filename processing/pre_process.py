import cv2
import torch
import numpy as np


class Rescale(object):
    def __init__(self, size: int) -> None:
        assert isinstance(size, (int, tuple))
        self.output_size = size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.output_size, self.output_size))


class ToTensor(object):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        # with rgb color
        tmp_img = np.zeros((image.shape[0], image.shape[1], 3))
        image = image / np.max(image)

        tmp_img[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmp_img[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        tmp_img[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmp_img = tmp_img.transpose((2, 0, 1))

        return tmp_img


class PreProcessing:
    def __init__(self, size: int) -> None:
        self.rescale = Rescale(size)
        self.to_tensor = ToTensor()

    def __call__(self, img: np.ndarray) -> np.ndarray:
        tensor = self.to_tensor(self.rescale(img))
        tensor = np.expand_dims(tensor, 0).astype(np.float32)
        return torch.from_numpy(tensor)
