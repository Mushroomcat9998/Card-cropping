import os
import cv2
import glob
import time
import torch
import argparse
import numpy as np
from imutils import perspective

from model import U2NET
from processing.pre_process import PreProcessing
from processing.post_process import PostProcessing


class CardCrop:
    def __init__(self, path_model: str, size: int = 320) -> None:
        self.size = size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.pre_process = PreProcessing(size=size)
        self.post_process = PostProcessing()

        self.model = U2NET(3, 1)
        self.model.load_state_dict(torch.load(path_model, map_location=self.device)['model'])
        # self.model.load_state_dict(torch.load(path_model, map_location=self.device))
        self.model.eval()

    @staticmethod
    def norm_pred(tensor: torch.Tensor) -> torch.Tensor:
        max_tensor = torch.max(tensor)
        min_tensor = torch.min(tensor)
        norm_tensor = (tensor - min_tensor) / (max_tensor - min_tensor)

        return norm_tensor

    def infer(self, img: np.ndarray) -> np.ndarray:
        """
        Crop ROI from input image

        :param img: an BGR image
        :return: cropped image
        """
        height, width = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        batch_tensor = self.pre_process(img_rgb)

        pred = self.model(batch_tensor)
        pred = pred[:, 0, :, :]
        pred = self.norm_pred(pred)

        output_predictions = pred.squeeze().detach().cpu()
        output_predictions = cv2.cvtColor(np.array(output_predictions * 255, dtype=np.uint8), cv2.COLOR_GRAY2BGR)

        _, output = cv2.threshold(output_predictions, 190, 255, cv2.THRESH_BINARY)

        try:
            pts = self.post_process(output)
            pts[:, 0] = pts[:, 0] * width // self.size
            pts[:, 1] = pts[:, 1] * height // self.size
            final = perspective.four_point_transform(img, pts)

        except Exception as ex:
            print("Cannot crop card, ", ex)
            final = img

        return final


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="U2NET Segmentation Training", add_help=add_help)

    parser.add_argument("--data-path", default='data', type=str, help="path to images folder")
    parser.add_argument("--model-path", default='u2net_crop_model.pth', type=str, help="path to model")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    # data_path = '/home/administrator/Projects/Laos eKYC/datasets/data_crop/pp_data/*.*'
    # model_path = '/home/administrator/Projects/Laos eKYC/api/ekyc/weights/card_crop/u2net_crop_model.pth'

    folder_img_path = os.path.join(args.data_path, "*.*")
    model_path = args.model_path

    model = CardCrop(model_path, size=320)

    for i, img_path in enumerate(sorted(glob.glob(folder_img_path))[:]):
        print(i, img_path)

        IMG = cv2.imread(img_path)

        start = time.time()
        OUT = model.infer(IMG)
        print(time.time() - start)

        cv2.imshow('org', IMG)
        cv2.imshow('out', OUT)
        cv2.waitKey(0)
