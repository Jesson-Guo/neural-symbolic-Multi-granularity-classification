import os
import cv2
import numpy as np

import torch
from PIL import Image


def visualize_box(img, bbox):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255,0,0))
    return img


def test(model, optimizer, scheduler, ckpt_path, test_path):
    # load checkpoint
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_path))

    # load test images
    test_path = os.path.join(test_path, 'images')
    test_images = []
    test_files = []
    for f in os.listdir(test_path):
        f = os.path.join(test_path, f)
        x = Image.open(f).convert('RGB')
        test_images.append(torch.Tensor(x))
        test_files.append(f)

    # predict
    model.eval()
    predictions = model(x)

    # visualize
    for i in range(len(test_images)):
        image = test_images[i]
        bbox = np.array(predictions[i]['boxes'])
        image = visualize_box(image, bbox)


if __name__ == "__main__":
    imagenet = './tiny-imagenet-200/test'
    test_path = os.path.join(imagenet, 'images')
    test_images = []
    for f in os.listdir(test_path):
        f = os.path.join(test_path, f)
        x = Image.open(f).convert('RGB')
        test_images.append(torch.Tensor(x))
