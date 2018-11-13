import sys, os
sys.path.insert(0, '../../../../vision/')
sys.path.append('../../../../../pytorch-segmentation-detection/')

from pytorch_segmentation_detection.datasets.pascal_voc import PascalVOCSegmentation
import pytorch_segmentation_detection.models.fcn as fcns
import pytorch_segmentation_detection.models.resnet_dilated as resnet_dilated
from pytorch_segmentation_detection.transforms import (ComposeJoint,
                                                       RandomHorizontalFlipJoint,
                                                       RandomScaleJoint,
                                                       CropOrPad,
                                                       ResizeAspectRatioPreserve)

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms

import numbers
import random

from matplotlib import pyplot as plt

import numpy as np
from PIL import Image

from sklearn.metrics import confusion_matrix


def flatten_logits(logits, number_of_classes):
    """Flattens the logits batch except for the logits dimension"""

    logits_permuted = logits.permute(0, 2, 3, 1)
    logits_permuted_cont = logits_permuted.contiguous()
    logits_flatten = logits_permuted_cont.view(-1, number_of_classes)

    return logits_flatten


def flatten_annotations(annotations):
    return annotations.view(-1)


def get_valid_annotations_index(flatten_annotations, mask_out_value=255):
    return torch.squeeze(torch.nonzero((flatten_annotations != mask_out_value)), 1)


# from pytorch_image_segmentation.transforms import RandomCropJoint

number_of_classes = 21

labels = range(number_of_classes)

# train_transform = ComposeJoint(
#                 [
#                     RandomHorizontalFlipJoint(),
#                     RandomCropJoint(crop_size=(224, 224)),
#                     #[ResizeAspectRatioPreserve(greater_side_size=384),
#                     # ResizeAspectRatioPreserve(greater_side_size=384, interpolation=Image.NEAREST)],

#                     #RandomCropJoint(size=(274, 274))
#                     # RandomScaleJoint(low=0.9, high=1.1),

#                     #[CropOrPad(output_size=(288, 288)), CropOrPad(output_size=(288, 288), fill=255)],
#                     [transforms.ToTensor(), None],
#                     [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None],
#                     [None, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long()) ]
#                 ])

# trainset = PascalVOCSegmentation('datasets',
#                                  download=False,
#                                  joint_transform=train_transform)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
#                                           shuffle=True, num_workers=4)


valid_transform = ComposeJoint(
    [
        [transforms.ToTensor(), None],
        [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), None],
        [None, transforms.Lambda(lambda x: torch.from_numpy(np.asarray(x)).long())]
    ])

valset = PascalVOCSegmentation('datasets',
                               train=False,
                               download=False,
                               joint_transform=valid_transform)

valset_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                            shuffle=False, num_workers=0)


# train_subset_sampler = torch.utils.data.sampler.SubsetRandomSampler(xrange(904))
# train_subset_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=1,
#                                                    sampler=train_subset_sampler,
#                                                    num_workers=2)


# Define the validation function to track MIoU during the training
def validate():
    fcn.eval()

    overall_confusion_matrix = None

    for image, annotation in valset_loader:

        image = Variable(image.cuda())
        logits = fcn(image)

        # First we do argmax on gpu and then transfer it to cpu
        logits = logits.data
        _, prediction = logits.max(1)
        prediction = prediction.squeeze(1)

        prediction_np = prediction.cpu().numpy().flatten()
        annotation_np = annotation.numpy().flatten()

        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled

        current_confusion_matrix = confusion_matrix(y_true=annotation_np,
                                                    y_pred=prediction_np,
                                                    labels=labels)

        if overall_confusion_matrix is None:

            overall_confusion_matrix = current_confusion_matrix
        else:

            overall_confusion_matrix += current_confusion_matrix

    intersection = np.diag(overall_confusion_matrix)
    ground_truth_set = overall_confusion_matrix.sum(axis=1)
    predicted_set = overall_confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    intersection_over_union = intersection / union.astype(np.float32)
    mean_intersection_over_union = np.mean(intersection_over_union)

    fcn.train()

    return mean_intersection_over_union

if __name__ == '__main__':
    fcn = resnet_dilated.Resnet18_8s(num_classes=21)
    fcn.load_state_dict(torch.load('resnet_18_8s_59.pth'))
    fcn.cuda()

    print("MIOU:", validate())
