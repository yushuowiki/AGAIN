"""
Organize the attribute labels of each image and turn the labels into a transposed visual structure
"""
import os
import torch
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Exogenknowledge.KnowledgeExtraction import run_knowledge
from Exogenknowledge.mlntool import load_model_output, empty_model_output
from analysis import top_k, binary_accuracy
from analysis import AverageMeter
import torch.nn as nn

from dataset import load_data, CUBDataset, load_data_several_class
from config import BASE_DIR
from torchvision import utils as vutils

if __name__ == '__main__':

    data_dir_ = "./CUB_processed/class_attr_data_10"
    eval_data = "test"
    use_attr = True
    no_img = False
    batch_size = 16
    image_dir = "images"
    n_class_attr = 2  # whether attr prediction is a binary or triary classification

    n_attributes = 112

    model_dir = "./ConceptModel__Seed1/outputs/best_model_1.pth"
    data_dir = os.path.join(BASE_DIR, data_dir_, eval_data + '.pkl')

    loader = load_data([data_dir], use_attr, no_img, batch_size, image_dir= image_dir,
                      n_class_attr=n_class_attr)
    empty_model_output()
    for data_idx, data in enumerate(loader):
        inputs_images, labels, attr_labels = data
        attr_labels = [i.long() for i in attr_labels]
        # attr_labels = torch.stack(attr_labels).t()
        load_model_output(attr_labels, batch_size, labels)
        print("finish process:" + str(data_idx))