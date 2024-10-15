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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_eval_acc = 0.0
all_eval_acc
all_eval_adv_acc = 0.0
all_eval_update_acc = 0.0
n_batch = 0

def simple_eval(inputs, outputs, attr_labels):
    attr_outputs = [torch.nn.Sigmoid()(o) for o in outputs]
    attr_outputs_sigmoid = attr_outputs
    attr_acc_meter = [AverageMeter()]
    for _ in range(n_attributes):
        attr_acc_meter.append(AverageMeter())

    for i in range(n_attributes):
        # acc = top_k(attr_outputs_sigmoid[i].squeeze(), attr_labels[:, i])
        acc = binary_accuracy(attr_outputs_sigmoid[i].squeeze(), attr_labels[:, i])
        acc = acc.data.cpu().numpy()
        # acc = accuracy(attr_outputs_sigmoid[i], attr_labels[:, i], topk=(1,))
        attr_acc_meter[0].update(acc, inputs.size(0))
        attr_acc_meter[i + 1].update(acc, inputs.size(0))
    #print('Average attribute accuracy: %.5f' % attr_acc_meter[0].avg)
    return attr_acc_meter[0].avg


def update_simple_eval(inputs, outputs, attr_labels):
    attr_acc_meter = [AverageMeter()]
    for _ in range(n_attributes):
        attr_acc_meter.append(AverageMeter())

    for i in range(n_attributes):
        # acc = top_k(attr_outputs_sigmoid[i].squeeze(), attr_labels[:, i])
        acc = binary_accuracy(outputs[i].squeeze(), attr_labels[:, i])
        acc = acc.data.cpu().numpy()
        # acc = accuracy(attr_outputs_sigmoid[i], attr_labels[:, i], topk=(1,))
        attr_acc_meter[0].update(acc, inputs.size(0))
        attr_acc_meter[i + 1].update(acc, inputs.size(0))
    #print('Average attribute accuracy: %.5f' % attr_acc_meter[0].avg)
    return attr_acc_meter[0].avg

def save_img(delta,inputs_images):
    vutils.save_image(inputs_images, './test_ori.jpg', normalize=True)
    result_images = (delta + inputs_images).detach()
    print(type(result_images))
    img_count = 0
    vutils.save_image(result_images, './test.jpg', normalize=True)


def pgd_attack_random(model, inputs_images, attr_labels, eps=1, alpha=1, iters=40, randomize=True):
    """ Construct L_inf adversarial examples on the examples X """
    model.eval()
    attr_labels = torch.autograd.Variable(attr_labels).float() # change float
    attr_labels = attr_labels.to(device)
    inputs_images = inputs_images.to(device)

    if randomize:
        delta = torch.rand_like(inputs_images, requires_grad=True).to(device)  # 生成随机数并创建与输入张量具有相同形状的张量的函数
        delta.data = delta.data * 2 * eps - eps
        delta.data = (delta.data + inputs_images).clamp(-0.5, 0.5) - (inputs_images)  # range[-0.5 - 0.5]
    else:
        delta = torch.zeros_like(inputs_images, requires_grad=True).to(device)

    inputs_images = inputs_images.clone().to(device)
    #print("[Perturbation loss start]")
    total_loss = 0.00
    for t in range(iters):
        outputs = model(inputs_images + delta)
        attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in outputs]
        total_top_k = 0
        for i in range(n_attributes):
            for j in range(len(attr_labels[:, i])):
                if attr_labels[j, i] == 1:
                    total_top_k = total_top_k + attr_outputs_sigmoid[i][j]
        total_loss = -total_top_k
        total_loss.backward()
        # loss = torch.nn.CrossEntropyLoss()(attr_outputs_sigmoid, attr_labels)
        # loss.backward()

        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-eps, eps)
        delta.data = (delta.data + inputs_images).clamp(-0.5, 0.5) - (inputs_images)
        delta.grad.zero_()
        result_images = []
    #print("[Perturbation finish]")
    save_img(delta, inputs_images)
    return (delta + inputs_images).detach()


if __name__ == '__main__':
    data_dir_ = "./CUB_processed/class_attr_data_10"
    eval_data = "test"
    use_attr = True
    no_img = False
    batch_size = 16
    image_dir = "images"
    n_class_attr = 2 # whether attr prediction is a binary or triary classification

    n_attributes = 112

    model_dir = "./ConceptModel__updateloss/outputs/best_model_1.pth"
    data_dir = os.path.join(BASE_DIR, data_dir_, eval_data + '.pkl')
    select_class = [3,4,5,6,7,8,9]# Specify the category of the filtered data set
    loader = load_data_several_class(select_class, [data_dir], use_attr, no_img, batch_size, image_dir= image_dir,
                       n_class_attr=n_class_attr)
    #loader = load_data([data_dir], use_attr, no_img, batch_size, image_dir= image_dir,
    #                  n_class_attr=n_class_attr)

    # img_path = "CBM/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
    # img = Image.open(img_path).convert('RGB')
    #model = torch.load(model_dir)
    model = torch.load(model_dir)['model']


    for eps_num in range(1,32):
        i = 0
        all_outputs, all_adv_outputs, all_update_output, all_attr_labels = [], [], [], []
        all_inputs_images, all_adv_images = 0, 0
        detect_num = 0
        all_detect_num = 0
        all_img_sum = 0
        print("eps_num:",eps_num)
        for data_idx, data in enumerate(loader):
            # print('data_idx: %d' % data_idx)
            n_batch = n_batch + 1
            empty_model_output()
            i = i + 1
            inputs_images, labels, attr_labels = data
            attr_labels = [i.long() for i in attr_labels]
            attr_labels = torch.stack(attr_labels).t() # Transfer
            # print("attr_labels long: "+str(attr_labels.shape[1]))

            adv_images = pgd_attack_random(model, inputs_images, attr_labels, eps=eps_num/255.0, alpha=1.0, iters=10, randomize=True)

            # inference
            inputs_var = torch.autograd.Variable(inputs_images).cuda()
            adv_inputs_var = torch.autograd.Variable(adv_images).cuda()

            outputs = model(inputs_var)
            adv_outputs = model(adv_inputs_var)
            load_model_output(adv_outputs, batch_size, labels)
            update_output, detect_num = run_knowledge(batch_size, n_attributes, labels)

            all_detect_num = all_detect_num + detect_num
            all_img_sum = all_img_sum + batch_size
            if i > 6:
                break
        print(all_detect_num / all_img_sum)
