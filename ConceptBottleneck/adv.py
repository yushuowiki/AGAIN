import os
import torch
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Exogenknowledge.KnowledgeExtraction import run_knowledge, run_delecter
from Exogenknowledge.mlntool import load_model_output, empty_model_output
from analysis import top_k, binary_accuracy, accuracy_y
from analysis import AverageMeter
import torch.nn as nn

from dataset import load_data, CUBDataset, load_data_several_class
from config import BASE_DIR
from torchvision import utils as vutils
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_eval_acc = 0.0
all_eval_adv_acc = 0.0
all_eval_update_acc = 0.0
all_eval_acc_y = 0.0
all_eval_adv_acc_y = 0.0
all_eval_update_acc_y = 0.0
all_cost = 0.0
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

def simple_eval_Y(class_outputs, labels):
    class_acc_meter = []
    class_acc_meter.append(AverageMeter())
    class_acc = accuracy_y(class_outputs, labels)
    return class_acc


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


def pgd_attack_random(model,model2, inputs_images,labels, attr_labels, eps=1, alpha=1, iters=40, randomize=True):
    """ Construct L_inf adversarial examples on the examples X """
    #print("Construct L_inf adversarial examples:")
    model.eval()
    model2.eval()
    # print(data)
    # inputs_images, labels, attr_labels = data
    # attr_labels = [i.long() for i in attr_labels]
    # attr_labels = torch.stack(attr_labels).t()  # N x 312

    #attr_labels = torch.autograd.Variable(attr_labels).float() # change float

    attr_labels = attr_labels.to(device)
    inputs_images = inputs_images.to(device)
    labels = labels.to(device)
    adv_img_list = []
    for i in tqdm(range(inputs_images.shape[0]), desc="Perturbation sample generation", unit="batch"):
        inputs_image = inputs_images[1,:,:,:]
        attr_label = attr_labels[i,:]
        label = labels[i]
        if randomize:
            delta = torch.rand_like(inputs_image, requires_grad=True).to(device)  
            delta.data = delta.data * 2 * eps - eps
            delta.data = (delta.data + inputs_image).clamp(-0.5, 0.5) - (inputs_image)  # range[-0.5 - 0.5]
        else:
            delta = torch.zeros_like(inputs_image, requires_grad=True).to(device)

        inputs_image = inputs_image.clone().to(device)
        #print("[Perturbation loss start]")
        total_loss = 0.00
        for t in range(iters):
            # print("Perturbation iterations: "+str(t)+"---"+str(total_loss))
            # model = nn.DataParallel(model)
            output = model(torch.unsqueeze(inputs_image + delta, dim=0))
            # print("output: "+str(outputs))
            # attr_outputs = [torch.nn.ReLU()(o) for o in outputs]
            attr_outputs_sigmoid = [torch.nn.Sigmoid()(o) for o in output]
            attr_outputs_sigmoid_tensor = torch.stack(attr_outputs_sigmoid)
            class_output = model2(attr_outputs_sigmoid_tensor.T)
            criterion = torch.nn.CrossEntropyLoss()
            class_output = torch.squeeze(class_output, dim=0)
            Y_loss = criterion(class_output, torch.unsqueeze(label, dim=0))
            # print("output sigmod: " + str(attr_outputs_sigmoid))
            #attr_outputs_sigmoid = torch.Tensor(attr_outputs_sigmoid)
            # print(outputs)

            #attr_criterion = []
            # for i in range(n_attributes):
            #    attr_criterion.append(torch.nn.CrossEntropyLoss())
            # total_top_k = 0
            # for i in range(n_attributes):
            #     for j in range(len(attr_label[i])):
            #         if attr_label[i] == 1:
            #
            #             total_top_k = total_top_k + attr_outputs_sigmoid[i][j]
            #         else:
            #             total_top_k = total_top_k - attr_outputs_sigmoid[i][j]
            total_top_k = 0
            for i in range(n_attributes):
                if attr_label[i] == 1:
                    total_top_k = total_top_k + attr_outputs_sigmoid[i]
                else:
                    total_top_k = total_top_k - attr_outputs_sigmoid[i]
            attr_criterion = []
            losses = []
            for i in range(n_attributes):
                attr_criterion.append(torch.nn.CrossEntropyLoss())
            #for i in range(len(attr_criterion)):
            #    losses.append(attr_criterion[i](torch.squeeze(attr_outputs_sigmoid[i], dim=0).T,torch.squeeze(attr_label[i], dim=0)))
            # total_loss = -sum(losses) / n_attributes
            seta = 0.90
            total_loss = seta * (sum(total_top_k) / n_attributes)+(1 - seta) * Y_loss
            #print("loss",t,":",(sum(total_top_k) / n_attributes),Y_loss)
            #total_loss = 0.3*(-total_top_k/n_attributes)-0.7*Y_loss
            #total_loss =  - 0.9 * Y_loss
            total_loss.backward()
            # loss = torch.nn.CrossEntropyLoss()(attr_outputs_sigmoid, attr_labels)
            # loss.backward()

            delta.data = (delta - alpha * delta.grad.detach().sign()).clamp(-eps, eps)
            delta.data = (delta.data + inputs_image).clamp(-0.5, 0.5) - (inputs_image)
            delta.grad.zero_()
        adv_img_list.append(torch.unsqueeze((delta + inputs_image).detach(),dim=0))
        result_images = []
    #print("[Perturbation finish]")
      #save_img(delta, inputs_image)
    return torch.cat(adv_img_list, dim=0)


if __name__ == '__main__':
    data_dir_ = "./CUB_processed/class_attr_data_10"
    eval_data = "test"
    use_attr = True
    no_img = False
    batch_size = 16
    image_dir = "images"
    n_class_attr = 2 # whether attr prediction is a binary or triary classification
    n_attributes = 112
    eps_list = [8]
    model_dir = "./ConceptModel__updateloss/outputs/best_model_1.pth"
    model_dir_c_to_Y = "./IndependentModel_WithVal___Seed1/outputs/best_model_1.pth"
    data_dir = os.path.join(BASE_DIR, data_dir_, eval_data + '.pkl')
    #select_class = [1,4,7,8,20]# Specify the category of the filtered data set
    #select_class = [3]  # Specify the category of the filtered data set
    select_class = [1,2,3,4,5,6,7,8,9,10,
                    11,12,13,14,15,16,17,18,19,20,
                    21,22,23,24,25,26,27,28,29,30,
                    31,32,33,34,35,36,37,38,39,40,
                    41,42,43,44,45,46,47,48,49,50,
                    51,52,53,54,55,56,57,58,59,60,
                    61,62,63,64,65,66,67,68,69,70,]
    loader = load_data_several_class(select_class, [data_dir], use_attr, no_img, batch_size, image_dir= image_dir,
                       n_class_attr=n_class_attr)
    #loader = load_data([data_dir], use_attr, no_img, batch_size, image_dir= image_dir,
    #                  n_class_attr=n_class_attr)

    # img_path = "ConceptBottleneck/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg"
    # img = Image.open(img_path).convert('RGB')
    #model = torch.load(model_dir)
    model = torch.load(model_dir)['model']
    modelc_to_y = torch.load(model_dir_c_to_Y)



    for eps_num in eps_list:
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

            adv_images = pgd_attack_random(model,modelc_to_y, inputs_images,labels, attr_labels, eps=eps_num/255.0, alpha=1.0, iters=10, randomize=True)

            # inference
            inputs_var = torch.autograd.Variable(inputs_images).cuda()
            adv_inputs_var = torch.autograd.Variable(adv_images).cuda()

            outputs = model(inputs_var)
            outputs_tensor = torch.stack(outputs)
            #outputs_tensor_var = torch.autograd.Variable(outputs_tensor).cuda
            #print(outputs_tensor_var)
            #print(outputs_tensor.shape[0])
            #print(outputs_tensor.shape[1])
            outputs_Y = modelc_to_y(outputs_tensor.T)
            #print(outputs_Y)
            adv_outputs = model(adv_inputs_var)
            adv_outputs_tensor = torch.stack(adv_outputs)
            adv_outputs_Y = modelc_to_y(adv_outputs_tensor.T)
            # print(type(adv_outputs))
            # load_model_output([torch.nn.Sigmoid()(o) for o in adv_outputs], batch_size, labels)
            load_model_output(adv_outputs, batch_size, labels)
            update_output, detect_num = run_knowledge(batch_size, n_attributes, labels)
            #update_output_tensor = [torch.unsqueeze(tensor, dim=0) for tensor in update_output]
            #update_output_tensor = torch.cat(update_output_tensor, dim=0)
            #update_outputs_tensor = torch.stack(update_output)
            #update_output_tensor_var = update_output_tensor.cuda()
            #update_outputs_Y = modelc_to_y(update_output_tensor_var.T)
            cost = run_delecter(batch_size, n_attributes, labels)

            #all_outputs = all_outputs + outputs
            #all_adv_outputs = all_adv_outputs + adv_outputs
            #all_update_output = all_update_output + update_output
            #all_attr_labels = all_attr_labels + .extend(attr_labels)
            #all_inputs_images = all_inputs_images + inputs_images.size(0)
            #all_adv_images = all_adv_images + adv_images.size(0)
            all_eval_acc = all_eval_acc + simple_eval(inputs_images, outputs, attr_labels)
            all_eval_adv_acc = all_eval_adv_acc + simple_eval(adv_images, adv_outputs, attr_labels)
            #all_eval_update_acc = all_eval_update_acc + update_simple_eval(adv_images, update_output, attr_labels)
            #all_cost = all_cost+cost

            all_eval_acc_y = all_eval_acc_y + simple_eval_Y(outputs_Y, labels)
            all_eval_adv_acc_y = all_eval_adv_acc_y + simple_eval_Y(adv_outputs_Y, labels)
            #all_eval_update_acc_y = all_eval_update_acc_y + simple_eval_Y(update_outputs_Y, labels)
            #print('detect number: %.5f' % detect_num)
            all_detect_num = all_detect_num + detect_num
            all_img_sum = all_img_sum + batch_size


            if i > len(select_class)-1:
                break
        print('----------------------------------------')
        print('attribute accuracy: %.5f' % (all_eval_acc/n_batch))
        print('attack attribute accuracy: %.5f' % (all_eval_adv_acc / n_batch))
        print('correct attribute accuracy: %.5f' % (all_eval_update_acc / n_batch))

        print('class accuracy: %.5f' % (all_eval_acc_y/n_batch))
        print('attack class accuracy: %.5f' % (all_eval_adv_acc_y / n_batch))
        print('correct class accuracy: %.5f' % (all_eval_update_acc_y / n_batch))

        print('detection rate: %.5f' % (all_detect_num / all_img_sum))
        print('cost time: %.5f' % (all_cost / n_batch))
        print('----------------------------------------')
        with open('adv_result.txt', 'a') as file:
            file.write("eps_num:" + str(eps_num) + '\n')
            file.write("time:"+"{:.5f}".format((all_cost / n_batch))+'\n')



    #vutils.save_image(inputs_images, './test_ori.jpg', normalize=True)

    #result_images = (delta + inputs_images).detach()
    #print(type(result_images))
    #img_count = 0
    #vutils.save_image(result_images, './test.jpg', normalize=True)
    # for img in enumerate(result_images):
    #     img.save('ConceptBottleneck/CUB_200_2011_adv_images/' + img_count + ".jpg")
    #     img_count = img_count + 1
    # img = Image.fromarray((delta + inputs_images).detach())  # If image_data is a NumPy array
    # img.save('ConceptBottleneck/CUB_200_2011_adv_images/' +i+".jpg")
