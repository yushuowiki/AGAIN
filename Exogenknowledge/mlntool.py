import torch
import numpy as np
import re
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_output(output, batch_size, labels):
    #new_tensor = torch.empty(batch_size, len(output))
    #new_tensor.to(device)
    #for i in output:
    #    new_tensor = torch.cat((new_tensor, i), dim=0).to(device)
    new_tensor = torch.stack(output)
    # print(output.shape[1])
    # print(output.shape[0])
    with open("Exogenknowledge/model_output.txt",'a') as file:
            for i in range(new_tensor.shape[1]):
                str_attr = str(labels[i]) + ' '
                for j in range(new_tensor.shape[0]):
                  str_attr = str_attr + str(1 if np.array(new_tensor[j, i].detach().cpu()) > 0.5 else 0) + ' '
                str_attr = str_attr + '\n'
                file.write(str_attr)
    #print("finish write model output to model_output.txt")


def load_model_output_to_foln(knowledge):
    foln_input = []
    with open("Exogenknowledge/model_output.txt", 'r') as file:
        lines = file.readlines()
        # tensor_2d = torch.zeros((batch_size, n_attributes))
        for line_count in range(len(lines)):
            # update_line = ''
            attr = lines[line_count].split()
            # vector = torch.empty(0)
            # dele_sign = 0
            foln_input_str = 'C' + re.search(r'\((\d+)\)', attr[0]).group(1) + "(x)"
            for i in range(1, len(attr)):
                for k in knowledge:
                    if i == k["output_idx"] + 1 and k["class_idx"] == int(re.search(r'\((\d+)\)', attr[0]).group(1)):  #
                        formula_ri = 'A' + str(k["attr_label_idx"]) + '_' + str(k["output_idx"])
                        formula_ri_logic_sign = ''
                        if k["logic"] == 0:
                            formula_ri_logic_sign = '!'
                        foln_input_str = foln_input_str + " ^ " + formula_ri_logic_sign + formula_ri + "(x)"
            foln_input.append(foln_input_str)
    return foln_input


def empty_model_output():
    with open("Exogenknowledge/model_output.txt",'w') as file:
            file.truncate(0)