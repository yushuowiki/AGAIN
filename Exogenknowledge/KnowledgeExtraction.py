import os
import torch
import sys
import re
import time
import numpy as np

from Exogenknowledge.mlntool import load_model_output_to_foln
from Exogenknowledge.run_factor_graph import run_foln

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

attr_label_idx = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

attr_label_idx_add_one = []
for i in range(len(attr_label_idx)):
    attr_label_idx_add_one.append(attr_label_idx[i]+1)
"""
attr_label_idx start = 1
output_idx start = 0
"""

def building_class_knowledge(classidx_list):
    knowledge = []
    for classidx in classidx_list:
        with open("ConceptBottleneck/attr_class/" + str(classidx) + ".txt", 'r') as file:
            lines = file.readlines()
            for line in lines:
                attr = line.split()
                if int(attr[1]) in attr_label_idx_add_one:
                    if float(attr[0]) < 15.0:
                        knowledge.append({"class_idx": int(classidx), "attr_label_idx": int(attr[1]), "output_idx": attr_label_idx_add_one.index(int(attr[1])),
                                          "logic": 0})
                    if float(attr[0]) > 55.0:
                        knowledge.append({"class_idx": int(classidx), "attr_label_idx": int(attr[1]), "output_idx": attr_label_idx_add_one.index(int(attr[1])),
                                          "logic": 1})
    return knowledge

def compare_model_output_knowledge(knowledge, batch_size, n_attributes):
    # class_label = 0
    detect_num = 0
    with open("Exogenknowledge/model_correction_output.txt", 'w') as file_w:
        with open("Exogenknowledge/model_output.txt", 'r') as file:
            lines = file.readlines()
            tensor_2d = torch.zeros((batch_size, n_attributes))
            for line_count in range(len(lines)):
                update_line = ''
                attr = lines[line_count].split()
                vector = torch.empty(0)
                dele_sign = 0
                for i in range(1, len(attr)):
                    has_k = 0
                    for k in knowledge:
                        if i == k["output_idx"] + 1 and k["class_idx"] == int(re.search(r'\((\d+)\)', attr[0]).group(1)) and k["logic"] != int(attr[i]): #
                            has_k = 1
                            dele_sign = 1
                            if k["logic"] == 1:
                                update_line += '1' + ' '
                                # vector = torch.cat((vector, torch.tensor([1])), dim=0)
                                tensor_2d[line_count, i - 1] = 1
                            else:
                                update_line += '0' + ' '
                                # vector = torch.cat((vector, torch.tensor([0])), dim=0)
                                tensor_2d[line_count, i - 1] = 0
                            break
                    if has_k == 0:
                        update_line += attr[i] + ' '
                        tensor_2d[line_count, i - 1] = int(attr[i])
                        # vector = torch.cat((vector, torch.tensor([int(attr[i])])), dim=0)
                    has_k = 0
                if dele_sign == 1:
                    detect_num = detect_num + 1
                update_line += '\n'
                # tensor_2d = torch.cat((tensor_2d, vector.unsqueeze(1)), dim=1)
                file_w.write(update_line)
            # tensor_list = torch.split(tensor_2d.T, 1, dim=1)
            #for row in tensor_2d:
            #    if not np.array_equal(row.numpy(), attr):
            #       detect_num = detect_num + 1
            new_list = tensor_2d.T.tolist()
            for i in range(len(new_list)):
                new_list[i] = torch.tensor(new_list[i])
            #print("finish update model output...")
    return new_list, detect_num


def knowledge_load_formula_predicate(knowledge, class_label, predicate):
    formula_le = 'C' + str(class_label)
    with open("Exogenknowledge/data.txt", 'a') as file_data:
        with open("Exogenknowledge/predicate.txt", 'a') as predicate_file:
            predicate_file.write('C' + str(class_label) + "(person)")
            file_data.write('C' + str(class_label) + "(x)" + '\n')
        with open("Exogenknowledge/formula.txt", 'a') as file:
            for k in knowledge:
                if k["class_idx"] == int(class_label):
                    formula_ri_logic_sign = ''
                    formula_ri = 'A' + str(k["attr_label_idx"]) + '_' + str(k["output_idx"])
                    if k["logic"] == 0:
                        formula_ri_logic_sign = '!'
                    formula_total = formula_le + "(x)" + " ^ " + formula_ri_logic_sign + formula_ri + "(x)"
                    file_data.write(formula_ri_logic_sign + formula_ri + "(x)" + '\n')
                    file.write(formula_total + '\n')
                    if formula_ri not in predicate:
                        predicate.append(formula_ri)
                        with open("Exogenknowledge/predicate.txt", 'a') as predicate_file:
                            predicate_file.write('\n' + formula_ri + "(person)")
            #print("finish load " + str(class_label) + "class")

    # with open("Exogenknowledge/formula.txt", 'a') as file:
    #     print(predicate)
    #     formula_le = 'C' + str(class_label)
    #     predicate.append(formula_le)
    #     file.write('C' + str(class_label) + "(person)" + '\n')
    #
    #     for k in knowledge:
    #         formula_ri_logic_sign = ''
    #         formula_ri = 'A' + str(k["attr_label_idx"]) + '_' + str(k["output_idx"])
    #         if k["logic"] == 0:
    #             formula_ri_logic_sign = '!'
    #         formula_total = formula_le + "(x)" + " ^ " + formula_ri_logic_sign + formula_ri + "(x)"
    #         file.write(formula_total + '\n')
    #         if formula_ri not in predicate:
    #             predicate.append(formula_ri)
    #     print("finish load " + str(class_label) + "class")

def write_predicate(predicate):
    with open("Exogenknowledge/predicate.txt", 'a') as file:
        for p in predicate:
            file.write(p + "(person)" + '\n')

def empty_formula_predicate():
    with open("Exogenknowledge/formula.txt",'w') as file:
            file.truncate(0)
    with open("Exogenknowledge/predicate.txt",'w') as files:
            files.truncate(0)
    with open("Exogenknowledge/data.txt", 'w') as file:
        file.truncate(0)

def run_knowledge(batch_size, n_attributes, labels):
    knowledge = building_class_knowledge(list(set(labels.numpy().astype(int))))# label.numpy().astype(int)-1 for label in labels
    # print(knowledge)
    return compare_model_output_knowledge(knowledge, batch_size, n_attributes)

def run_delecter(batch_size, n_attributes, labels):
    classidx_list = labels.numpy().astype(int)
    classidx_list = list(set(classidx_list))
    knowledge = building_class_knowledge(classidx_list)
    run_knowledge_load_formula(classidx_list, knowledge)
    batch_avg_cost = 0
    for foln_input in load_model_output_to_foln(knowledge):
        # print(foln_input)
        start_time = time.time()
        run_foln(foln_input)
        end_time = time.time()
        cost = end_time - start_time
        batch_avg_cost = batch_avg_cost + (cost/batch_size)
    return batch_avg_cost

def run_knowledge_load_formula(classidx_list, knowledge):
    predicate = []
    empty_formula_predicate()
    # classidx_list = labels.numpy().astype(int)
    # classidx_list = list(set(classidx_list))
    # knowledge = building_class_knowledge(classidx_list)
    for classidx in classidx_list:
        print("load in formula..." + str(classidx) + "class")
        knowledge_load_formula_predicate(knowledge, classidx, predicate)

def run_knowledge_single_load_formula(class_idx):
    predicate = []
    print("load in formula..." + str(class_idx) + "class")
    knowledge = building_class_knowledge(class_idx)
    knowledge_load_formula_predicate(knowledge, i, predicate)



if __name__ == '__main__':
    empty_formula_predicate()
    #run_knowledge_load_formula(200)
