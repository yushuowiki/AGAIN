attr_lines = []
with open('../CUB_200_2011/attributes.txt', 'r') as file:
    attr_lines = file.readlines()



with open('../CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt', 'r') as file:
    lines = file.readlines()
    file_name_flag = 0
    for line in lines:
        attr = line.split()
        i = 0
        with open("E:/python_workspace/ConceptBottleneck-master/CUB_200_2011/attr_class/"+str(file_name_flag) + ".txt", 'w') as file:
            for text in attr:
                str_attr = text + " " + attr_lines[i]
                file.write(str_attr)
                i = i + 1
        file_name_flag = file_name_flag + 1
        print("finish:" + str(file_name_flag))

all_attr_labels = [j for j in range(312)]
# n = len(all_attr_labels)
#all_attr_acc, all_attr_f1 = [], []
for i in range(112):
    #attr_preds = [all_attr_outputs_int[j] for j in range(n) if j % args.n_attributes == i]
    attr_labels = [all_attr_labels[j] for j in range(312) if j % 112 == i]
print(attr_labels)
