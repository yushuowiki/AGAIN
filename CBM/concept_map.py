import os
import csv
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
attr_label_idx = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23
    , 25, 29, 30, 35, 36, 38, 40, 44, 45, 50
    , 51, 53, 54, 56, 57, 59, 63, 64, 69, 70
    , 72, 75, 80, 84, 90, 91, 93, 99, 101, 106
    , 110, 111, 116, 117, 119, 125, 126, 131, 132, 134
    , 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

attr_label_idx_1 = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23
    , 25, 29, 30, 35, 36, 38, 40, 44, 45, 50
    , 51, 53, 54, 56, 57, 59, 63, 64, 69, 70
    , 72, 75, 80, 84, 90, 91, 93, 99, 101, 106
    , 110, 111, 116, 117, 119, 125, 126, 131, 132, 134
    , 145, 149, 151, 152, 153, 157]
attr_label_idx_2 = [158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

attr_label_idx_30 = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23
    , 25, 29, 30, 35, 36, 38, 40, 44, 45, 50
    , 51, 53, 54, 56, 57, 59, 63, 64, 69, 70]
def extract_first_number(line):
    words = line.split()
    return words[0]

def process_txt_file(file_path):
    numbers = []
    with open(file_path, 'r', encoding='utf-8') as file:
        i = 1
        for line in file:
            if i in attr_label_idx:
               number = extract_first_number(line)
               if number is not None:
                  numbers.append(number)
            i += 1
    return numbers

def main(input_folder, output_csv,corr_csv):
    # 主函数，遍历文件夹中的txt文件，提取数字并写入CSV文件
    all_numbers = []
    #file_names = []



    # 写入CSV文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        #csv_writer.writerow(['File Name'] + [f'Line {i+1}' for i in range(len(all_numbers)//len(file_names))])
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.txt'):
                file_path = input_folder + "/" + file_name
                print(file_path)
                numbers = process_txt_file(file_path)
                csv_writer.writerow(numbers)
                all_numbers.append(numbers)

            #row_data = [file_names[i]] + all_numbers[i::len(file_names)]
    # 假设你有一个二维列表 vectors，表示向量集合

    # 将二维列表转换为NumPy数组
    vectors_array = np.array(all_numbers)
    vectors_array = vectors_array.astype(float)
    tn = torch.from_numpy(vectors_array)
    # 计算余弦相似度矩阵
    tnT = tn.T
    cosine_similarity_matrix = torch.nn.functional.cosine_similarity(tnT.unsqueeze(1), tnT.unsqueeze(0), dim=-1)
    cosine_similarity_matrix = cosine_similarity_matrix.tolist()
    with open(corr_csv, 'w', newline='', encoding='utf-8') as csv_file2:
        csv_writer = csv.writer(csv_file2)
        for x in cosine_similarity_matrix:
            csv_writer.writerow(x)
    # 打印余弦相似度矩阵
    print(cosine_similarity_matrix)

if __name__ == "__main__":
    input_folder = '/home/kang/lyc/ConceptBottleneck-master/CBM/attr_class'
    output_csv = 'concept_map.csv'
    corr_csv = 'CUB_corr.csv'
    main(input_folder, output_csv,corr_csv)

