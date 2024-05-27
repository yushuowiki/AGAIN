import os
import csv
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
    vectors_array = np.array(all_numbers)
    vectors_array = vectors_array.astype(float)
    print(vectors_array[:111])
    # high 12-34,36-9,12-6,26-11,26-5,108-59
    #middle 38-4
    #low 81-84 -0.841935692956464,
    #low 110-112 -0.659449708605464
    #low 78-81 -0.625885948918029
    #low 84-79 -0.680499269691616
    #low 54-81 -0.589006182163358
    #low 110-86 -0.5716308
    tenth_column = []
    for i in range(112):
        tenth_column_temp = [row[i] for row in vectors_array]
        tenth_column.append(tenth_column_temp)
        #tenth_column_34 = [row[10] for row in vectors_array]

    cosine_similarity_matrix = np.zeros((112, 112))  # 创建一个全零的矩阵用于存储相似度
    for i in range(112):
        for j in range(112):
            cosine_similarity_matrix[i, j] = np.corrcoef(tenth_column[i], tenth_column[j])[0, 1]

    import matplotlib.pyplot as plt

    # 绘制散点图
    plt.figure(figsize=(4, 4))
    plt.scatter(tenth_column[107], tenth_column[58], s=15, c='red')
    # 添加标签和标题
    #plt.xlabel('association scores')
    #plt.ylabel('association scores')
    plt.xlim(-5, 105)  # x 轴刻度范围从 0 到 6
    plt.ylim(-5, 105)
    plt.savefig('E:\\学习文件\\work1知识集成\\2024NIPS\\corr-img-point\\12.svg')
    plt.show()
    #print(cosine_similarity_matrix)
    #print(np.sum(cosine_similarity_matrix < 0))
    np.savetxt('cosine_similarity_matrix.csv', cosine_similarity_matrix, delimiter=',')
if __name__ == "__main__":
    input_folder = 'E:\\python_workspace\\ConceptBottleneck-master\\CUB_200_2011\\attr_class'
    output_csv = 'concept_map.csv'
    corr_csv = 'CUB_corr.csv'
    main(input_folder, output_csv,corr_csv)