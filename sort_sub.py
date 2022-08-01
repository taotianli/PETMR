import pandas as pd
import csv

xlsx_path = 'D:/Down/PET_data/Diagnosis_Information.xlsx'
sub_path = 'D:/Down/PET_data/sub_list.xlsx'
sub_df = pd.read_excel(xlsx_path, engine='openpyxl')  # 读取病历信息
sub_list = pd.read_excel(sub_path, engine='openpyxl')
sub_info = dict(zip(sub_df['ID'], sub_df['Label']))
sub_zs_info = dict(zip(sub_list['sub_ID'], sub_list['zs_ID']))

fileName = "Sub_Diagnosis.csv"
##保存文件
csv_file = open(fileName, 'w', newline='', encoding='gbk')
writer = csv.writer(csv_file)
writer.writerow(['sub_ID', 'label'])
for k in sub_zs_info.keys():
    # print(k, sub_zs_info[k], sub_info[sub_zs_info[k]])
    writer.writerow([k, sub_info[sub_zs_info[k]]])
csv_file.close()

