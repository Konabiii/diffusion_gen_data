import os
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import math
from numpy import * 
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import StandardScaler


def Max(array):
    length = len(array)
    max_tam = -10000
    for i in range(length):
        if (max_tam < max(array[i])):
            max_tam = max(array[i])
    return max_tam


def Split_String(name):
    if "diffusers" in name:
        return name.split("diffusers", 1)[1].lstrip("/")
    else:
        return os.path.basename(name)  

def List_Name_Image(path):
    list_name = []
    files_path = [x for x in os.listdir(path) if x.endswith(".csv")]
    for i in files_path:
        list_name.append(i)   
    return list_name

def Load_Data(Path, list_name):
    X=[]
    for i in list_name:
        filename = Path +'/' + i
        with open(filename) as f:
            reader = csv.reader(f)
            tam=[]
            line = [row for row in reader]
            for k in range(4,len(line)):
                tam1=[]
                for l in range(len(line[k])):
                    tam1.append(float(line[k][l]))
                tam.append(tam1)

            draft = np.array(tam)
            rotated_image = np.rot90(draft,2)
            rotated_image = np.diff(rotated_image)

            if((('Step_R' in i)==True) or (('Step_T' in i)==True)):
                Draft_1 = np.zeros((32,31))
                for i in range(32):
                    Draft_1[i]=rotated_image[i][::-1]
                rotated_image = Draft_1

            tam_1 = [[0] for _ in range(32)]
            rotated_image = insert(rotated_image,[31],tam_1,1)

            X.append(rotated_image)
            tam = np.array(X)
            print(len(X))
    return X 

def Load_Data_Test(Path,list_name):
    X=[]
    for i in list_name:
        filename = Path +'/' + i
        with open(filename) as f:
            reader = csv.reader(f)
            tam=[]
            line = [row for row in reader]
            for k in range(len(line)):
                tam1=[]
                for l in range(len(line[k])):
                    tam1.append(float(line[k][l]))
                tam.append(tam1)

            draft = np.array(tam)
            rotated_image = np.rot90(draft,1)
            draft_1 = np.zeros((32,31))
            for o in range(31):
                draft_1[o]=rotated_image[o]

            tam_1 = [[0] for _ in range(32)]
            rotated_image = insert(draft_1,[31],tam_1,1)

            X.append(rotated_image)
    return X 

def Load_Data_Train_Test(path_file):
    path = path_file
    List_Name_Image_Ellipse = List_Name_Image(path)
    list_tam=np.array(List_Name_Image_Ellipse)
    y_name = list_tam
    X_data=Load_Data(path,list_tam)
    X_data=np.array(X_data)
    return X_data, np.array(y_name)

def Train_Test_Split(standardized,name_data):
    x_data = np.array(standardized)
    y_train, y_test, y_train_name, y_test_name = train_test_split(
        x_data, name_data, test_size=0.3, random_state=4)
    x_train=[]
    x_test=[]
    for i in range(len(y_train)):
        img = y_train[i]
        img = cv2.resize(img, (16, 16), interpolation = cv2.INTER_CUBIC)
        x_train.append(img)
    for i in range(len(y_test)):
        img = y_test[i]
        img = cv2.resize(img, (16, 16), interpolation = cv2.INTER_CUBIC)
        x_test.append(img)
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_test=np.array(x_test)
    y_test=np.array(y_test)
    return x_train,y_train,x_test,y_test,y_train_name, y_test_name

def Train_Test_Split_Test(name_path):
    path = name_path
    List_Name_Image_Test = List_Name_Image(path)
    list_tam=np.array(List_Name_Image_Test)
    y_name_test =  list_tam
    X_data_test = Load_Data_Test(path,list_tam)
    X_data_test = np.array(X_data_test)
    return X_data_test,y_name_test


def Real_Data_Test_Split(x_real_data_high):
    x_real_data_low=[]
    for i in range(len(x_real_data_high)):
        img = x_real_data_high[i]
        img = cv2.resize(img, (16, 16), interpolation = cv2.INTER_CUBIC)
        x_real_data_low.append(img)
    x_real_data_low = np.array(x_real_data_low)
    return x_real_data_low, x_real_data_high

def Creating_Low_Image(list_data):
    list_data=np.array(list_data)
    x_real_data_low=[]
    for i in range(len(list_data)):
        img = list_data[i]
        img = cv2.resize(img, (16, 16), interpolation = cv2.INTER_CUBIC)
        x_real_data_low.append(img)
    x_real_data_low = np.array(x_real_data_low)
    return x_real_data_low

def Error_Image(real_image,predicted_image):
    list_error=[]
    for i in range(len(real_image)):
        S_real=np.sum(real_image[i])
        S_predicted=np.sum(predicted_image[i])
        error = np.float16(S_predicted/S_real)
        error=error**(1/2)
        list_error.append(error)
    return list_error

def del_row_col_Real_Dta(Matrix):
    tam=[]
    for i in range(len(Matrix)):
        m = delete(Matrix[i],[Matrix[i].shape[0]-1],0)
        m = delete(m,s_[m.shape[1]-1],1)
        tam.append(m)
    return np.array(tam)

def del_row_col_Dipole_Model(Matrix):
    tam=[]
    for i in range(len(Matrix)):
        m = delete(Matrix[i],s_[Matrix[i].shape[1]-1],1)
        tam.append(m)
    return np.array(tam)

def vmin_vmax(Matrix_A,Matrix_B):
    max_A = np.max(Matrix_A)
    min_A = np.min(Matrix_A)
    max_B = np.max(Matrix_B)
    min_B = np.min(Matrix_B)
    vmax = max(max_A,max_B)
    vmin = min(min_A,min_B)
    return vmax,vmin
"""
#Read Dipole Model Data
"""
path_file = 'C:/Users/admin/Documents/GPBL/diffusers/Data_Dipole_Model/step_t'

x_data,name_data = Load_Data_Train_Test(path_file)
print(x_data.shape)

x_both_data=x_data
x_both_data=np.array(x_both_data)
print(x_both_data.shape)
"""
#Normalize Dipole Model Data
"""
a = x_both_data.shape[0]
b = x_both_data.shape[1]
c = x_both_data.shape[2]
print(a,b,c)  

data_normalize = x_both_data.reshape(a*b,c)
scaler = StandardScaler()
standardized = scaler.fit_transform(data_normalize)
print(standardized.shape)
data_normalize_reshape=standardized.reshape(a,b,c)
print(data_normalize_reshape.shape)
"""
#Save normalized data into another folder
"""
output_root = "C:/Users/admin/Documents/GPBL/diffusers/step_t_Normalized"
os.makedirs(output_root, exist_ok=True)

print("Số file:", len(name_data))
print("Ví dụ tên file:", name_data[:5])

for idx, name in enumerate(name_data):
    rel_dir = os.path.dirname(name)
    out_dir = os.path.join(output_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(output_root, name)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data_normalize_reshape[idx]:
            writer.writerow(row.tolist())

print(f"✅ Đã lưu toàn bộ dữ liệu normalize vào: {output_root}")
