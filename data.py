import os
import csv
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import math
from numpy import * 
from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import StandardScaler

def Max(array):
  length=len(array)
  max_tam=-10000
  for i in range(length):
    if(max_tam < max(array[i])):
      max_tam = max(array[i])
  return max_tam

# define a function for peak signal-to-noise ratio (PSNR)
def psnr(target, ref):
         
    # assume RGB image
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(Max(target)/ rmse)

#This is the function which helps me split the string
def Split_String(name):
  draf,draf1=name.split("diffusers")
  return draf1

# =============================================================================
def List_Name_Image(path):
  list_name=[]
  files_path=[os.path.abspath(x) for x in os.listdir(path)]
  #print(files_path)

  for i in files_path:
      list_name.append(Split_String(i))
  return list_name
# =============================================================================

def Load_Data(Path,list_name):
  X=[] #Dùng để lưu dữ liệu load lên từ thư mục
  for i in list_name:
    filename = Path +'/' + i
    #print(filename)
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

      tam_1 = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
      rotated_image = insert(rotated_image,[31],tam_1,1)
      
      #print(rotated_image)

      X.append(rotated_image)
      tam = np.array(X)
      print(len(X))
      f.close
      
  return X 

def Load_Data_Test(Path,list_name):
  X=[] #Dùng để lưu dữ liệu load lên từ thư mục
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

      tam_1 = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
      rotated_image = insert(draft_1,[31],tam_1,1)

      X.append(rotated_image)

      f.close

  return X 

def Load_Data_Train_Test(path_file):
  path = path_file#"/content/drive/MyDrive/Dipole Model/Train_Test_Data/Trainning_Data/Trainning"

  List_Name_Image_Ellipse = List_Name_Image(path)
  
  list_tam=np.array(List_Name_Image_Ellipse)

  y_name =  list_tam

  #print(y_name)
  X_data=Load_Data(path,list_tam)
  
  X_data=np.array(X_data)

  return X_data, np.array(y_name)

def Train_Test_Split(standardized,name_data):

  x_data = np.array(standardized)
  # x_real_data_high = np.array(x_real_data_high)

  y_train, y_test, y_train_name, y_test_name = train_test_split( x_data, name_data, test_size=0.3, random_state=4 )

  x_train=[]
  x_test=[]
  # x_real_data_low=[]

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

  path = name_path #"/content/drive/MyDrive/Dipole Model/Train_Test_Data/Testing_Data/Experiment_1"

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

  if(max_A>max_B):
    vmax=max_A
  else:
    vmax=max_B

  if(min_A<min_B):
    vmin=min_A
  else:
    vmin=min_B
  
  return vmax,vmin

def SSIM(list_img_original,list_img_predicted):
    
    
  result=[]
  for i in range(len(list_img_original)):
    result.append(ssim(list_img_original[i],list_img_predicted[i]))
  return result
"""
#Read Dipole Model Data
"""
path_file = 'C:/Users/admin/Documents/GPBL/diffusers/Data_Dipole_Model'

x_data,name_data = Load_Data_Train_Test(path_file)

print(x_data.shape)

x_both_data=x_data

x_both_data=np.array(x_both_data)

print(x_both_data.shape)
#####################################################################################################################

"""
#Normalize Dipole Model Data
"""
a = x_both_data.shape[0]
b = x_both_data.shape[1]
c = x_both_data.shape[2]
print(a,b,c)  

data_normalize = x_both_data.reshape(a*b,c)

data_normalize.shape

# load data
data = data_normalize
# create scaler
scaler = StandardScaler()
# fit and transform in one step
standardized = scaler.fit_transform(data)
# # inverse transform
# inverse = scaler.inverse_transform(standardized)

print(standardized.shape)

data_normalize_reshape=standardized.reshape(a,b,c)

print(data_normalize_reshape.shape)