import os
import cv2
import numpy as np
import math
import shutil

udisk = "/media/exia/KINGSTON/"
local = "./"

imread_labels_path = udisk + "labels/"            #从上交开源的https://github.com/Spphire/RM-labeling-tool下载的labels的文件夹路径
save_labels_path = udisk + "converted_labels/"    #更改为关键点检测的labels
image_path = udisk + "images_jpg/"                #下载的图片文件夹的路径
save_images_path = udisk + "converted_images/"

train_images = udisk + "train/images/"             #分开后的训练集路径
train_lables = udisk + "train/labels/"             

val_images = udisk + "val/images/"                 #分开后的验证集路径
val_labels = udisk + "val/labels/"

train_ratio = 0.8                           #训练集占比

mode = "copy"                               #选择分离模式，move or copy,注意，train和val文件夹会先清空

minification = 8                            #图片缩小尺寸

def distance(x1,y1,x2,y2):
    dis_2 = (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)
    return math.sqrt(dis_2)

#用来检查关键点的顺序

def draw():
    file_name = os.listdir(imread_labels_path)
    name = [()]
    j = 1
    for i in file_name:
        name.append(list(os.path.splitext(i)))
        # print(name[j])
        # print(i)
        j = j+1

    j = 1
    for i in file_name:
        read_image = cv2.imread(image_path + name[j][0] + ".jpg")
        data = []
        size = read_image.shape
        cols = size[1]
        rows = size[0]
        cols = cols / minification 
        rows = rows / minification
        f = open(imread_labels_path + i,'r')
        while True:

            Lines = f.readline()
            if Lines:
                Line = Lines.split()
                line = []
                for n in range(0,11):
                    line.append(float(Line[n]))
                # print(line)
                # for j in range(0,11):
                #     print(line[j])
                dst = cv2.resize(read_image,(int(cols),int(rows)))

                cv2.circle(dst,(int((line[1])*cols),int((line[2])*rows)),3,(114,514,191),-1)
                cv2.circle(dst,(int((line[3])*cols),int((line[4])*rows)),3,(11,51,19),-1)
                cv2.circle(dst,(int((line[5])*cols),int((line[6])*rows)),3,(114,14,91),-1)
                cv2.circle(dst,(int((line[7])*cols),int((line[8])*rows)),3,(214,214,191),-1)
                cv2.circle(dst,(int((line[9])*cols),int((line[10])*rows)),3,(214,214,191),-1)

                cv2.putText(dst,"0", (int(float(line[1])*cols),int(float(line[2])*rows)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 1, cv2.LINE_AA)
                cv2.putText(dst,"1", (int(float(line[3])*cols),int(float(line[4])*rows)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 1, cv2.LINE_AA)
                cv2.putText(dst,"2", (int(float(line[5])*cols),int(float(line[6])*rows)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 1, cv2.LINE_AA)
                cv2.putText(dst,"3", (int(float(line[7])*cols),int(float(line[8])*rows)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 1, cv2.LINE_AA)
                cv2.putText(dst,"4", (int(float(line[9])*cols),int(float(line[10])*rows)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 1, cv2.LINE_AA)

                Class = line[0]

                # 能量机关中心点的数据处理

                if Class == 0 or Class == 3:          
                    symbol_w = 0
                    symbol_h = 0
                    symbol_center = [0.0,0.0]

                    symbol_center[0] = (line[1] + line[3] + line[5] + line[7] + line[9])/5
                    symbol_center[1] = (line[2] + line[4] + line[6] + line[8] + line[10])/5
                    dislist = []
                    for n in range(1,6):
                        dislist.append(distance(symbol_center[0],symbol_center[1],line[2*n-1],line[2*n]))
                    dis = max(dislist)
                    symbol_h = dis*2
                    symbol_w = dis*2
                    print("symbol_h:",symbol_h,"\n","symbol_w:",symbol_w)

                    cv2.circle(dst,(int(symbol_center[0]*cols),int(symbol_center[1] * rows)),3,(114,191,0),-1)
                    cv2.circle(dst,(int(symbol_center[0]*cols),int(symbol_center[1] * rows)),int(dis*math.sqrt(cols*cols+rows*rows)),(114,191,0))

                    cv2.putText(dst,"5", (int(symbol_center[0]*cols),int(symbol_center[1] * rows)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 1, cv2.LINE_AA) 

                    line_data = [str(int(Class)),str(symbol_center[0]),str(symbol_center[1]),str(dis*2),str(dis*2)]

                    for n in range(1,11):
                        line_data.append(str(line[n]))
                    line_data.append('\n')

                    data.append(line_data)
                    
                    print("line_data:\n",line_data)

                # 激活和未激活的能量机关叶片的数据处理
                else:
                    rune_center = [0.0,0.0]
                    rune_center[0] = (line[1] + line[3] + line[5] + line[7] + line[9] ) / 5
                    rune_center[1] = (line[2] + line[4] + line[6] + line[8] + line[10]) / 5
                    rune_w = distance(line[5],line[6],line[9],line[10])
                    rune_h = distance(line[7],line[8], (line[1] + line[3]) / 2 , (line[2]+line[4]) / 2)

                    cv2.circle(dst,(int(rune_center[0] * cols),int(rune_center[1] * rows)),3,(212,41,10),-1)

                    line_data = [str(int(Class)),str(rune_center[0]),str(rune_center[1]),str(rune_w),str(rune_h)]
                    for n in range(1,11):
                        line_data.append(str(line[n]))
                    print("line_data:\n",line_data)
                    data.append(line_data)

                cv2.imshow("modify",dst)
                if cv2.waitKey(1000) == 'q':
                    exit()
                cv2.destroyAllWindows()



            else:
                f.close()
                j = j+1
                print("writing data:\n",data)

                break


    return 1

#写入修改后的labels
def write():
    file_name = os.listdir(imread_labels_path)
    name = [()]
    j = 1
    for i in file_name:
        name.append(list(os.path.splitext(i)))
        # print(name[j])
        # print(i)
        j = j+1

    j = 1
    for i in file_name:
        read_image = cv2.imread(image_path + name[j][0] + ".jpg")
        data = []
        size = read_image.shape
        cols = size[1]
        rows = size[0]
        cols = cols / minification 
        rows = rows / minification

        dst = cv2.resize(read_image,(int(cols),int(rows)))
        cv2.imwrite(save_images_path + name[j][0] + '.jpg',dst)
        


        f = open(imread_labels_path + i,'r')
        
        while True:

            Lines = f.readline()
            if Lines:
                Line = Lines.split()
                line = []
                for n in range(0,11):
                    line.append(float(Line[n]))


                Class = line[0]

                # 能量机关中心点的数据处理

                if Class == 0 or Class == 3:          
                    symbol_w = 0
                    symbol_h = 0
                    symbol_center = [0.0,0.0]

                    symbol_center[0] = (line[1] + line[3] + line[5] + line[7] + line[9])/5
                    symbol_center[1] = (line[2] + line[4] + line[6] + line[8] + line[10])/5
                    dislist = []
                    for n in range(1,6):
                        dislist.append(distance(symbol_center[0],symbol_center[1],line[2*n-1],line[2*n]))
                    dis = max(dislist)
                    symbol_h = dis*2
                    symbol_w = dis*2
                    print("symbol_h:",symbol_h,"\n","symbol_w:",symbol_w)

                    line_data = [str(int(Class)),str(symbol_center[0]),str(symbol_center[1]),str(dis*2),str(dis*2)]

                    for n in range(1,11):
                        line_data.append(str(line[n]))
                    line_data.append('\n')

                    data.append(line_data)
                    
                    print("line_data:\n",line_data)

                # 激活和未激活的能量机关叶片的数据处理
                else:
                    armor_center = [0.0,0.0]
                    armor_center[0] = (line[1] + line[3] + line[5] + line[7] + line[9] ) / 5
                    armor_center[1] = (line[2] + line[4] + line[6] + line[8] + line[10]) / 5
                    rune_w = distance(line[5],line[6],line[9],line[10])
                    rune_h = distance(line[7],line[8], (line[1] + line[3]) / 2 , (line[2]+line[4]) / 2)

                    cv2.circle(dst,(int(armor_center[0] * cols),int(armor_center[1] * rows)),3,(212,41,10),-1)

                    line_data = [str(int(Class)),str(armor_center[0]),str(armor_center[1]),str(rune_w),str(rune_h)]
                    for n in range(1,11):
                        line_data.append(str(line[n]))
                    line_data.append('\n')
                    print("line_data:\n",line_data)
                    data.append(line_data)


            else:
                f.close()
                j = j+1

                print("writing data:\n",data)
                f = open(save_labels_path + i,'w')
                for n in data:
                    for m in n:
                        f.write(m)
                        if m != "\n":
                            f.write(" ")
                
                f.close()
                break
        if j == (len(file_name)+1):
            print("the rows is :",int(rows))
            print("the cols is :",int(cols))


    return 1

#用于将分开后的训练集和验证集清空
def rm():
    train_i = os.listdir(train_images)
    val_i = os.listdir(val_images)
    train_l = os.listdir(train_lables)
    val_l = os.listdir(val_labels)
    if len(train_i) != 0:
        for i in range (0,len(train_i)):
            os.remove(train_images + '/' + train_i[i])
            os.remove(train_lables + '/' + train_l[i])
        print("train clear!")

    if len(val_i) !=0:
        for i in range(0,len(val_l)):
            os.remove(val_images + '/' + val_i[i])
            os.remove(val_labels + '/' + val_l[i])
        print("val clear!")
    return True




#用来将数据集分为训练集和验证集    
def separate():
    image = os.listdir(image_path)
    label = os.listdir(save_labels_path)

    if len(image) != len(label):
        print("the number of images don't equal to the the number of labels")
        exit()
    else :
        length = len(label)

    rm()

    if mode == "copy":

        for i in range (0,int(length * train_ratio)):
            name = image[i].split(".")
            print("coping : ",name[0])
            shutil.copy(save_images_path + name[0] + ".jpg",train_images)
            shutil.copy(save_labels_path + name[0] + ".txt",train_lables)

        for i in range(int(train_ratio * length) , length):
            name = image[i].split(".")
            print("coping : ",name[0])
            shutil.copy(save_images_path + name[0] + ".jpg",val_images)
            shutil.copy(save_labels_path + name[0] + ".txt",val_labels)

    elif mode == "move":

        for i in range (0,int(length * train_ratio)):
            name = image[i].split(".")
            print("coping : ",name[0])
            shutil.move(save_images_path + name[0] + ".jpg",train_images)
            shutil.move(save_labels_path + name[0] + ".txt",train_lables)

        for i in range(int(train_ratio * length) , length):
            name = image[i].split(".")
            print("coping : ",name[0])
            shutil.move(save_images_path + name[0] + ".jpg",val_images)
            shutil.move(save_labels_path + name[0] + ".txt",val_labels)

    image = os.listdir(train_images)
    label = os.listdir(val_images)

    print("the number of train is :",len(image))
    print("the number of val is :",len(label))


#用于转换为
# https://github.com/qinggangwu/yolov7-pose_Npoint_Ncla.git
# 相应的格式

def my_label():
    images = os.listdir(train_images)
    f = open("./train.txt","w")
    for i in images:
        string = train_images + i + '\n'
        f.write(string)
    f.close()
    images = os.listdir(val_images)
    f = open("./val.txt","w")
    for i in images:
        string = val_images + i + '\n'
        f.write(string)
    f.close()
    print("my label :finish!")

#得到数据集的数量和图片的行和列（假设所有图片的尺寸相同）

def get_param():
    images = os.listdir(train_images)
    labels = os.listdir(train_lables)
    img = cv2.imread(train_images + images[0])
    size = img.shape
    rows = size[0]
    cols = size[1]
    print("train's images:",len(images),"\ntrain's labels:",len(labels))
    images = os.listdir(val_images)
    labels = os.listdir(val_labels)
    print("val's images:",len(images),"\nval's labels:",len(labels))
    print("rows:",rows,"\ncols:",cols)

if __name__=="__main__":
    draw()
    # write()
    # separate()
    # my_label()
    # get_param()



