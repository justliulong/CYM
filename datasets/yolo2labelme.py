'''
可以将yolov实例分割生成的txt格式的标注转为json，可以使用labelme查看标注
该方法可以用于辅助数据标注，yolo的数据格式为txt,单行的格式如下：
<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
采用这种格式、 <class-index> 是对象的类的索引，而 <x1> <y1> <x2> <y2> ... <xn> <yn> 是对象的分割掩码的边界坐标。坐标之间用空格隔开。
'''

import os
import glob
import numpy as np
import cv2
import json


def convert_txt_to_labelme_json(txt_path, image_path, output_dir, class_name):
    """
    将文本文件转换为LabelMe格式的JSON文件。
    此函数处理文本文件中的数据，将其转换成LabelMe标注工具使用的JSON格式。包括读取图像，
    解析文本文件中的标注信息，并生成相应的JSON文件。
    :param txt_path: 文本文件所在的路径
    :param image_path: 图像文件所在的路径
    :param output_dir: 输出JSON文件的目录
    :param class_name: 类别名称列表，索引对应类别ID
    :param image_fmt: 图像文件格式，默认为'.jpg'
    :return:
    """
    # 获取所有文本文件路径
    txts = glob.glob(os.path.join(txt_path, "*.txt"))
    for txt in txts:
        # 初始化LabelMe JSON结构
        labelme_json = {
            'version': '5.4.1',  # labelme版本号
            'flags': {},
            'shapes': [],
            'imagePath': None,
            'imageData': None,
            'imageHeight': None,
            'imageWidth': None,
        }
        # 获取文本文件名
        txt_name = os.path.basename(txt)
        # 根据文本文件名生成对应的图像文件名，优先找'.jpg'，再找'.png'
        if os.path.exists(os.path.join(image_path, txt_name.split(".")[0] + ".jpg")):
            image_name = txt_name.split(".")[0] + ".jpg"
        elif os.path.exists(os.path.join(image_path, txt_name.split(".")[0] + ".png")):
            image_name = txt_name.split(".")[0] + ".png"
        else:
            raise Exception('txt 文件={},找不到对应的图像={}.jpg|.png'.format(txt,
                                                                    os.path.join(image_path,
                                                                                 txt_name.split(".")[0])))

        labelme_json['imagePath'] = image_name
        # 构造完整图像路径
        image_name = os.path.join(image_path, image_name)
        # 读取图像
        image = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 获取图像高度和宽度
        h, w = image.shape[:2]
        labelme_json['imageHeight'] = h
        labelme_json['imageWidth'] = w
        # 读取文本文件内容
        with open(txt, 'r') as t:
            lines = t.readlines()
            for line in lines:
                point_list = []
                content = line.split(' ')
                # 根据类别ID获取标签名称
                label = class_name[int(content[0])]  # 标签
                # 解析点坐标
                for index in range(1, len(content)):
                    if index % 2 == 1:  # 下标为奇数，对应横坐标
                        x = (float(content[index])) * w
                        point_list.append(x)
                    else:  # 下标为偶数，对应纵坐标
                        y = (float(content[index])) * h
                        point_list.append(y)
                # 将点列表转换为二维列表，每两个值表示一个点
                point_list = [point_list[i:i + 2] for i in range(0, len(point_list), 2)]
                # 构造shape字典
                shape = {
                    'label': label,
                    'points': point_list,
                    'group_id': None,
                    'description': None,
                    'shape_type': 'polygon',
                    'flags': {},
                    'mask': None
                }
                labelme_json['shapes'].append(shape)
            # 生成JSON文件名
            json_name = txt_name.split('.')[0] + '.json'
            json_name_path = os.path.join(output_dir, json_name)
            # 写入JSON文件
            fd = open(json_name_path, 'w')
            json.dump(labelme_json, fd, indent=2)
            fd.close()
            # 输出保存信息
            print("save json={}".format(json_name_path))


if __name__ == '__main__':
    txt_path = r'./DSB2018/labels/val'
    image_path = r'./DSB2018/images/val'
    output_dir = r'./DSB2018/json/val'
    # 标签列表
    class_name = ['cell']  # 标签类别名
    convert_txt_to_labelme_json(txt_path, image_path, output_dir, class_name)
