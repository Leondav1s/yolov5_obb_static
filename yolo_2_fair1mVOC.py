import os
import cv2
import pathlib
from xml.dom.minidom import Document
 
# 支持多种格式图片,可以自己添加
# 这么做是为了适应不同文件名，比如 'xxx.xxx xx.jpeg'等
def search_file_name(file_path,img_root):
    file_format = ['.jpg','.JPG','.png','.PNG','.JPEG','.jpeg','.tif']
    file_path = pathlib.Path(file_path)
    file_name = file_path.stem
    for endsw in file_format:
        file_judge = file_name + endsw
        if os.path.isfile(os.path.join(img_root,file_judge)):
            return file_name,endsw
    return 'stop',0
 
def yolo2voc(src_img_root, src_label_root, dst_label_root,class_map):
    """
    src_img_root:图片路径（需要获取图片shape）
    src_label_root：输入label路径（YOLO的txt格式）
    dst_label_root：输出label路径（PASCAL VOC的xml格式）
    target：txt格式转为xml格式
    """
    not_have_img = []
    if not os.path.exists(dst_label_root):
        os.makedirs(dst_label_root)
    # 遍历所有txt文件
    for i, src_label_name in enumerate(os.listdir(src_label_root)):
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        src_label_path = os.path.join(src_label_root,src_label_name)
        file_name,endsw = search_file_name(src_label_path,src_img_root)
        if file_name == 'stop':
            not_have_img.append(os.path.basename(src_label_path))
            continue
        with open(src_label_path, 'r') as fr:
            txtlines = fr.readlines()
        src_img_name = file_name + endsw
        src_img_path = os.path.join(src_img_root,src_img_name)
        image_np = cv2.imread(src_img_path)
        Pheight, Pwidth, Pdepth = image_np.shape

        ###source###
        source = xmlBuilder.createElement("source")  # source

        filename = xmlBuilder.createElement("filename")  # filename标签
        filenamecontent = xmlBuilder.createTextNode(src_img_name)
        filename.appendChild(filenamecontent)
        source.appendChild(filename)
        
        origin = xmlBuilder.createElement("origin")
        origincontent = xmlBuilder.createTextNode("GF2/GF3")
        origin.appendChild(origincontent)
        source.appendChild(origin)

        annotation.appendChild(source) 
        ###end of source###
       
        #### size标签###
        size = xmlBuilder.createElement("size")  # size标签

        width = xmlBuilder.createElement("width")  # size子标签width
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)  # size子标签width结束
 
        height = xmlBuilder.createElement("height")  # size子标签height
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)  # size子标签height结束
 
        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)  # size子标签depth结束

        annotation.appendChild(size) 
        ### size标签结束###

        objects = xmlBuilder.createElement("objects")  #  
        for line in txtlines:
            oneline = line.strip().split(" ")
            object = xmlBuilder.createElement("object")  #  

            coordinate = xmlBuilder.createElement("coordinate")  # pixel
            coordinatecontent = xmlBuilder.createTextNode("pixel")
            coordinate.appendChild(coordinatecontent)
            object.appendChild(coordinate)  #  
 
            type = xmlBuilder.createElement("type")  # type
            typeContent = xmlBuilder.createTextNode("rectangle")
            type.appendChild(typeContent)
            object.appendChild(type)  #  
 
            description = xmlBuilder.createElement("description")  #  
            descriptioncontent = xmlBuilder.createTextNode("None")
            description.appendChild(descriptioncontent)
            object.appendChild(description)  #  

            possibleresult = xmlBuilder.createElement("possibleresult")
            picname = xmlBuilder.createElement("name")  #  
            namecontent = xmlBuilder.createTextNode(class_map[oneline[0]])
            picname.appendChild(namecontent)
            possibleresult.appendChild(picname)
            object.appendChild(possibleresult)
            
            points = xmlBuilder.createElement("points")  # bndbox标签
            
            point = xmlBuilder.createElement("point")  # xmin标签
            pointcontent = xmlBuilder.createTextNode(str(oneline[1])+','+str(oneline[2]))
            point.appendChild(pointcontent)  # xmin标签结束
            points.appendChild(point)

            point = xmlBuilder.createElement("point")  # xmin标签
            pointcontent = xmlBuilder.createTextNode(str(oneline[3])+','+str(oneline[4]))
            point.appendChild(pointcontent)  # xmin标签结束
            points.appendChild(point)

            point = xmlBuilder.createElement("point")  # xmin标签
            pointcontent = xmlBuilder.createTextNode(str(oneline[5])+','+str(oneline[6]))
            point.appendChild(pointcontent)  # xmin标签结束
            points.appendChild(point)

            point = xmlBuilder.createElement("point")  # xmin标签
            pointcontent = xmlBuilder.createTextNode(str(oneline[7])+','+str(oneline[8]))
            point.appendChild(pointcontent)  # xmin标签结束
            points.appendChild(point)

            point = xmlBuilder.createElement("point")  # xmin标签
            pointcontent = xmlBuilder.createTextNode(str(oneline[1])+','+str(oneline[2]))
            point.appendChild(pointcontent)  # xmin标签结束
            points.appendChild(point)
            
            object.appendChild(points)  # bndbox标签结束
            objects.appendChild(object)  # object标签结束
        
        annotation.appendChild(objects)  # object标签结束
        dst_label_path = os.path.join(dst_label_root,file_name + '.xml')
        with open(dst_label_path,'w') as f:
            xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        print(src_img_path)
    print('not have img:',not_have_img,'\n',not_have_img.__len__())
 
 
if __name__ == "__main__":
    #注意： opencv 不支持中文路径，路径及文件名需要是英文数字符号等
    src_img_root = "/kaggle/input/fair1m-date/test/images"
    src_label_root = "/kaggle/working/yolov5_obb_static/runs/detect/exp/labels"    
    dst_label_root = "/kaggle/working/yolov5_obb_static/runs/detect/exp/dst_label"
    # txt文件中每列第一个为类别序号，分别对应label名
    class_map = {'0':'Boeing737',
                 '1':'Boeing747',
                 '2':'Boeing777',
                 '3':'Boeing787',
                 '4':'C919',
                 '5':'A220',
                 '6':'A321',
                 '7':'A330',
                 '8':'A350',
                 '9':'ARJ21',
        '10':'other-airplane',
        '11':'Passenger Ship',
        '12':'Motorboat',
        '13':'Fishing Boat',
        '14':'Tugboat',
        '15':'Engineering Ship',
        '16':'Liquid Cargo Ship',
        '17':'Dry Cargo Ship',
        '18':'Warship',
        '19':'other-ship',
        '20':'Small Car',
        '21':'Bus',
        '22':'Cargo Truck',
        '23':'Dump Truck',
        '24':'Van',
        '25':'Trailer',
        '26':'Tractor',
        '27':'Excavator',
        '28':'Truck Tractor',
        '29':'other-vehicle',
        '30':'Basketball Court',
        '31':'Tennis Court',
        '32':'Football Field',
        '33':'Baseball Field',
        '34':'Intersection',
        '35':'Roundabout',
        '36':'Bridge'}
    #
    #step1 对于结果，首先添加空的情况
    for i in range(18021):
        strname = src_label_root+'/'+str(i)+'.txt'
        if not os.path.exists(strname):
            print(strname)
            file = open(strname,'w')
            file.close()

    #step2 生成xml
    yolo2voc(src_img_root, src_label_root, dst_label_root,class_map)