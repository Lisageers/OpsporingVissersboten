import arcgis
import arcpy
from PIL import Image,ExifTags
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import cos, tan, sin

def visualiseDetectionFromPath(path, n):
    model = arcgis.learn.YOLOv3()
    for i in range(0,n,2):
        img_path = f"{path}{494+i}.JPG"
        prediction = model.predict(img_path)
        
        x = np.array(Image.open(img_path), dtype=np.uint8)
        fig, ax = plt.subplots(1)
        ax.imshow(x)

        # Create a Rectangle patches of bounding boxes from prediction
        bbs = prediction[0]
        for i, bb  in enumerate(bbs):
            rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='r', facecolor="none")
            # Add the patch to the Axes
            ax.add_patch(rect)
            # add label
            ax.text(bb[0], bb[1], prediction[1][i], ha='left', va='bottom',color='red')

        plt.show()
        print(prediction)
        print("\n")


def readMetadata(img_path):
    img = Image.open(img_path)
    exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    print(exif)

    f = img_path
    fd = open(f, encoding = 'latin-1')
    d= fd.read()
    xmp_start = d.find('<x:xmpmeta')
    xmp_end = d.find('</x:xmpmeta')
    xmp_str = d[xmp_start:xmp_end+12]
    print(xmp_str)


def getMetadata(img_path):
    f = img_path
    fd = open(f, encoding = 'latin-1')
    d= fd.read()
    xmp_start = d.find('<x:xmpmeta')
    xmp_end = d.find('</x:xmpmeta')
    xmp_str = d[xmp_start:xmp_end+12]
    # print(xmp_str, "\n")
    s = xmp_str
    lat_i = s.find('GpsLatitude')
    lat = float(s[lat_i+14:lat_i+24])
    
    long_i = s.find('GpsLongitude')
    long = float(s[long_i+15:long_i+24])

    h_i = s.find('RelativeAltitude')
    h = float(s[h_i+18:h_i+24])
    
    yaw_i = s.find('GimbalYawDegree')
    yaw =  float(s[yaw_i+17:yaw_i+23])

    pitch_i = s.find('GimbalPitchDegree')
    pitch =  float(s[pitch_i+19:pitch_i+25])

    # niet altijd hetzelfde aantal nummers!!
    roll_i = s.find('GimbalRollDegree')
    roll =  float(s[roll_i+18:roll_i+23])

    img = Image.open(img_path)
    # kan sneller
    exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    cf =  float(exif['FocalLength'])

    return (lat, long, h, yaw, pitch, roll, cf)


def searchPredictions(prediction):
    for i, p in enumerate(prediction[1]):
        if p == "boat":
            return (prediction[0][i], prediction[1][i], prediction[2][i])
    return (None, None, None)

def clearLayer():
    arcpy.env.overwriteOutput = True
    inputFCL =  r"C:\Users\lgeers\Documents\ArcGIS\Projects\Opsporingvissersboten\Opsporingvissersboten.gdb\dronepoint"
    pointGeom = arcpy.PointGeometry(arcpy.Point(0,0),arcpy.SpatialReference(4326))
    
    arcpy.CopyFeatures_management([pointGeom], inputFCL)
    with arcpy.da.UpdateCursor(inputFCL, ["SHAPE@XY"]) as uCur:
        for row in uCur:
            uCur.deleteRow()

def pointToMap(p):
    arcpy.env.overwriteOutput = True
    # project = arcpy.mp.ArcGISProject(r"C:\Users\lgeers\Documents\ArcGIS\Projects\Opsporingvissersboten\Opsporingvissersboten.aprx")
    # maps = project.listMaps()
    # print(maps)
    # for map in maps:
    #     print(map.name)
    # map = maps[-1]
    
    pnt= arcpy.Point(p[1],p[0])
    pointGeom = arcpy.PointGeometry(pnt,arcpy.SpatialReference(4326))
   
    inputFCL =  r"C:\Users\lgeers\Documents\ArcGIS\Projects\Opsporingvissersboten\Opsporingvissersboten.gdb\dronepoint"

    with arcpy.da.InsertCursor(inputFCL,["SHAPE@XY"]) as iCur:
        iCur.insertRow(pointGeom)



def localise(img_path, prediction):
    metadata = getMetadata(img_path)
    # print(metadata)
    # print(prediction)
    yaw = metadata[3]
    pitch = metadata[4]
    roll = metadata[5]
    cf = metadata[6]
    x = prediction[0][0]
    y = prediction[0][1]
    a1 = cos(yaw) * cos(pitch)
    a2 = cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll)
    a3 = cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll)
    a4 = sin(yaw) * cos(pitch)
    a5 = sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll)
    a6 = sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll)
    a7 = -sin(pitch)
    a8 = cos(pitch) * sin(roll)
    a9 = cos(pitch) * cos(roll)

    x_geo = (a1*x + a4*y + a7*cf)/(a3*x + a6*y + a9*cf) * -metadata[2] + metadata[0]
    print(x_geo)

    # generate rotation matrix
    # rotM = [[a1, a2, a3], [a4, a5, a6], [a7, a8, a9]]
    #calculate coordinates from metadata
    return metadata


def opsporingsLoop(path, n):
    model = arcgis.learn.YOLOv3()
    for i in range(0,n,2):
        img_path = f"{path}{494+i}.JPG"
        prediction = model.predict(img_path)
        filtered_prediction = searchPredictions(prediction)
        # print(prediction)
        if filtered_prediction[1] == "boat":
            point = localise(img_path, filtered_prediction)
            # pointToMap(point)
        # print(filtered_prediction)


def main():
    # clearLayer()
    opsporingsLoop(r"C:\\Users\\lgeers\\Pictures\\Lisa Den Oever 2022 15 juli 01\\DJI_0", 2)
#    visualiseDetectionFromPath(r"C:\\Users\\lgeers\\Pictures\\Lisa Den Oever 2022 15 juli 01\\DJI_0", 39) 
#    img_path = r"C:\Users\lgeers\OneDrive - Esri Nederland\Lisa Den Oever 2022 15 juli 01\DJI_0098.JPG"
#    readMetadata(img_path)


if __name__ == "__main__":
    main()