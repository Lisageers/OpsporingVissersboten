import arcgis
import arcpy
from PIL import Image,ExifTags
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import cos, tan, sin, atan, radians
from os.path import exists

def visualiseDetectionFromPath(path, n):
    model = arcgis.learn.YOLOv3()
    for i in range(0,n,2):
        img_path = f"{path}{494+i}.JPG"
        img_path = r"C:\\Users\\lgeers\\Pictures\\Lisa Den Oever 2022 15 juli 01\\DJI_0546.JPG" 
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
    metadict = {}
    f = img_path
    fd = open(f, encoding = 'latin-1')
    d= fd.read()
    xmp_start = d.find('<x:xmpmeta')
    xmp_end = d.find('</x:xmpmeta')
    xmp_str = d[xmp_start:xmp_end+12]
    # print(xmp_str, "\n")
    s = xmp_str
    lat_i = s.find('GpsLatitude')
    metadict['latitude'] = float(s[lat_i+14:lat_i+24])
    
    long_i = s.find('GpsLongitude')
    metadict['longitude'] = float(s[long_i+15:long_i+24])

    h_i = s.find('RelativeAltitude')
    metadict['altitude'] = float(s[h_i+18:h_i+24])
    
    yaw_i = s.find('GimbalYawDegree')
    metadict['yaw'] =  float(s[yaw_i+17:yaw_i+23].replace('"', ''))
    if metadict['yaw'] < 0:
        metadict['yaw'] = 360 - abs(metadict['yaw'])

    pitch_i = s.find('GimbalPitchDegree')
    metadict['pitch'] =  float(s[pitch_i+19:pitch_i+25])

    # niet altijd hetzelfde aantal nummers!!
    roll_i = s.find('GimbalRollDegree')
    metadict['roll'] =  float(s[roll_i+18:roll_i+23])

    img = Image.open(img_path)
    # kan sneller
    exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    metadict['focallength'] =  float(exif['FocalLength'])
    metadict['pixelwidth'] =  float(exif['ImageWidth'])
    metadict['pixelheight'] =  float(exif['ImageLength'])

    metadict['camxdim'] = 6.4
    metadict['camydim'] = 4.8

    return metadict


def searchPredictions(prediction):
    pred_boats = []
    for i, p in enumerate(prediction[1]):
        if p == "boat":
            pred_boats.append((prediction[0][i], prediction[1][i], prediction[2][i]))
    return pred_boats

def clearLayer():
    arcpy.env.overwriteOutput = True

    inputFCL =  r"C:\Users\lgeers\Documents\ArcGIS\Projects\Opsporingvissersboten\Opsporingvissersboten.gdb\calculatedpolygon"
    array = arcpy.Array([arcpy.Point(0,0),
                        arcpy.Point(0,0),
                        arcpy.Point(0,0),
                        arcpy.Point(0,0)
                        ])
    geom = arcpy.Polygon(array,arcpy.SpatialReference(4326))  
    arcpy.CopyFeatures_management([geom], inputFCL)
    arcpy.AddField_management(inputFCL, "Path", "TEXT")
    with arcpy.da.UpdateCursor(inputFCL, ["SHAPE@"]) as uCur:
        for row in uCur:
            uCur.deleteRow()

    inputFCL =  r"C:\Users\lgeers\Documents\ArcGIS\Projects\Opsporingvissersboten\Opsporingvissersboten.gdb\dronepoint"
    geom = arcpy.PointGeometry(arcpy.Point(0,0),arcpy.SpatialReference(4326))
    arcpy.CopyFeatures_management([geom], inputFCL)
    arcpy.AddField_management(inputFCL, "Path", "TEXT")
    with arcpy.da.UpdateCursor(inputFCL, ["SHAPE@XY", "Path"]) as uCur:
        for row in uCur:
            uCur.deleteRow()

def pointToMap(p, str):
    arcpy.env.overwriteOutput = True

    array = arcpy.Array([arcpy.Point(p[1][0], p[0][0]),
                        arcpy.Point(p[1][1], p[0][1]),
                        arcpy.Point(p[1][2], p[0][2]),
                        arcpy.Point(p[1][3], p[0][3])
                        ])
    polygon = arcpy.Polygon(array, arcpy.SpatialReference(4326))  
    inputFCL =  r"C:\Users\lgeers\Documents\ArcGIS\Projects\Opsporingvissersboten\Opsporingvissersboten.gdb\calculatedpolygon"
    with arcpy.da.InsertCursor(inputFCL,["SHAPE@", "Path"]) as iCur:
        iCur.insertRow([polygon, str])
   
    inputFCL =  r"C:\Users\lgeers\Documents\ArcGIS\Projects\Opsporingvissersboten\Opsporingvissersboten.gdb\dronepoint"
    pnt= arcpy.Point(p[2][1],p[2][0])
    geom = arcpy.PointGeometry(pnt,arcpy.SpatialReference(4326))
    with arcpy.da.InsertCursor(inputFCL,["SHAPE@XY", "Path"]) as iCur:
        iCur.insertRow([geom, str])



def localise(img_path, prediction):
    metadata = getMetadata(img_path)

    # calculate meters per pixel
    viewangl_x = 2 * atan(metadata['camxdim']/(2*metadata['focallength']))
    viewangl_y = 2 * atan(metadata['camydim']/(2*metadata['focallength']))
    mmp_x = (2 * metadata['altitude'] * tan((viewangl_x/2))) / metadata['pixelwidth']
    mmp_y = (2 * metadata['altitude'] * tan((viewangl_y/2))) / metadata['pixelheight']

    # rotate using yaw
    a = radians(metadata['yaw'])
    pixel_x = [prediction[0][0], prediction[0][0]+ prediction[0][2]]
    pixel_y = [prediction[0][1], prediction[0][1]+ prediction[0][3]]


    middle_x, middle_y = round(metadata['pixelwidth'] / 2), round(metadata['pixelheight'] / 2)
    pixeldist_x = [i - middle_x for i in pixel_x]
    pixeldist_y = [i - middle_y for i in pixel_y]

    bounding_box = [(pixeldist_x[0], pixeldist_y[0]), (pixeldist_x[1], pixeldist_y[0]),
                    (pixeldist_x[1], pixeldist_y[1]), (pixeldist_x[0], pixeldist_y[1])]

    rot_y = [p[0] * cos(a) + p[1] * sin(a) for p in bounding_box]
    rot_x = [p[0] * -sin(a) + p[1] * cos(a) for p in bounding_box]
    
    # update lat long
    x_lat = [((x*mmp_x) / 111319.9) + metadata['latitude'] for x in rot_x]
    y_long = [((y*mmp_y) / 111319.9) + metadata['longitude'] for y in rot_y]

    return (x_lat, y_long, (metadata['latitude'], metadata['longitude']))


def opsporingsLoop(path, n):
    model = arcgis.learn.YOLOv3()
    for i in range(0,n,1):
        img_path = f"{path}{100+i}.JPG"
        # img_path = r"C:\\Users\\lgeers\\Pictures\\Lisa Den Oever 2022 15 juli 01\\DJI_0450.JPG"
        # img_path = r"C:\\Users\\lgeers\\Pictures\\Lisa Den Oever 2022 15 juli 01\\DJI_0590.JPG"  
        if exists(img_path):
            print(img_path)
            prediction = model.predict(img_path)
            filtered_prediction = searchPredictions(prediction)

            if filtered_prediction:
                for pred_boat in filtered_prediction:
                    coords = localise(img_path, pred_boat)
                    pointToMap(coords, img_path)



def main():
    clearLayer()
    opsporingsLoop(r"C:\\Users\\lgeers\\Pictures\\Lisa Den Oever 2022 15 juli 01\\DJI_0", 527)
    # visualiseDetectionFromPath(r"C:\\Users\\lgeers\\Pictures\\Lisa Den Oever 2022 15 juli 01\\DJI_0", 2) 
#    img_path = r"C:\Users\lgeers\OneDrive - Esri Nederland\Lisa Den Oever 2022 15 juli 01\DJI_0098.JPG"
#    readMetadata(img_path)






if __name__ == "__main__":
    main()