# import sys

# sys.path.append(r'c:\users\lgeers\appdata\local\programs\python\python310\lib\site-packages')

from re import I
import arcgis
import arcpy
import cv2
from PIL import Image,ExifTags
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from math import cos, tan, sin, atan, radians
from os.path import exists
import os.path
import io
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import urllib.request
import torch

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


def getMetadata(img):
    metadict = {}
    img = Image.open(img)
    xmp = img.getxmp()['xmpmeta']['RDF']['Description']
 
    metadict['latitude'] = float(xmp['GpsLatitude'])
    metadict['longitude'] = float(xmp['GpsLongitude'])
    metadict['altitude'] = float(xmp['RelativeAltitude'])
    metadict['yaw'] =  float(xmp['GimbalYawDegree'])
    if metadict['yaw'] < 0:
        metadict['yaw'] = 360 - abs(metadict['yaw'])
    
    # kan sneller
    # exif2 = img.getexif()
    # exif3 = exif2._get_merged_dict()
    # # exif4 = exif2._get_ifd_dict()
    exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
    metadict['focallength'] =  float(exif['FocalLength'])
    # metadict['pixelwidth'] =  float(exif['ImageWidth'])
    # metadict['pixelheight'] =  float(exif['ImageLength'])

    metadict['pixelwidth'] =  416
    metadict['pixelheight'] =  416
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

def pointToMap(p, id):
    arcpy.env.overwriteOutput = True

    array = arcpy.Array([arcpy.Point(p[1][0], p[0][0]),
                        arcpy.Point(p[1][1], p[0][1]),
                        arcpy.Point(p[1][2], p[0][2]),
                        arcpy.Point(p[1][3], p[0][3])
                        ])
    polygon = arcpy.Polygon(array, arcpy.SpatialReference(4326))  
    inputFCL =  r"C:\Users\lgeers\Documents\ArcGIS\Projects\Opsporingvissersboten\Opsporingvissersboten.gdb\calculatedpolygon"
    with arcpy.da.InsertCursor(inputFCL,["SHAPE@", "Path"]) as iCur:
        iCur.insertRow([polygon, f"https://drive.google.com/file/d/{id}"])
   
    inputFCL =  r"C:\Users\lgeers\Documents\ArcGIS\Projects\Opsporingvissersboten\Opsporingvissersboten.gdb\dronepoint"
    pnt= arcpy.Point(p[2][1],p[2][0])
    geom = arcpy.PointGeometry(pnt,arcpy.SpatialReference(4326))
    with arcpy.da.InsertCursor(inputFCL,["SHAPE@XY", "Path"]) as iCur:
        iCur.insertRow([geom, f"https://drive.google.com/file/d/{id}"])

def pointToOnline(lat, long, id):
    gis = arcgis.GIS("HOME")
    items = gis.content.search("id:241f2b13bf3c468d912fb7bbf817ef28")
    fl = items[0].layers[0]
    newfeature = arcgis.features.Feature({"x":long,"y":lat,"spatialReference":{"wkid":4326}}, {"Path": f"https://drive.google.com/file/d/{id}"})
    results = fl.edit_features(adds = [newfeature])


def polygonToOnline(p, id):
    gis = arcgis.GIS("HOME")

    print("Searching items")
    items = gis.content.search("id:8af5c0fd78c24725b700e6fe5e980bba")
    fl = items[0].layers[0]
    # newfeature = arcgis.features.Feature({"x":p[1][0],"y":p[0][0],"spatialReference":{"wkid":4326}}, {"Path": f"https://drive.google.com/file/d/{id}"})
    newfeature = arcgis.features.Feature({"rings": [[[p[1][0], p[0][0]],[p[1][1], p[0][1]],[p[1][2], p[0][2]],
    [p[1][3], p[0][3]]]],"spatialReference":{"wkid":4326}}, {"Path": f"https://drive.google.com/file/d/{id}"})
    results = fl.edit_features(adds = [newfeature])



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

    rot_x = [p[0] * cos(a) + p[1] * sin(a) for p in bounding_box]
    rot_y = [p[0] * -sin(a) + p[1] * cos(a) for p in bounding_box]
    
    # update lat long
    y_lat = [((y*mmp_y) / 111319.9) + metadata['latitude'] for y in rot_y]
    x_long = [((x*mmp_x) / 111319.9) + metadata['longitude'] for x in rot_x]

    return (y_lat, x_long, (metadata['latitude'], metadata['longitude']))


def download_file(id, service):
    try:
        # pylint: disable=maybe-no-member
        request = service.files().get_media(fileId=id)
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

    except HttpError as error:
        print(F'An error occurred: {error}')
        file = None

    img = cv2.imdecode(np.asarray(bytearray(file.getvalue()), dtype=np.uint8), 1)
    # imS = cv2.resize(img, (960, 540))
    # cv2.imshow('image',imS)
    # cv2.waitKey(10)
    return img, file


def opsporingsLoop():
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    creds = Credentials.from_authorized_user_file(r'C:\Users\lgeers\Documents\OpsporingVissersboten\token.json', SCOPES)
    # create drive api client
    service = build('drive', 'v3', credentials=creds)

    # Call the Drive v3 API
    results = service.files().list(
        pageSize=10, fields="nextPageToken, files(id, name)", q="mimeType='image/jpeg' and name contains 'DJI'").execute()
    items = results.get('files', [])
    file_id = items[0]['id']
    

    stack = [file_id]
    visited = []

    model = arcgis.learn.YOLOv3()
    while stack:

        id = stack.pop(0)
        print(id)
        img, bytes = download_file(id, service)
        print(img.shape)
        img = cv2.resize(img, (416, 416))

        metadata = getMetadata(bytes)
        pointToOnline(metadata['latitude'], metadata['longitude'], id)
        prediction = model.predict(img)
        filtered_prediction = searchPredictions(prediction)

        for pred_boat in filtered_prediction:
            coords = localise(bytes, pred_boat)
            polygonToOnline(coords, id)
        
        visited.append(id)
        print("visited",len(visited))
        print(len(stack))

        results = service.files().list(
        pageSize=10, fields="nextPageToken, files(id, name)",q="mimeType='image/jpeg' and name contains 'DJI'").execute()
        items = results.get('files', [])
        items.reverse()
        for pic in items:
            if pic['id'] in visited:
                break
            stack.append(pic['id'])
            print(pic)
            


def main():
    # clearLayer()
    opsporingsLoop()
    # visualiseDetectionFromPath(r"C:\\Users\\lgeers\\Pictures\\Lisa Den Oever 2022 15 juli 01\\DJI_0", 2) 
#    img_path = r"C:\Users\lgeers\OneDrive - Esri Nederland\Lisa Den Oever 2022 15 juli 01\DJI_0098.JPG"
#    readMetadata(img_path)



if __name__ == "__main__":
    main()