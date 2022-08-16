import arcgis
import arcpy
from PIL import Image,ExifTags
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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


def localise(img_path, prediction):
    metadata = getMetadata(img_path)
    print(metadata)
    #calculate coordinates from metadata


def getMetadata(img_path):
    metadict = {}
    f = img_path
    fd = open(f, encoding = 'latin-1')
    d= fd.read()
    xmp_start = d.find('<x:xmpmeta')
    xmp_end = d.find('</x:xmpmeta')
    xmp_str = d[xmp_start:xmp_end+12]
    print(xmp_str, "\n")
    s = xmp_str
    lat_i = s.find('GpsLatitude')
    lat = float(s[lat_i+14:lat_i+24])
    long_i = s.find('GpsLongitude')
    long = float(s[long_i+15:long_i+24])
    print(s.find('GimbalPitchDegree'))
    return (lat, long)


def searchPredictions(prediction):
    for i, p in enumerate(prediction[1]):
        if p == "boat":
            return (prediction[0][i], prediction[1][i], prediction[2][i])
    return (None, None, None)


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


def opsporingLoop(path, n):
    model = arcgis.learn.YOLOv3()
    for i in range(0,n,2):
        img_path = f"{path}{494+i}.JPG"
        prediction = model.predict(img_path)
        filtered_prediction = searchPredictions(prediction)
        # print(prediction)
        if filtered_prediction[1] == "boat":
            localise(img_path, filtered_prediction)
        # print(filtered_prediction)


def main():
    opsporingLoop(r"C:\\Users\\lgeers\\Pictures\\Lisa Den Oever 2022 15 juli 01\\DJI_0", 2)
#    visualiseDetectionFromPath(r"C:\\Users\\lgeers\\Pictures\\Lisa Den Oever 2022 15 juli 01\\DJI_0", 39) 
#    img_path = r"C:\Users\lgeers\OneDrive - Esri Nederland\Lisa Den Oever 2022 15 juli 01\DJI_0098.JPG"
#    readMetadata(img_path)


if __name__ == "__main__":
    main()