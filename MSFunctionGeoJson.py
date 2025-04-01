# from ApuCellGlobal.MSApuCellGlobalImport import *
# from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
# from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
# from ApuCellGlobal.GeneralFunction.MSFunctionImageOperator import CFunctionImageOperator

import tifffile as tiff

from enum import Enum, unique
import numpy as np
import json
import cv2 
import os

@unique
class EnumTypeGeometry(Enum):

    Point = 1
    MultiPoint = 2
    LineString = 3
    MultiLineString = 4
    Polygon = 5
    MultiPolygon = 6
    GeometryCollection = 7

class CFunctionGeoJson:

    @staticmethod
    def ReadGeoJson(path_file:str):

        # if not os.path.exists(path_file):
        #     CLogging.LogForLackPath(path_file)

        with open(path_file, 'r') as f:
            data = json.load(f)
        return data

    @staticmethod
    def ParseGeoJson(data):

        features = data['features']
        label_point_list = []

        for feature in features:
            # 获取要素的属性信息
            # properties = feature['properties']
            # 获取要素的几何数据
            geometry = feature['geometry']
            # geometry["type"] =
            if geometry["type"] in EnumTypeGeometry.__members__:
                if EnumTypeGeometry[geometry["type"]] == EnumTypeGeometry.Polygon:
                    point_list = geometry["coordinates"][0]
                    label_point_list.append(np.array(point_list, dtype=np.int32))
                else:
                    continue
            else:
                continue
        return label_point_list

    @staticmethod
    def GetMaskFromGeoJson(path_geojson:str, row_num:int, col_num:int)->np.array:

        geojson_data = CFunctionGeoJson.ReadGeoJson(path_geojson)
        label_point_list = CFunctionGeoJson.ParseGeoJson(geojson_data)

        img_mask = np.zeros((row_num, col_num), dtype=np.int32)
        cv2.fillPoly(img_mask, label_point_list, color=255)
        img_mask = img_mask.astype(np.uint8)
        return img_mask


path_geojson = r"C:\ProjectCode\TGClassifier\HippoData\Quv2\Neg_demopath.geojson"

path_img = r"C:\ProjectCode\TGClassifier\demo\demopath.tif"
img = tiff.imread(path_img)

img_mask = CFunctionGeoJson.GetMaskFromGeoJson(path_geojson, img.shape[0], img.shape[1])
tiff.imwrite(os.path.join(r"C:\ProjectCode\TGClassifier\demo", "Neg_mask_flag.tif"), img_mask)
# search_list = [255]
# img_line = CFunctionImageOperator.GetImageLabelLineByPath(img_mask, search_list)
# CFunctionFileIO.WriteImage(r"E:\data\11111", "mask_flag_line.png", img_line)

# img[:, :, 0][img_line > 0] = 255
# img[:, :, 1][img_line > 0] = 0
# img[:, :, 2][img_line > 0] = 0

# CFunctionFileIO.WriteImage(r"E:\data\11111", "mask_flag_img.png", img)
