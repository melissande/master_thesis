from osgeo import gdal, osr,ogr
import numpy as np
import json

def vectorize_raster(geoJsonFileName,array2d,layerName="BuildingID",fieldName="BuildingID"):
    

    memdrv = gdal.GetDriverByName('MEM')
    src_ds = memdrv.Create('', array2d.shape[1], array2d.shape[0], 1)
    band = src_ds.GetRasterBand(1)
    band.WriteArray(array2d)

    dst_layername = "BuildingID"
    drv = ogr.GetDriverByName("geojson")
    dst_ds = drv.CreateDataSource(geoJsonFileName)
    dst_layer = dst_ds.CreateLayer(layerName, srs=None)

    fd = ogr.FieldDefn(fieldName, ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = 1

    gdal.Polygonize(band, None, dst_layer, dst_field, [], callback=None)

    return
def predict_score_batch(temporary_fold,batch_y,prediction):
    tot_score_batch=0
    tot_f1_score_batch=0

    for i in range(len(batch_y)):
        vectorize_raster(temporary_fold+'test_gt.geojson',batch_y[i])
        vectorize_raster(temporary_fold+'test_pred.geojson',prediction[i])
        with open(temporary_fold+'test_gt.geojson') as f:
            geojson_groundtruth = json.load(f)
        with open(temporary_fold+'test_pred.geojson') as f:
            geojson_prediction = json.load(f)
        score=0
        
        M=len(geojson_prediction['features'])
        N=len(geojson_groundtruth['features'])
        for feature_pred in geojson_prediction['features']:   
            IoUs=[]
            for feature_gt in geojson_groundtruth['features']:
                
                poly1=ogr.CreateGeometryFromJson(json.dumps(feature_gt['geometry']))
                poly2=ogr.CreateGeometryFromJson(json.dumps(feature_pred['geometry']))
                intersection = poly1.Intersection(poly2)
                union = poly1.Union(poly2)
                if intersection is None:
                    IoUs.append(0.0)
                else:
                    IoUs.append(intersection.GetArea()/union.GetArea())
                
            IoUs=np.asarray(IoUs)
            IoUs=(IoUs>0.5).astype(int)*IoUs
#             print(IoUs)
            if (IoUs.size and np.amax(IoUs)>0):
                index=np.argmax(IoUs)
#                 print(index)
                geojson_groundtruth['features'].remove(geojson_groundtruth['features'][0])
                score+=1
            tot_score_batch+=score/M
            tot_f1_score_batch+=2*score/(M+N)
        tot_score_batch/=len(batch_y)
        tot_f1_score_batch/=len(batch_y)
    return tot_score_batch*100,tot_f1_score_batch*100
#     print(tot_score_batch)
        