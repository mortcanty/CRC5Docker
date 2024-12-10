'''
Created on 08.04.2019

Modified on 28.11.2024

@author: mort

ipywidget interface to the GEE for IR-MAD

'''
import ee, time, warnings
import ipywidgets as widgets
import numpy as np

from IPython.display import display
from ipyleaflet import (Map,DrawControl,TileLayer,
                        basemaps,basemap_to_tiles,
                        LayersControl,
                        MeasureControl,
                        LayerGroup,
                        FullScreenControl)
from auxil.eeMad import imad, radcal, chi2cdf
from geopy.geocoders import Nominatim

ee.Initialize()

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


geolocator = Nominatim(timeout=10,user_agent='interface.ipynb')

water_mask = ee.Image('UMD/hansen/global_forest_change_2015').select('datamask').eq(1)

poly = ee.Geometry.MultiPolygon()

def makefeature(data):
    ''' for exporting as CSV to Drive '''
    return ee.Feature(None, {'data': data})

def rgblayer(image, rgb, symmetric = False, clip = True):
    ''' two percent linear stretch '''
    rgbim = image.select(rgb).rename('r','g','b')
    if clip:
        rgbim = rgbim.clip(poly)
    ps = rgbim.reduceRegion(ee.Reducer.percentile([2,98]),maxPixels=1e10).getInfo()
    mx = [ps['r_p98'],ps['g_p98'],ps['b_p98']]
    if symmetric:
        mn = [-x for x in mx]
    else:
        mn = [ps['r_p2'],ps['g_p2'],ps['b_p2']]
    return image.select(rgb).visualize(min = mn, max = mx)

def handle_draw(self, action, geo_json):
    global poly
    if action == 'created':
        coords =  geo_json['geometry']['coordinates']
        poly = poly.union(ee.Geometry.Polygon(coords))
        w_collect.disabled = False
        w_export_assets.disabled = True
        w_export_drive.disabled = True

dc = DrawControl(polyline={},circlemarker={})
dc.rectangle = {"shapeOptions": {"fillColor": "#0000ff","color": "#0000ff","fillOpacity": 0.1}}
dc.polygon = {"shapeOptions": {"fillColor": "#0000ff","color": "#0000ff","fillOpacity": 0.1}}

dc.on_draw(handle_draw)

def GetTileLayerUrl(ee_image_object):
    map_id = ee.Image(ee_image_object).getMapId()
    return map_id["tile_fetcher"].url_format
w_location = widgets.Text(
    layout = widgets.Layout(width='200px'),
    value='Jülich, Germany',
    placeholder=' ',
    description='',
    disabled=False
)
w_platform = widgets.RadioButtons(
    options=['SENTINEL/S2(VNIR/SWIR)','SENTINEL/S2(NIR/SWIR)','LANDSAT LC08'],
    value='SENTINEL/S2(VNIR/SWIR)',
    description='Platform:',
    disabled=False
)
w_startdate1 = widgets.Text(
    value='2019-06-01',
    placeholder=' ',
    description='Start T1:',
    disabled=False
)
w_enddate1 = widgets.Text(
    value='2019-06-30',
    placeholder=' ',
    description='End T1:',
    disabled=False
)
w_startdate2 = widgets.Text(
    value='2020-06-01',
    placeholder=' ',
    description='Start T2:',
    disabled=False
)
w_enddate2 = widgets.Text(
    value='2020-06-30',
    placeholder=' ',
    description='End T2:',
    disabled=False
)
w_maxiter = widgets.IntText(
    layout = widgets.Layout(width='200px'),
    value=20,
    description='Max iter',
    disabled=False
)
w_scale = widgets.FloatText(
    layout = widgets.Layout(width='150px'),
    value=10,
    placeholder=' ',
    description='Scale ',
    disabled=False
)
w_significance = widgets.BoundedFloatText(
    layout = widgets.Layout(width='200px'),
    value=0.0001,
    min=0,
    max=0.05,
    step=0.0001,
    description='Significance:',
    disabled=False
)
w_asset_exportname = widgets.Text(
    value='projects/gee-tf/assets/',
    placeholder=' ',
    disabled=False
)
w_drive_exportname = widgets.Text(
    value='<path>',
    placeholder=' ',
    disabled=False
)
w_cloud_exportname = widgets.Text(
    value='<bucket>:<path>',
    placeholder=' ',
    disabled=False
)
w_out = widgets.Output(
    layout=widgets.Layout(width='700px',border='1px solid black')
)

w_goto = widgets.Button(description='GoTo')
w_collect = widgets.Button(description="Collect",disabled=True)
w_preview = widgets.Button(description="Preview",disabled=True)
w_review = widgets.Button(description="Review",disabled=False)
w_export_assets = widgets.Button(description='ToAssets',disabled=True)
w_export_drive = widgets.Button(description='ToDrive',disabled=True)
w_export_cloud = widgets.Button(description='ToCloud',disabled=True)
w_dates1 = widgets.VBox([w_startdate1,w_enddate1,w_maxiter])
w_scalesig = widgets.HBox([w_scale,w_significance])
w_dates2 = widgets.VBox([w_startdate2,w_enddate2,w_scalesig])
w_dates = widgets.HBox([w_platform,w_dates1,w_dates2])
w_exp = widgets.HBox([w_export_assets,w_asset_exportname,
                      widgets.VBox([ widgets.HBox([w_export_drive,w_drive_exportname]),
                                     widgets.HBox([w_export_cloud,w_cloud_exportname])])])
w_coll = widgets.HBox([w_collect,widgets.VBox([w_preview,w_review]),w_exp])
w_reset = widgets.Button(description='Reset',disabled=False)
w_bot = widgets.HBox([w_out,w_reset,w_goto,w_location])
box = widgets.VBox([w_dates,w_coll,w_bot])

def on_widget_change(b):
    w_export_assets.disabled = True
    w_export_drive.disabled = True

def on_platform_widget_change(b):
    w_export_assets.disabled = True
    w_export_drive.disabled = True
    if b['new']=='SENTINEL/S2(VNIR/SWIR)':
        w_scale.value=10
    elif b['new']=='SENTINEL/S2(NIR/SWIR)':
        w_scale.value=20
    else:
        w_scale.value=30

w_platform.observe(on_platform_widget_change,names='value')
w_startdate1.observe(on_widget_change,names='value')
w_enddate1.observe(on_widget_change,names='value')
w_startdate2.observe(on_widget_change,names='value')
w_enddate2.observe(on_widget_change,names='value')

def on_goto_button_clicked(b):
    with w_out:
        try:
            location = geolocator.geocode(w_location.value)
            m.center = (location.latitude,location.longitude)
            m.zoom = 11
        except Exception as e:
            print('Error: %s'%e)

w_goto.on_click(on_goto_button_clicked)

def clear_layers():
    for i in range(20,2,-1):
        if len(m.layers)>i:
            m.remove(m.layers[i])

def on_reset_button_clicked(b):
    global poly
    with w_out:
        try:
            w_preview.disabled = True
            w_export_assets.disabled = True
            w_export_drive.disabled = True
            w_export_cloud.disabled = True
            poly = ee.Geometry.MultiPolygon()
            clear_layers()
            w_out.clear_output()
            print('Algorithm output')
        except Exception as e:
            print('Error: %s'%e)

w_reset.on_click(on_reset_button_clicked)

def on_collect_button_clicked(b):
    global collection,count,nbands,bands, \
           w_startdate1,w_enddate1,w_startdate2, \
           w_platfform,w_enddate2,w_changemap, \
           image1,image2, result, \
           madnames,poly,timestamp1,timestamp2
    with w_out:
        w_out.clear_output()
        try:
            clear_layers()
            print('Collecting ...')
            cloudcover = 'CLOUD_COVER'
            rgb = ['B4','B5','B7']
            if w_platform.value=='SENTINEL/S2(VNIR/SWIR)':
                collectionid = 'COPERNICUS/S2_SR'
                bands = ['B2','B3','B4','B8']
                rgb = ['B4','B3','B2']
                cloudcover = 'CLOUDY_PIXEL_PERCENTAGE'
            elif w_platform.value=='SENTINEL/S2(NIR/SWIR)':
                collectionid = 'COPERNICUS/S2_SR'
                bands = ['B5','B6','B7','B8A','B11','B12']
                rgb = ['B5','B7','B11']
                cloudcover = 'CLOUDY_PIXEL_PERCENTAGE'
            elif w_platform.value=='LANDSAT LC08':
                collectionid = 'LANDSAT/LC08/C02/T1_L2'
                bands = ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7']
                rgb = ['SR_B4','SR_B3','SR_B2']

            collection1 = ee.ImageCollection(collectionid) \
                      .filterBounds(poly) \
                      .filterDate(ee.Date(w_startdate1.value), ee.Date(w_enddate1.value)) \
                      .filter(ee.Filter.contains(rightValue=poly,leftField='.geo')) \
                      .sort(cloudcover, True)
            count = collection1.size().getInfo()
            if count==0:
                raise ValueError('No images found for first time interval '+collectionid)
            collection2 = ee.ImageCollection(collectionid) \
                      .filterBounds(poly) \
                      .filterDate(ee.Date(w_startdate2.value), ee.Date(w_enddate2.value)) \
                      .filter(ee.Filter.contains(rightValue=poly,leftField='.geo')) \
                      .sort(cloudcover, True)
            count = collection2.size().getInfo()
            if count==0:
                raise ValueError('No images found for second time interval')
            image1 = ee.Image(collection1.first()).select(bands)
            timestamp1 = ee.Date(image1.get('system:time_start')).getInfo()
            timestamp1 = time.gmtime(int(timestamp1['value'])/1000)
            timestamp1 = time.strftime('%c', timestamp1)
            systemid1 = image1.get('system:id').getInfo()
            cloudcover1 = image1.get(cloudcover).getInfo()
            image2 = ee.Image(collection2.first()).select(bands)
            timestamp2 = ee.Date(image2.get('system:time_start')).getInfo()
            timestamp2 = time.gmtime(int(timestamp2['value'])/1000)
            timestamp2 = time.strftime('%c', timestamp2)
            systemid2 = image2.get('system:id').getInfo()
            cloudcover2 = image2.get(cloudcover).getInfo()
            print('Img1: %s'%systemid1)
            print('Date: %s, Cloud cover: %f'%(timestamp1,cloudcover1))
            print('Img2: %s'%systemid2)
            print('Date: %s, Cloud cover: %f'%(timestamp2,cloudcover2))
            nbands = image1.bandNames().length()
            madnames = ['MAD'+str(i+1) for i in range(nbands.getInfo())]

            #The iMAD algorithm
            inputlist = ee.List.sequence(1,w_maxiter.value)
            first = ee.Dictionary({'done':ee.Number(0),
                                   'scale':ee.Number(w_scale.value),
                                   'image':image1.addBands(image2).clip(poly),#.updateMask(water_mask),
                                   'allrhos': [ee.List.sequence(1,nbands)],
                                   'chi2':ee.Image.constant(0),
                                   'MAD':ee.Image.constant(0)})
            result = ee.Dictionary(inputlist.iterate(imad,first))
            #Display preview
            m.add_layer(TileLayer(url=GetTileLayerUrl(rgblayer(image1,rgb)),name=timestamp1))
            m.add_layer(TileLayer(url=GetTileLayerUrl(rgblayer(image2,rgb)),name=timestamp2))
            w_preview.disabled = False
            w_export_assets.disabled = False
            w_export_drive.disabled = False
            w_export_cloud.disabled = False
        except Exception as e:
            print('Error: %s'%e)

w_collect.on_click(on_collect_button_clicked)

def on_preview_button_clicked(b):
    global MADs, allrhos, ninvar, coeffs
    with w_out:
        try:
            #iMAD
            print('iMAD ...')
            MAD = ee.Image(result.get('MAD')).rename(madnames)
            #threshold iMAD image
            chi2 = ee.Image(result.get('chi2')).rename(['chi2'])
            pval = chi2cdf(chi2,nbands).subtract(1).multiply(-1)
            tst = pval.gt(ee.Image.constant(w_significance.value))
            MAD = MAD.where(tst,ee.Image.constant(0))
            nc_mask = pval.gt(w_significance.value)
            #radcal
            inputlist1 = ee.List.sequence(0,nbands.subtract(1))
            first = ee.Dictionary({'image':image1.addBands(image2),
                                   'scale':ee.Number(w_scale.value),
                                   'ncmask':nc_mask,
                                   'nbands':nbands,
                                   'rect':poly,
                                   'coeffs': ee.List([]),
                                   'normalized':ee.Image.constant(0)})
            result1 = ee.Dictionary(inputlist1.iterate(radcal,first))
            sel = ee.List.sequence(1,nbands)
            normalized = ee.Image(result1.get ('normalized')).select(sel).rename(bands)
            #for export
            coeffs = np.array(ee.List(result1.get('coeffs')).getInfo()).round(4)
            ninvar = ee.String(nc_mask.reduceRegion(ee.Reducer.sum().unweighted(),
                                 scale=w_scale.value,maxPixels= 1e10).toArray().project([0]))
            MADs = ee.Image.cat(MAD,chi2,nc_mask,image1,image2,normalized).clip(poly)
            allrhos = ee.Array(result.get('allrhos')).toList()
            #output to text window
            rhos = np.array(ee.List(result.get('allrhos')).get(-1).getInfo()).round(4)
            print('Rhos: %s'%str(rhos))
            print('Radiometric normalization [slope, intercept, R]:')
            for i in range(nbands.getInfo()):
                print(str(coeffs[i]))
            #display
            #m.add(TileLayer(url=GetTileLayerUrl(chi2.visualize(min=1000,max=10000)),name='chi square'))
            m.add(TileLayer(url=GetTileLayerUrl(rgblayer(MAD,[0,1,2],True)),name='MAD123'))
        except Exception as e:
            print('Error: %s'%e)

w_preview.on_click(on_preview_button_clicked)

def on_review_button_clicked(b):
    with w_out:
        w_out.clear_output()
        try:
            print(w_asset_exportname.value)
            MADs = ee.Image(w_asset_exportname.value)
            m.add_layer(TileLayer(url=GetTileLayerUrl(rgblayer(MADs,[0,1,2],True,False)),name='MAD123'))
            metadata = ee.FeatureCollection(w_asset_exportname.value+'_meta')
            T1 = metadata.aggregate_array('T1').getInfo()
            T2 = metadata.aggregate_array('T2').getInfo()
            rhos = metadata.aggregate_array('rhos').getInfo()
            coeffs = metadata.aggregate_array('coeffs').getInfo()
            print('Time interval: ',T1,T2)
            print('Rhos: ',rhos)
            print('Coeffs: ',coeffs)
        except Exception as e:
            print('Error: %s'%e)

w_review.on_click(on_review_button_clicked)

def on_export_assets_button_clicked(b):
    global nbands, bands, MADs, allrhos, ninvar, coeffs
    with w_out:
        try:
            MAD = ee.Image(result.get('MAD')).rename(madnames)
    #      threshold iMAD image
            chi2 = ee.Image(result.get('chi2')).rename(['chi2'])
            pval = chi2cdf(chi2,nbands).subtract(1).multiply(-1)
            tst = pval.gt(ee.Image.constant(w_significance.value))
            MAD = MAD.where(tst,ee.Image.constant(0))
            nc_mask = pval.gt(w_significance.value)
            sel = ee.List.sequence(1,nbands)
    #      radcal
            inputlist1 = ee.List.sequence(0,nbands.subtract(1))
            first = ee.Dictionary({'image':image1.addBands(image2),
                                   'scale':ee.Number(w_scale.value),
                                   'ncmask':nc_mask,
                                   'nbands':nbands,
                                   'rect':poly,
                                   'coeffs': ee.List([]),
                                   'normalized':ee.Image.constant(0)})
            result1 = ee.Dictionary(inputlist1.iterate(radcal,first))
            normalized = ee.Image(result1.get ('normalized')).select(sel).rename(bands)
    #      export
            coeffs = ee.List(result1.get('coeffs'))
            MADs = ee.Image.cat(MAD,chi2,nc_mask,image1,image2,normalized).clip(poly)
            allrhos = ee.Array(result.get('allrhos')).toList()

            assexport_image = ee.batch.Export.image.toAsset(MADs,
                                        description = 'assetExportTask',
                                        assetId = w_asset_exportname.value,scale = w_scale.value,maxPixels = 1e9)
            assexport_image.start()
            print('Exporting change maps to %s\n task id: %s'%(w_asset_exportname.value,str(assexport_image.id)))

            rhos = ee.String.encodeJSON(allrhos.get(-1))
            coeffs = ee.String.encodeJSON(coeffs)
            data = ee.Dictionary({'T1':timestamp1,'T2':timestamp2,'rhos':rhos,'coeffs':coeffs})
            metadata = ee.FeatureCollection(ee.Feature(poly,data))

            assexport_meta = ee.batch.Export.table.toAsset(metadata,
                                        description = 'assetExportTask',
                                        assetId = w_asset_exportname.value+'_meta')
            assexport_meta.start()
            print('Exporting metadata to %s\n task id: %s'%(w_asset_exportname.value+'_meta',str(assexport_meta.id)))

        except Exception as e:
            print('Error: %s'%e)

w_export_assets.on_click(on_export_assets_button_clicked)

def on_export_drive_button_clicked(b):
    global nbands, group1, group2, bands, MADs, allrhos, ninvar, coeffs
    with w_out:
        try:
            MAD = ee.Image(result.get('MAD')).rename(madnames)
    #      threshold iMAD image
            chi2 = ee.Image(result.get('chi2')).rename(['chi2'])
            pval = chi2cdf(chi2,nbands).subtract(1).multiply(-1)
            tst = pval.gt(ee.Image.constant(w_significance.value))
            MAD = MAD.where(tst,ee.Image.constant(0))
            nc_mask = pval.gt(w_significance.value)
            sel = ee.List.sequence(1,nbands)
    #      radcal
            inputlist1 = ee.List.sequence(0,nbands.subtract(1))
            first = ee.Dictionary({'image':image1.addBands(image2),
                                   'scale':ee.Number(w_scale.value),
                                   'ncmask':nc_mask,
                                   'nbands':nbands,
                                   'rect':poly,
                                   'coeffs': ee.List([]),
                                   'normalized':ee.Image.constant(0)})
            result1 = ee.Dictionary(inputlist1.iterate(radcal,first))
            normalized = ee.Image(result1.get ('normalized')).select(sel).rename(bands)
    #      export
            coeffs = ee.List(result1.get('coeffs'))
            ninvar = ee.String(nc_mask.reduceRegion(ee.Reducer.sum().unweighted(),
                                 scale=w_scale.value,maxPixels= 1e10).toArray().project([0]))
            MADs = ee.Image.cat(MAD,chi2,nc_mask,image1,image2,normalized).float().clip(poly)
            allrhos = ee.Array(result.get('allrhos')).toList()

            gdexport = ee.batch.Export.image.toDrive(MADs,
                                        description='driveExportTask',
                                        fileNamePrefix=w_drive_exportname.value,scale=w_scale.value,maxPixels=1e9)
            gdexport.start()
            print('Exporting change map to %s\n task id: %s'%(w_drive_exportname.value,str(gdexport.id)))

        except Exception as e:
            print('Error: %s'%e)

w_export_drive.on_click(on_export_drive_button_clicked)

def on_export_cloud_button_clicked(b):
    global nbands, group1, group2, bands, MADs, allrhos, ninvar, coeffs
    with w_out:
        try:
            MAD = ee.Image(result.get('MAD')).rename(madnames)
    #      threshold iMAD image
            chi2 = ee.Image(result.get('chi2')).rename(['chi2'])
            pval = chi2cdf(chi2,nbands).subtract(1).multiply(-1)
            tst = pval.gt(ee.Image.constant(w_significance.value))
            MAD = MAD.where(tst,ee.Image.constant(0))
            nc_mask = pval.gt(w_significance.value)
            sel = ee.List.sequence(1,nbands)
    #      radcal
            inputlist1 = ee.List.sequence(0,nbands.subtract(1))
            first = ee.Dictionary({'image':image1.addBands(image2),
                                   'scale':ee.Number(w_scale.value),
                                   'ncmask':nc_mask,
                                   'nbands':nbands,
                                   'rect':poly,
                                   'coeffs': ee.List([]),
                                   'normalized':ee.Image.constant(0)})
            result1 = ee.Dictionary(inputlist1.iterate(radcal,first))
            normalized = ee.Image(result1.get ('normalized')).select(sel).rename(bands)
    #      export
            coeffs = ee.List(result1.get('coeffs'))
            ninvar = ee.String(nc_mask.reduceRegion(ee.Reducer.sum().unweighted(),
                                 scale=w_scale.value,maxPixels= 1e10).toArray().project([0]))
            MADs = ee.Image.cat(MAD,chi2,nc_mask,image1,image2,normalized).float().clip(poly)
            allrhos = ee.Array(result.get('allrhos')).toList()

            bucket, fileNamePrefix = w_cloud_exportname.value.split(':')

            gdexport = ee.batch.Export.image.toCloudStorage(MADs,
                                        description='driveExportTask',
                                        bucket=bucket,
                                        fileNamePrefix=fileNamePrefix,scale=w_scale.value,maxPixels=1e9)
            gdexport.start()
            print('Exporting change map to %s\n task id: %s'%(w_cloud_exportname.value,str(gdexport.id)))

        except Exception as e:
            print('Error: %s'%e )

w_export_cloud.on_click(on_export_cloud_button_clicked)


def run():
    global m,dc,lc,center,osm,ewi
    center = [51.0,6.4]
    osm = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
    ews = basemap_to_tiles(basemaps.Esri.WorldStreetMap)
    ewi = basemap_to_tiles(basemaps.Esri.WorldImagery)
    lc = LayersControl(position='topright')
    fs = FullScreenControl(position='topleft')
    mc = MeasureControl(position='topright',primary_length_unit='kilometers')
    m = Map(center=center, zoom=11, layout={'height':'500px'},layers=(ewi,ews,osm),controls=(mc,dc,lc,fs))
    with w_out:
        w_out.clear_output()
        print('Algorithm output')
    display(m)
    return box
