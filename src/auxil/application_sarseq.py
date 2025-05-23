# application_sarseq.py
# widget interface for SAR sequential change detection, full scale version
# uses ipyLeaflet, not geeMap
# Mort Canty
# Dec, 2024

import ee
ee.Initialize()

from auxil.eesarseq import assemble_and_run
import time
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from ipyleaflet import (Map,DrawControl,TileLayer,
                        FullScreenControl,
                        basemaps,basemap_to_tiles,
                        LayersControl)
from geopy.geocoders import Nominatim

'''
 ********************
 The widget interface
 ********************
'''

aoi = None

geolocator = Nominatim(timeout=10, user_agent='mort.canty@gmail.com')

dyn = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
                    .filterDate('2021-09-01', '2022-12-31') \
                    .select('label').mosaic()

def maskNoBuildings(image):
    return image.where(dyn.lte(5), 0)

# watermask = ee.Image('UMD/hansen/global_forest_change_2017_v1_5').select('datamask').eq(1)
watermask = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('max_extent').eq(0)

ground_truth = ee.FeatureCollection('projects/sentinel-change-detection/assets/ground_truth/houston_candid_11-2020_4-2021')
groundTruth = ground_truth.reduceToImage(["diff"], ee.Reducer.first())

w_location = widgets.Text(
    layout = widgets.Layout(width='150px'),
    value='Odessa',
    placeholder=' ',
    description='',
    disabled=False
)
w_orbitpass = widgets.RadioButtons(
    layout = widgets.Layout(width='200px'),
    options=['ASCENDING','DESCENDING'],
    value='ASCENDING',
    description='Pass:',
    disabled=False
)
w_changemap = widgets.RadioButtons(
    options=['Bitemp', 'First', 'Last', 'Frequency', 'Plot', 'ATSF'],
    value='First',
    layout = widgets.Layout(width='200px'),
    disabled=False
)
w_visual = widgets.RadioButtons(
    options=['S2', 'NAIP'],
    value='S2',
    layout = widgets.Layout(width='200px'),
    disabled=False
)
w_interval = widgets.BoundedIntText(
    min=1,
    value=1,
    layout = widgets.Layout(width='200px'),
    description='BitempInt:',
    disabled=True
)
w_maxfreq = widgets.BoundedIntText(
    min=1,
    value=20,
    layout = widgets.Layout(width='200px'),
    description='MaxFreq:',
    disabled=True
)
w_minfreq = widgets.BoundedIntText(
    min=1,
    value=1,
    layout = widgets.Layout(width='200px'),
    description='MinFreq:',
    disabled=True
)
w_platform = widgets.RadioButtons(
    layout = widgets.Layout(width='200px'),
    options=['Both','A','B'],
     value='A',
    description='Platform:',
    disabled=False
)
w_plot_type = widgets.RadioButtons(
    layout = widgets.Layout(width='200px'),
    options=['All','Direction'],
    value='All',
    description='PlotType:',
    disabled=False
)
w_relativeorbitnumber = widgets.IntText(
    value='0',
    layout = widgets.Layout(width='150px'),
    description='RelOrbit:',
    disabled=False
)
w_exportassetsname = widgets.Text(
    layout = widgets.Layout(width='200px'),
    value='projects/<your cloud project>/assets/',
    placeholder=' ',
    disabled=False
)
w_exportdrivename = widgets.Text(
    layout = widgets.Layout(width='200px'),
    value='<path>',
    placeholder=' ',
    disabled=False
)
w_exportscale = widgets.FloatText(
    layout = widgets.Layout(width='150px'),
    value=10,
    placeholder=' ',
    description='Export Scale',
    disabled=False
)
w_startdate = widgets.Text(
    layout = widgets.Layout(width='200px'),
    value='2024-08-01',
    placeholder=' ',
    description='StartDate:',
    disabled=False
)
w_enddate = widgets.Text(
    layout = widgets.Layout(width='200px'),
    value='2024-12-31',
    placeholder=' ',
    description='EndDate:',
    disabled=False
)
w_median = widgets.Checkbox(
    layout = widgets.Layout(width='200px'),
    value=False,
    description='MedianFilter',
    disabled=False
)
w_quick = widgets.Checkbox(
    value=False,
    description='QuickPreview',
    disabled=False
)
w_significance = widgets.BoundedFloatText(
    layout = widgets.Layout(width='200px'),
    value='0.01',
    min=0.0001,
    max=0.05,
    step=0.001,
    description='Signif:',
    disabled=False
)
w_maskchange = widgets.Checkbox(
    value=True,
    description='NCMask',
    disabled=False
)
w_maskwater = widgets.Checkbox(
    value=True,
    description='WaterMask',
    disabled=False
)
w_stride = widgets.BoundedIntText(
    value=1,
    min=1,
    description='Stride:',
    layout = widgets.Layout(width='200px'),
    disabled=False
)
w_dw = widgets.Checkbox(
    value=False,
    description='NoBuildings Mask',
    disabled=False
)
w_out = widgets.Output(
    layout=widgets.Layout(width='700px',border='1px solid black')
)

w_collect = widgets.Button(description="Collect",disabled=True)
w_preview = widgets.Button(description="Preview",disabled=True)
w_reset = widgets.Button(description='Reset',disabled=False)
w_review = widgets.Button(description="ReviewAsset",disabled=False)
w_view = widgets.Button(description="View",disabled=False)
w_classify = widgets.Button(description="Classify",disabled=True)
w_goto = widgets.Button(description='GoTo',disabled=False)
w_export_ass = widgets.Button(description='ExportToAssets',disabled=True)
w_export_drv = widgets.Button(description='ExportToDrive',disabled=True)

w_masks = widgets.VBox([w_maskchange,w_maskwater,w_dw,w_quick])
w_dates = widgets.VBox([w_startdate,w_enddate])
w_bmap = widgets.VBox([w_interval,w_maxfreq,w_minfreq])
w_export = widgets.VBox([widgets.HBox([w_export_ass,w_exportassetsname]),
                         widgets.HBox([w_export_drv,w_exportdrivename])])
w_signif = widgets.VBox([w_significance,w_median])

#Assemble the interface

row1 = widgets.HBox([w_platform,w_orbitpass,w_relativeorbitnumber,w_dates])
row2 = widgets.HBox([w_collect,w_signif,w_stride,w_export])
row3 = widgets.HBox([widgets.VBox([w_preview,w_review,w_reset]),w_changemap,widgets.VBox([w_view,w_exportscale,w_plot_type]),w_visual,w_bmap,w_masks])
row4 = widgets.HBox([w_out,w_goto,w_location])

box = widgets.VBox([row1,row2,row3,row4])

#Event handers

def on_widget_change(b):
    w_preview.disabled = True
    w_export_ass.disabled = True
    w_export_drv.disabled = True
    
def on_changemap_widget_change(b):   
    if b['new'] == 'Bitemp':
        w_interval.disabled=False
    else:
        w_interval.disabled=True
    if b['new']=='Frequency':
        w_maxfreq.disabled=False
        w_minfreq.disabled=False
    else:
        w_maxfreq.disabled=True 
        w_minfreq.disabled=True   

#These widget changes require a new collect
w_orbitpass.observe(on_widget_change,names='value')
w_platform.observe(on_widget_change,names='value')
w_relativeorbitnumber.observe(on_widget_change,names='value')
w_startdate.observe(on_widget_change,names='value')
w_enddate.observe(on_widget_change,names='value')
w_median.observe(on_widget_change,names='value')
w_significance.observe(on_widget_change,names='value')
w_changemap.observe(on_changemap_widget_change,names='value')  

        
def clear_layers():
    for i in range(20,2,-1): 
        if len(m.layers)>i:
            m.remove_layer(m.layers[i])    

def on_reset_button_clicked(b):
    try:
        clear_layers()
        w_out.clear_output()
    except Exception as e:
        with w_out:
            print('Error: %s'%e)

w_reset.on_click(on_reset_button_clicked)         

def on_goto_button_clicked(b):
    try:
        location = geolocator.geocode(w_location.value)
        m.center = (location.latitude, location.longitude)
        m.zoom = 11
        with w_out:
            w_out.clear_output()
            print(location)
    except Exception as e:
        with w_out:
            print('Error: %s'%e)

w_goto.on_click(on_goto_button_clicked)

def GetTileLayerUrl(image):
    map_id = ee.Image(image).getMapId()
    return map_id["tile_fetcher"].url_format        

def handle_draw(target, action, geo_json):
    global aoi
    coords =  geo_json['geometry']['coordinates']
    if action == 'created':
        aoi = ee.Geometry.Polygon(coords)
        w_preview.disabled = True
        w_export_ass.disabled = True
        w_export_drv.disabled = True 
        w_collect.disabled = False
    elif action == 'deleted':
        aoi = None
        w_collect.disabled = True  
        #w_preview.disabled = True
        w_export_ass.disabled = True
        w_export_drv.disabled = True

def rgbLayer(image):
    ''' one percent linear stretch '''
    rgbim = image.rename('r','g','b')
    ps = rgbim.reduceRegion(ee.Reducer.percentile([1,99]),scale=10,maxPixels=1e10).getInfo()
    mx = [ps['r_p99'],ps['g_p99'],ps['b_p99']]
    mn = [ps['r_p1'],ps['g_p1'],ps['b_p1']]
    return rgbim.visualize(min=mn,max=mx)

def plot_bmap(image):
    '''plot change fractions from bmap bands'''
    def plot_iter(current,prev):
        current = ee.Image.constant(current)
        plots = ee.List(prev)
        res = bmap1.multiply(0) \
                  .where(bmap1.eq(current), 1) \
                  .reduceRegion(ee.Reducer.mean(), scale=10, maxPixels=10e10)
        return ee.List(plots.add(res))
    with w_out:
        try:
            w_out.clear_output()
            print('Change fraction plots ...')
            k = image.bandNames().length().subtract(3).getInfo()
            bmap1 = image.select(ee.List.sequence(3, k+2)).clip(aoi)
            if w_maskwater.value:
                bmap1 = bmap1.updateMask(watermask)
            plots = ee.List(ee.List([1, 2, 3]).iterate(plot_iter, ee.List([]))).getInfo()

            bns = np.array(list([s[3:9] for s in list(plots[0].keys())]))
            x = range(2, k+2)
            _ = plt.figure(figsize=(15, 8))
            posdef = np.array(list(plots[0].values()))
            negdef = np.array(list(plots[1].values()))
            indef = np.array(list(plots[2].values()))
            alldef = posdef + negdef + indef
            # leave out first three changes in plot (anlauf Zeit?)
            plt.ylim(bottom=0.0,top=max(alldef)*1.1)
            if w_plot_type.value == 'Direction':
                plt.plot(x[3:-1], posdef[3:-1], 'ro-', label='posdef')
                plt.plot(x[3:-1], negdef[3:-1], 'co-', label='negdef')
                plt.plot(x[3:-1], indef[3:-1], 'yo-', label='indef')
            else:
                plt.plot(x[3:-1], alldef[3:-1], 'bo-', label='all')
            ticks = range(0, k+2)
            labels = [str(i) for i in range(0, k+2)]
            labels[0] = ' '
            labels[-1] = ' '
            labels[1:-1] = bns
            if k>80:
                for i in range(1, k+1, 2):
                    labels[i] = ''
            plt.xticks(ticks, labels, rotation=90)
            plt.legend()
#            fn = w_exportassetsname.value.replace('/','-')+'.png'
#            plt.savefig(fn,bbox_inches='tight')
            w_out.clear_output()
            plt.show()
#            print('Saved to ~/%s'%fn)
        except Exception as e:
            print('Error: %s'%e)
    
def on_collect_button_clicked(b):
    ''' Collect a time series from the archive '''
    global cmaps, bmaps, atsf, count, crs
    with w_out:
        try:
            w_out.clear_output()
            clear_layers()
            print('Running on GEE archive COPERNICUS/S1_GRD')
            # assemble time series and run the algorithm
            # count is the number of images in the time series!
            cmaps, bmaps, count, rons, collection, atsf, _, _ = assemble_and_run(aoi, median=w_median.value,
                                                      significance=w_significance.value, startdate=w_startdate.value,
                                                      enddate=w_enddate.value, platform=w_platform.value, stride=w_stride.value,
                                                      orbitpass=w_orbitpass.value, ron=w_relativeorbitnumber.value)
            crs = ee.Image(collection.first()).select(0).projection().crs().getInfo()
            w_preview.disabled = False
            w_export_ass.disabled = False
            w_export_drv.disabled = False
            #Display S1 mosaic
            if len(rons)>0:
                print('Shortest orbit path series length: %i images\n please wait for raster overlay ...'%count)
                clear_layers()
                S1 = collection.mean()
            m.add(TileLayer(url=GetTileLayerUrl(S1.select(0).visualize(min=-15, max=4)),name='S1'))
        except Exception as e:
            print('Error: %s' % e)
            
w_collect.on_click(on_collect_button_clicked)                  

def on_preview_button_clicked(b):
    ''' Preview change maps '''
    with w_out:  
        try:       
            jet = 'black,blue,cyan,yellow,red'
            rcy = 'black,red,cyan,yellow'
            mn = 0
            palette = jet
            w_out.clear_output()
            print('Shortest orbit path series length: %i images\n previewing please wait for raster overlay ...'%count)

            changemap = w_changemap.value

            if changemap=='First':
                mp = ee.Image(cmaps.select('smap')).byte()
                mx = count
                print('Interval of first change:\n blue = early, red = late')
            elif changemap=='Last':
                mp=ee.Image(cmaps.select('cmap')).byte()
                mx = count
                print('Interval of last change:\n blue = early, red = late')
            elif changemap=='Frequency':
                mp = ee.Image(cmaps.select('fmap')).byte()
                mx = w_maxfreq.value
                print('Change frequency :\n blue = few, red = many')
            elif changemap == 'Bitemp':
                sel = int(w_interval.value)
                sel = min(sel,count-1)
                sel = max(sel,1)
                changemap = 'Bitemp%i'%sel
                mp = ee.Image(bmaps.select(sel)).byte()
                print('red = positive definite, cyan = negative definite, yellow = indefinite')                 
                palette = rcy
                mx = 3
            elif changemap=='ATSF':
                atsf_db = atsf.log10().multiply(10)
                mp = ee.Image.rgb( atsf_db.select(1), atsf_db.select(0), atsf.select(1).divide(atsf.select(0)))
                mn = [-15, -15, 0]
                mx = [ -2,   4, 1]
                palette = None
                print( 'ATSF' )
            elif changemap=='Plot':
                w_out.clear_output()
                raise RuntimeError('Available only for ReviewAsset')
            elif changemap=='S2':
                w_out.clear_output()
                image_s2 = collect_s2()
                mp = ee.Image(image_s2)
                mn = [500, 500, 500]
                mx = [4000, 4000, 4000]
                palette = None
            elif changemap=='NAIP':
                w_out.clear_output()
                image_naip = collect_naip()
                mp = ee.Image(image_naip)
                mn = [0, 0, 0]
                mx = [255, 255, 255]
                palette = None
            if not w_quick.value:
                mp = mp.reproject(crs=crs, scale=float(w_exportscale.value))
            if w_dw.value:
                mp = maskNoBuildings(mp)
            if w_maskwater.value==True:
                mp = mp.updateMask(watermask) 
            if w_maskchange.value==True:   
                if w_changemap.value=='Frequency':
                    mp = mp.updateMask(mp.gte(w_minfreq.value))
                elif w_changemap.value=='ATSF':
                    pass    
                else:
                    mp = mp.updateMask(mp.gt(0))    
            m.add(TileLayer(url=GetTileLayerUrl(mp.visualize(min=mn, max=mx,
                                  palette=palette)), name=changemap))
        except Exception as e:
            print('Error: %s'%e)

w_preview.on_click(on_preview_button_clicked)

def on_review_button_clicked(b):
    ''' Examine change maps exported to user's assets '''

    changemap = w_changemap.value

    with w_out:  
        try: 
#          test for existence of asset                  
            _ = ee.Image(w_exportassetsname.value).getInfo()
#          ---------------------------            
            asset = ee.Image(w_exportassetsname.value)
            aoi = ee.Geometry.Polygon(ee.Geometry(asset.get('system:footprint')).coordinates())
            crs = asset.projection().crs().getInfo()
            center = aoi.centroid().coordinates().getInfo()
            center.reverse()
            m.center = center
            bitemp_names = asset.bandNames().getInfo()[3:]
            # the intervals are named by the date of the second image
            # bitemp_count is the number of bitemporal change images
            bitemp_count = len(bitemp_names)
            jet = 'black,blue,cyan,yellow,red'
            rcy = 'black,red,cyan,yellow'
            smap = asset.select('smap').byte()
            cmap = asset.select('cmap').byte()
            fmap = asset.select('fmap').byte()
            bmps = asset.select(bitemp_names).byte()
            palette = jet
            w_out.clear_output()
            print('Bitemporal series length: %i images, reviewing (please wait for raster overlay) ...'%(bitemp_count))
            if changemap=='First':
                mp = smap
                mn = 0          
                mx = bitemp_count
                print('Interval of first change:\n blue = early, red = late')
            elif changemap=='Last':
                mp = cmap
                mn = 0
                mx = bitemp_count
                print('Interval of last change:\n blue = early, red = late')
            elif changemap=='Frequency':
                mp = fmap
                mn = 0
                mx = w_maxfreq.value
                print('Change frequency :\n blue = few, red = many')
            elif changemap=='Bitemp':
                sel = int(w_interval.value)
                sel = min(sel,bitemp_count-1)
                sel = max(sel,1)
                changemap = 'Bitemp%i' % sel
                print('Interval ending %s'%bitemp_names[sel])
                print('red = positive definite, cyan = negative definite, yellow = indefinite')  
                mp = ee.Image(bmps.select(sel))
                palette = rcy
                mn = 0
                mx = 3
            elif changemap == 'ATSF':
                w_out.clear_output()
                raise RuntimeError('Available only for Preview')
            elif changemap == 'S2':
                w_out.clear_output()
                raise RuntimeError('Available only for Preview')
            elif changemap == 'NAIP':
                w_out.clear_output()
                raise RuntimeError('Available only for Preview')
            elif changemap == 'Plot':
                plot_bmap(asset)
                return None
            if w_maskwater.value==True:
                mp = mp.updateMask(watermask) 
            if w_dw.value:
                mp = maskNoBuildings(mp)
            if w_maskchange.value==True:   
                if w_changemap.value=='Frequency':
                    mp = mp.updateMask(mp.gte(w_minfreq.value)) 
                else:
                    mp = mp.updateMask(mp.gt(0))
            if not w_quick.value:
                mp = mp.reproject(crs=crs, scale=float(w_exportscale.value))
            m.add(TileLayer(url=GetTileLayerUrl(mp.visualize(min=mn, max=mx,
                                         palette=palette)),name=changemap))
        except Exception as e:
            print('Error: %s'%e)
    
w_review.on_click(on_review_button_clicked)   

def on_export_ass_button_clicked(b):
    ''' Export to assets '''
    try:       
        assexport = ee.batch.Export.image.toAsset(ee.Image.cat(cmaps, bmaps).clip(aoi),
                                    description='assetExportTask', 
                                    pyramidingPolicy={".default": 'mode'},
                                    assetId=w_exportassetsname.value, scale=w_exportscale.value, maxPixels=1e11)
        assexport.start()
        with w_out: 
            w_out.clear_output() 
            print('Exporting change maps to %s\n task id: %s'%(w_exportassetsname.value,str(assexport.id)))
    except Exception as e:
        with w_out:
            print('Error: %s'%e)                                          
    
w_export_ass.on_click(on_export_ass_button_clicked)  


def on_export_drv_button_clicked(b):
    ''' Export to Google Drive '''
    try:
        fileNamePrefix=w_exportdrivename.value.replace('/','-')            
        gdexport1 = ee.batch.Export.image.toDrive(ee.Image.cat(cmaps,bmaps).byte().clip(aoi),
                                    description='driveExportTask', 
                                    folder = 'gee',
                                    fileNamePrefix=fileNamePrefix,scale=10,maxPixels=1e10)   
        gdexport1.start()
        gdexport2 = ee.batch.Export.image.toDrive(ee.Image.cat(cmaps,bmaps).byte().clip(aoi),
                                    description='driveExportTask', 
                                    folder = 'gee',
                                    fileNamePrefix=fileNamePrefix+'_ATSF',scale=10,maxPixels=1e10)   
        gdexport2.start()
        with w_out:
            w_out.clear_output()
            print('Exporting change maps to Drive/gee/%s\n task id: %s'%(fileNamePrefix,str(gdexport1.id))) 
            print('Exporting ATSF image to Drive/gee/%s\n task id: %s'%(fileNamePrefix+'_ATSF',str(gdexport2.id)))
    except Exception as e:
        with w_out:
            print('Error: %s'%e) 

w_export_drv.on_click(on_export_drv_button_clicked)

def collect_s2():
    with w_out:
        w_out.clear_output()
        try:
            print('Most cloud-free Sentinel-2 RGB image ...')
            collectionid = 'COPERNICUS/S2_SR'
            rgb = ['B4', 'B3', 'B2']
            cloudcover = 'CLOUDY_PIXEL_PERCENTAGE'
            collection_s2 = ee.ImageCollection(collectionid) \
                .filterBounds(aoi) \
                .filterDate(ee.Date(w_startdate.value), ee.Date(w_enddate.value)) \
                .filter(ee.Filter.contains(rightValue=aoi, leftField='.geo')) \
                .sort(cloudcover, True)
            cnt = collection_s2.size().getInfo()
            if cnt == 0:
                raise ValueError('No S2 images found')
            image_s2 = ee.Image(collection_s2.first()).select(rgb).clip(aoi)
            timestamp_s2 = ee.Date(image_s2.get('system:time_start')).getInfo()
            timestamp_s2 = time.gmtime(int(timestamp_s2['value']) / 1000)
            timestamp_s2 = time.strftime('%c', timestamp_s2)
            cloudcover_s2 = image_s2.get(cloudcover).getInfo()
            print('Acquired: %s'%timestamp_s2)
            print('Cloudcover: %s'%cloudcover_s2)
        except Exception as e:
            print('Error: %s' % e)
    return(image_s2)

def collect_naip():
    with w_out:
        w_out.clear_output()
        try:
            print('NAIP Image (NIR,G,B) ...')
            collectionid = 'USDA/NAIP/DOQQ'
            rgb = ['N', 'G', 'B']
            collection_naip = ee.ImageCollection(collectionid) \
                .filterBounds(aoi) \
                .filterDate(ee.Date(w_startdate.value), ee.Date(w_enddate.value))
            cnt = collection_naip.size().getInfo()
            if cnt == 0:
                raise ValueError('No NAIP images found')
            image_naip = rgbLayer(collection_naip.mosaic().clip(aoi).select(rgb))
        except Exception as e:
            print('Error: %s' % e)
    return(image_naip)

def on_view_button_clicked(b):
    ''' View optical imagery '''
    with w_out:
        try:
            jet = 'black,blue,cyan,yellow,red'
            rcy = 'black,red,cyan,yellow'
            mn = 0
            palette = jet
            w_out.clear_output()
            print('Viewing please wait for raster overlay ...')
            if w_visual.value=='S2':
                w_out.clear_output()
                image_s2 = collect_s2()
                mp = ee.Image(image_s2)
                mn = [500, 500, 500]
                mx = [4000, 4000, 4000]
                palette = None
            elif w_visual.value=='NAIP':
                w_classify.disabled=False
                w_out.clear_output()
                image_naip = collect_naip()
                mp = ee.Image(image_naip)
                mn = [0, 0, 0]
                mx = [255, 255, 255]
                palette = None
            if w_dw.value:
                mp = maskNoBuildings(mp)
            if w_maskwater.value==True:
                mp = mp.updateMask(watermask)
            if w_maskchange.value==True:
                if w_changemap.value=='Frequency':
                    mp = mp.updateMask(mp.gte(w_minfreq.value))
                elif w_changemap.value=='ATSF':
                    pass
                else:
                    mp = mp.updateMask(mp.gt(0))
            m.add(TileLayer(url=GetTileLayerUrl(mp.visualize(min=mn, max=mx,
                                  palette=palette)), name=w_visual.value))
        except Exception as e:
            if w_visual.value == 'NAIP':
                w_classify.disabled = True
            print('Error: %s'%e)

w_view.on_click(on_view_button_clicked)

def run():
    ''' Run the interface '''
    global m

    osm = basemap_to_tiles(basemaps.OpenStreetMap.Mapnik)
    ews = basemap_to_tiles(basemaps.Esri.WorldStreetMap)
    ewi = basemap_to_tiles(basemaps.Esri.WorldImagery)

    dc = DrawControl(aoiline={},circlemarker={})
    dc.rectangle = {"shapeOptions": {"fillColor": "#0000ff","color": "#0000ff","fillOpacity": 0.05}}
    dc.polygon = {"shapeOptions": {"fillColor": "#0000ff","color": "#0000ff","fillOpacity": 0.05}}

    dc.on_draw(handle_draw)
    
    lc = LayersControl(position='topright')
    fs = FullScreenControl()

    location = geolocator.geocode('Jülich')
    m = Map(center=(location.latitude, location.longitude),
                    zoom=11,
                    layout={'height': '600px', 'width': '1000px'},
                    layers=(osm, ews, ewi),
                    controls=(dc, lc, fs))
    with w_out:
        w_out.clear_output()
        print('Algorithm output')

    display(m)
    
    return box