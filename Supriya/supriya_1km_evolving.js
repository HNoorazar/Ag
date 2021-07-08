

var SF = ee.FeatureCollection(SF_table);
print("Number of polygons in shapefile is", SF_table.size());


var xmin = -117.1;
var xmax = -115.7;

var ymin = 43.2;
var ymax = 43.93;

var xmed1 = (xmin + xmax) / 2.0;
var xmed2 = (xmin + xmax) / 2.0;

var rectangle_1 = ee.Geometry.Polygon([[xmin, ymin], [xmin, ymax], [xmed1, ymax], [xmed1, ymin], [xmin, ymin]]);
var big_rectangle = [rectangle_1];


var SF_regions = ee.FeatureCollection(big_rectangle);
var reduction_geometry = ee.FeatureCollection(SF);
///////////////////
/////////////////// Visualize the rectangle
///////////////////
// Display the polygons by adding them to the map.
Map.centerObject(rectangle_1, 7);
Map.addLayer(SF, {color: 'blue'}, 'geodesic polygon');
Map.addLayer(rectangle_1, {color: 'FF0000'}, 'geodesic polygon');

// var empty = ee.Image().byte();
// var outlinectOriginal = empty.paint({
//   featureCollection: SF_regions,
//   color: 1,
//   width: 2
// });
// Map.addLayer(outlinectOriginal, {palette: '0000ff'}, 'SF_regions');



////////////////////////////////////////////////////////////////////////////////////////
///
///                           functions definitions start
///
////////////////////////////////////////////////////////////////////////////////////////
///
///  Function to mask clouds using the Sentinel-2 QA band.
///

function maskS2clouds(image) {
    var qa = image.select('QA60');

    // Bits 10 and 11 are clouds and cirrus, respectively.
    var cloudBitMask = 1 << 10;
    var cirrusBitMask = 1 << 11;

    // Both flags should be set to zero, indicating clear conditions.
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
                        qa.bitwiseAnd(cirrusBitMask).eq(0));

    // Return the masked and scaled data, without the QA bands.
    return image.updateMask(mask).divide(10000)
                          .select("B.*")
                          .copyProperties(image, ["system:time_start"]);
}

////////////////////////////////////////////////
///
///     add Year (Perhaps can be ommited and obtained in python from `system_start_time`)
///

function addYear_to_image(image){
  var year = image.date().get('year');
  var yearBand = ee.Image.constant(year).uint16().rename('image_year');
  return image.addBands(yearBand);
}

function addYear_to_collection(collec){
  var C = collec.map(addYear_to_image);
  return C;
}


////////////////////////////////////////////////
///
/// add Day of Year (Perhaps can be ommited and obtained in python from `system_start_time`)
///

function addDoY_to_image(image){
  var doy = image.date().getRelative('day', 'year');
  // var doy = ee.Date(image.get('system:time_start')).getRelative('day', 'year');
  var doyBand = ee.Image.constant(doy).uint16().rename('doy');
  doyBand = doyBand.updateMask(image.select('B8').mask());

  return image.addBands(doyBand);
}

function addDoY_to_collection(collec){
  var C = collec.map(addDoY_to_image);
  return C;
}


////////////////////////////////////////////////
///
/// add Date of image to an imageCollection (Perhaps can be ommited and obtained in python from `system_start_time`)
///

function add_system_start_time_image(image) {
  image = image.addBands(image.metadata('system:time_start').rename("system_start_time"));
  return image;
}

function add_system_start_time_collection(colss){
 var c = colss.map(add_system_start_time_image);
 return c;
}


////////////////////////////////////////////////
///
///         EVI
///

function addEVI_to_image(image) {
  var evi = image.expression(
                      '2.5 * ((NIR - RED) / (NIR + (6 * RED) - (7.5 * BLUE) + 1.0))', {
                      'NIR': image.select('B8'),
                      'RED': image.select('B4'),
                      'BLUE': image.select('B2')
                  }).rename('EVI');
  return image.addBands(evi);
}

function add_EVI_collection(image_IC){
  var EVI_IC = image_IC.map(addEVI_to_image);
  return EVI_IC;
}


////////////////////////////////////////////////
///
///         Do the Job function
///

function extract_sentinel_IC(a_feature,start_date, end_date){
    var geom = a_feature.geometry(); //a_feature is a feature collection
    var newDict = {'original_polygon_1': geom};
    var imageC = ee.ImageCollection('COPERNICUS/S2')
                .filterDate(start_date, end_date)
                .filterBounds(geom)
                // Pre-filter to get less cloudy granules.
                //.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', "less_than", cloud_perc)
                .map(function(image){return image.clip(geom)})
                .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_perc))
                //.filter('CLOUDY_PIXEL_PERCENTAGE < 70')
                .sort('system:time_start', true);
    
    // toss out cloudy pixels
    imageC = imageC.map(maskS2clouds);
    
    
    // add year as a band
    imageC = addYear_to_collection(imageC);
    
    // add DoY as a band
    imageC = addDoY_to_collection(imageC);
    
    imageC = add_system_start_time_collection(imageC);
    
    // add EVI as a band
    imageC = add_EVI_collection(imageC);
    
    // add original geometry to each image
    // we do not need to do this really:
    imageC = imageC.map(function(im){return(im.set(newDict))});
    
    // add original geometry and WSDA data as a feature to the collection
    imageC = imageC.set({ 'original_polygon': geom});
  return imageC;
}

function mosaic_and_reduce_IC_mean(an_IC,a_feature,start_date,end_date){
  an_IC = ee.ImageCollection(an_IC);
  var reduction_geometry = a_feature;
  var start_date_DateType = ee.Date(start_date);
  var end_date_DateType = ee.Date(end_date);
  //######**************************************
  // Difference in days between start and end_date

  var diff = end_date_DateType.difference(start_date_DateType, 'day');

  // Make a list of all dates
  var range = ee.List.sequence(0, diff.subtract(1)).map(function(day){
                                    return start_date_DateType.advance(day,'day')});

  // Funtion for iteraton over the range of dates
  function day_mosaics(date, newlist) {
    // Cast
    date = ee.Date(date);
    newlist = ee.List(newlist);

    // Filter an_IC between date and the next day
    var filtered = an_IC.filterDate(date, date.advance(1, 'day'));

    // Make the mosaic
    var image = ee.Image(filtered.mosaic());

    // Add the mosaic to a list only if the an_IC has images
    return ee.List(ee.Algorithms.If(filtered.size(), newlist.add(image), newlist));
  }

  // Iterate over the range to make a new list, and then cast the list to an imagecollection
  var newcol = ee.ImageCollection(ee.List(range.iterate(day_mosaics, ee.List([]))));
  //print("newcol 1", newcol);
  //######**************************************

  var reduced = newcol.map(function(image){
                            return image.reduceRegions({
                                                        collection:reduction_geometry,
                                                        reducer:ee.Reducer.mean(), 
                                                        scale: 10//,
                                                        //tileScale: 16
                                                      });
                                          }
                        ).flatten();
                          
  reduced = reduced.set({ 'original_polygon': reduction_geometry});
  
  return(reduced);
}

// remove geometry on each feature before printing or exporting

var myproperties=function(feature){
  feature=ee.Feature(feature).setGeometry(null);
  return feature;
};


// remove geometry on each feature before printing or exporting

var myproperties=function(feature){
  feature=ee.Feature(feature).setGeometry(null);
  return feature;
};

/////////
///////// Main Body
/////////

var wstart_date = '2016-01-01';
var wend_date = '2016-12-31';
var cloud_perc = 70;

var imageC = extract_sentinel_IC(SF_regions, wstart_date, wend_date);
var reduced = mosaic_and_reduce_IC_mean(imageC, reduction_geometry, wstart_date, wend_date);  
var featureCollection = reduced;


var outfile_name = 'IDb_Canyon_Grid_1km' + cloud_perc ;
Export.table.toDrive({
  collection: featureCollection,
  description:outfile_name,
  folder:"IDb_Canyon_Grid_1km",
  fileNamePrefix: outfile_name,
  fileFormat: 'CSV',
  selectors:["id", "doy", "EVI",
             "system_start_time", "image_year", "B8"]
});








