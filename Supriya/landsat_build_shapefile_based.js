
// var SF_grids = ee.FeatureCollection(SF_table);
// print("Number of polygons in shapefile is", SF_table.size());

var SF_cnty = ee.FeatureCollection(SF_county);


////////////////////////////////////////////////////////////////////////////////////////
///
///                           Set up the damn parameters here
///
////////////////////////////////////////////////////////////////////////////////////////
var start_date = '2010-01-01';
var end_date   = '2012-12-10';
var cloud_perc = 10;
var satellite_collection = 'LANDSAT/LT05/C01/T1_TOA';

var l5 = ee.ImageCollection(satellite_collection)
           .filterDate(start_date, end_date);

////////////////////////////////////////////////////////////////////////////////////////
///
///                           functions definitions start
///
////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////
///
///  Function to clear clouds in Landsat
///

function clean_clouds_from_one_image_landsat(img){
  // toss out cloudy pixels
  // The following is not robust anyway:
  // https://developers.google.com/earth-engine/apidocs/ee-algorithms-landsat-simplecloudscore
  var cloud = ee.Algorithms.Landsat.simpleCloudScore(img);
  var mask = cloud.select(['cloud']).lte(cloud_perc);
  var clean_image = img.updateMask(mask);
  return (clean_image);
}

////////////////////////////////////////////////
///
///  Add NDVI
///

var addNDVI = function(image) {
  var ndvi = image.normalizedDifference(['B4', 'B3']).rename('NDVI');
  return image.addBands(ndvi);
};

function add_NDVI_collection(image_IC){
  var NDVI_IC = image_IC.map(addNDVI);
  return NDVI_IC;
}


////////////////////////////////////////////////
///
/// add system_time_start to an imageCollection
///

function add_Date_Time_image(image) {
  image = image.addBands(image.metadata('system:time_start').rename("system_start_time"));
  return image;
}

function add_Date_Time_collection(colss){
 var c = colss.map(add_Date_Time_image);
 return c;
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
  // var doy = image.date().getRelative('day', 'year');
  var doy = ee.Date(image.get('system:time_start')).getRelative('day', 'year');
  var doyBand = ee.Image.constant(doy).uint16().rename('doy');
  doyBand = doyBand.updateMask(image.select('B1').mask()); // Why this is a problem in landsat????

  return image.addBands(doyBand);
}

function addDoY_to_collection(collec){
  var C = collec.map(addDoY_to_image);
  return C;
}

////////////////////////////////////////////////
///
///         Do the Job function
///

function extract_landsat_IC(a_feature, start_date, end_date){
  var geom = a_feature.geometry(); // a_feature is a feature collection
  var newDict = {'original_polygon_1': geom};
  var imageC = ee.ImageCollection(satellite_collection)
                 .filterDate(start_date, end_date)
                 .filterBounds(geom)
                 .map(clean_clouds_from_one_image_landsat)
                 .map(function(image){return image.clip(geom)})
                 //.filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_perc)) // does not work in landsat
                 .sort('system:time_start', true);
    
  imageC = add_NDVI_collection(imageC);   // add NDVI as a band
  imageC = addYear_to_collection(imageC); // add year as a band
  imageC = addDoY_to_collection(imageC);  // add DoY as a band
  imageC = add_Date_Time_collection(imageC);
  
  imageC = imageC.map(function(im){return(im.set(newDict))});
  return imageC;
}


function mosaic_and_reduce_IC_mean(an_IC, a_feature, start_date, end_date){
  an_IC = ee.ImageCollection(an_IC);
  //print('mosaic_start_date:', start_date);
  //var reduction_geometry = ee.Feature(ee.Geometry(an_IC.get('original_polygon')));
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

    var image = ee.Image(filtered.mosaic()); // Make the mosaic

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
  
  return(reduced);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Define a point of interest. Use the UI Drawing Tools to import a point
// geometry and name it "point" or set the point coordinates with the
// ee.Geometry.Point() function as demonstrated here.

var point = ee.Geometry.Point([-116.4, 43.565]);
var pointBuffer = point.buffer({'distance': 1000}); // Apply the buffer method to the Point object.
pointBuffer = ee.Feature(pointBuffer); // convert the polygon type to a ee.Feature.
print ("pointBuffer:", pointBuffer);
print ("pointBuffer.geometry():", pointBuffer.geometry());

Map.centerObject(point, 11);
Map.addLayer(pointBuffer, {color: 'red'});

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

var ROI = pointBuffer.geometry();

// var cloudy_NDVI = l5.filterBounds(ROI).map(addNDVI);
// var chart = ui.Chart.image.series({
//   imageCollection: cloudy_NDVI.select('NDVI'),
//   region: ROI,
//   reducer: ee.Reducer.mean(),
//   scale: 500
// }).setOptions({title: 'Cloudy NDVI (Buffer)'});
// print(chart);



// var cloudlessNDVI = l5.filterBounds(ROI)
//                       .map(clean_clouds_from_one_image_landsat)
//                       .map(addNDVI);

// var chart_cloudlessNDVI = ui.Chart.image.series({
//   imageCollection: cloudlessNDVI.select('NDVI'),
//   region: ROI,
//   reducer: ee.Reducer.mean(),
//   scale: 10
// }).setOptions({title: 'Cloudless NDVI (Buffer)'});
// print ("chart_cloudlessNDVI", chart_cloudlessNDVI);


/////////////////////////////////////////////////////////////////////
///
///     Pull, Reduce, and export 
///
/////////////////////////////////////////////////////////////////////

var extracted_cloudless_NDVI = extract_landsat_IC(pointBuffer, start_date, end_date);

// print ("extracted_cloudless_NDVI:", extracted_cloudless_NDVI);
// print (ui.Chart.image.series({
//   imageCollection: extracted_cloudless_NDVI.select('NDVI'),
//   region: ROI,
//   reducer: ee.Reducer.mean(),
//   scale: 10
// }).setOptions({title: 'Cloudless NDVI (Buffer)'}));



/////////////////////////////////////////////////////////////////////
///
///     Expand Mosaic Function Below
///
/////////////////////////////////////////////////////////////////////
//
// Below, the reduction_geometry, will be the "a_feature" input of Mosaic function.
//          var reduced = mosaic_and_reduce_IC_mean(imageC, reduction_geometry, wstart_date, wend_date);
//

var reduced = mosaic_and_reduce_IC_mean(extracted_cloudless_NDVI, pointBuffer, start_date, end_date);

// var reduced = extracted_cloudless_NDVI.map(function(image){
//                           return image.reduceRegions({
//                                                       collection:ROI,
//                                                       reducer:ee.Reducer.mean(), 
//                                                       scale: 10
//                                                     });
//                                         }
//                         ).flatten();

print ("reduced", reduced);

/////////////////////////////////////////////////////////////////////
///
///     Expand Mosaic Function Above
///
/////////////////////////////////////////////////////////////////////

var outfile_name = 'buffer_region_reduced_' + cloud_perc + 'cloud';
Export.table.toDrive({
  collection: reduced,
  description:outfile_name,
  folder:"Supriya",
  fileNamePrefix: outfile_name,
  fileFormat: 'CSV',
  selectors:["system_start_time", "image_year", "doy", "NDVI"]
});



////////////////////////////////////////////////////////////////////////////////////////////////
////////////
////////////   Change region of interest to the shapefile
////////////
////////////////////////////////////////////////////////////////////////////////////////////////





