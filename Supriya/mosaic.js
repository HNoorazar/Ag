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



//########### Expanded that works




var an_IC = ee.ImageCollection(extracted_cloudless_NDVI);
var reduction_geometry = ee.FeatureCollection(pointBuffer);

var start_date_DateType = ee.Date(start_date);
var end_date_DateType = ee.Date(end_date);
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

print ("newcol:", newcol);

var reduced = newcol.map(function(image){
                            return image.reduceRegions({
                                                        collection:reduction_geometry,
                                                        reducer:ee.Reducer.mean(), 
                                                        scale: 10//,
                                                        //tileScale: 16
                                                      });
                                          }
                        ).flatten();
