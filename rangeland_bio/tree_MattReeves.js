////////////////////////////////

//// SET PARAMETERS HERE ///////

////////////////////////////////
 

// Raster data to run zonal stats on
var collection_name = 'projects/rap-data-365417/assets/vegetation-cover-v3' //projects/rangeland-analysis-platform/vegetation-cover-v3';


// Specify which band to query. If the band name isn't specified (it wasn't named) it's probably "b1"
// If you need to list band names, uncomment the following line and look at what it prints out

print('Band Names', ee.ImageCollection(collection_name).first().bandNames());

var band_name = "SHR";

 

// Zone/Geometry asset path

var zones_name = table;

 

// Set the meters/pixel scale you want to process the data at. You can set it to a different

// resolution than the underlying data to affect processing time. Or just set it to the data size

var processing_scale = 30; 

// Set the output name (without extension, it'll be a CSV)

var output_name = 'Hudson_Proposed';

// Set an output folder. Leave blank to just write to root Drive

var output_folder = 'Tess';

 

// Define a reducer to crunch the data. The combine function efficiently chains

// multiple reducers. In this case, mean and standard deviation

var redu = ee.Reducer.mean();

  // .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})

  // .combine({reducer2: ee.Reducer.count(), sharedInputs: true});

 

////////////////////////////////

////////////////////////////////

 

 

var col = ee.ImageCollection(collection_name);

var single_band = col.select(band_name);

 

// Add it to the map with some visualization parameters

Map.addLayer(single_band, {palette:['red', 'yellow', 'green']}, band_name);

 

 

// Get the geometry file

var zones = ee.FeatureCollection(zones_name);

// print('example zone', zones.first());

 

// Add it to the map if you want

// Map.addLayer(zones);

 

 

// Turn the image collection into a single image with bands from each single-band image in the collection

var img = single_band.toBands();

 

// Reduce the image collection on the zones

var samp = img.reduceRegions({

  collection: zones,

  reducer: redu,

  scale: processing_scale,

  tileScale: 1, // NEW 2022-01-12, increases the size allowed to crunch at once

});

 

 

// print('Stats', samp.first());

// print('Whole', samp);

 

// var samp2 = samp.first();

// samp2 = ee.Feature(samp2);

// print('samp2', samp2);

// samp2 = ee.Feature(null).setMulti(samp2.toDictionary());

// print('null', samp2);

 

var fc = samp.map(function(f) {

  f = ee.Feature(f);

  var g = ee.Feature(null).setMulti(f.toDictionary());

  return g;

});

 

//print('fc', fc);

 

// Export the data as a CSV to Cloud Storage. Can also export to Drive.

// Remember you still have to press "Run" in the "Tasks" pane after this.

Export.table.toDrive({

  collection: fc,

  description: output_name,

  folder: output_folder,

  fileFormat: 'CSV'

});

