import arcpy,csv
from arcpy.sa import *
import math

#def tableToCSV(input_tbl, csv_filepath):
#    fld_list = arcpy.ListFields(input_tbl)
#    fld_names = [fld.name for fld in fld_list]
#    with open(csv_filepath, 'wb') as csv_file:
#        writer = csv.writer(csv_file)
#        writer.writerow(fld_names)
#        with arcpy.da.SearchCursor(input_tbl, fld_names) as cursor:
#            for row in cursor:
#                writer.writerow(row)
#        print(csv_filepath + " CREATED")
#    csv_file.close()



arcpy.env.workspace = "U:/Projects/Rangeland/SpatialData/SpatialData.gdb"
outcsv_path = "U:/Projects/Rangeland/"
county = Raster("county90m")
ecozone = Raster("ecol3_90m")
gridmet = Raster("gridmet90m")
rangepixel = Raster("range1") # "U:/Projects/Rangeland/Rangelands_v1/Rangelands_v1.tif")

xmin = float(arcpy.management.GetRasterProperties(rangeland,"LEFT").getOutput(0))
xmax = float(arcpy.management.GetRasterProperties(rangeland,"RIGHT").getOutput(0))
ymin = float(arcpy.management.GetRasterProperties(rangeland,"BOTTOM").getOutput(0))
ymax = float(arcpy.management.GetRasterProperties(rangeland,"TOP").getOutput(0))
cellsize = int(arcpy.management.GetRasterProperties(rangeland,"CELLSIZEX").getOutput(0))
cols = int(arcpy.management.GetRasterProperties(rangeland,"COLUMNCOUNT").getOutput(0))
rows = int(arcpy.management.GetRasterProperties(rangeland,"ROWCOUNT").getOutput(0))
arcpy.env.cellSize = cellsize

outcsv_file = outcsv_path + "comb_county_gridmet_range.csv"
foutcsv = open(outcsv_file, "w")
foutcsv.write("county,gridmet,count\n")

#divided into smaller region
dcols = 10
drows = 8
dsub_cols = int(math.ceil(cols / dcols))
dsub_rows = int(math.ceil(rows / drows))
#subymin = ymin
for row in range(0,drows):
  subymin = ymin + row * cellsize * dsub_rows
  subymax = subymin + cellsize * dsub_rows
  if (subymax > ymax):
    subymax = ymax
  print("row:" + str(row) + "\tymin:" + str(subymin) + "\tymax:" + str(subymax))
  subxmin = xmin
  for col in range(0,dcols):
    subxmin = xmin + col * cellsize * dsub_cols
    subxmax = subxmin + cellsize * dsub_cols
    if (subxmax > xmax):
      subxmax = xmax
    print("col:" + str(col) + "\txmin:" + str(subxmin) + "\txmax:" + str(subxmax))
    #combine
    arcpy.env.extent = arcpy.Extent(subxmin,subymin,subxmax,subymax)  
    arcpy.env.mask = rangepixel    
    if arcpy.Exists("test"):
      arcpy.Delete_management("test")
    #range = Con(rangeland.RANGE_NUM == 1, 1)
    test = Combine([county, gridmet])
    test.save("test")
    #arcpy.management.BuildRasterAttributeTable(test,"Overwrite")
    fields = ['county90m','gridmet90m','Count']
    with arcpy.da.SearchCursor("test", fields) as cursor:
        for row in cursor:
          #print(u'{0}, {1}, {2}'.format(row[0], row[1], row[2]))
          outtxt = str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + "\n"
          foutcsv.write(outtxt)
    #update location
    #subxmin += cellsize * dsub_cols
  #subymin += cellsize * dsub_rows
foutcsv.close()