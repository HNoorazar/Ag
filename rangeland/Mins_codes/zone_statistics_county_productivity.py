import arcpy,csv
from arcpy.sa import *
import math

arcpy.env.workspace = "U:/Projects/Rangeland/SpatialData/SpatialData.gdb"
outcsv_path = "U:/Projects/Rangeland/"
county = Raster("county90m")
ecozone = Raster("ecol3_90m")
gridmet = Raster("gridmet90m")
rangepixel = Raster("range1_250m") # "U:/Projects/Rangeland/Rangelands_v1/Rangelands_v1.tif")
path_productivity = "U:/Projects/Rangeland/ForageDownload/"


outcsv_file = outcsv_path + "county_annual_productivity.csv"
foutcsv = open(outcsv_file, "w")
foutcsv.write("year,county,productivity\n")

for year in range(1984,2022): #,2022):
  if year <= 2019:
    prod = Raster(path_productivity + "rngprod_" + str(year))
  else:
    prod = Raster(path_productivity + "rngprod_" + str(year) + ".tif")
  arcpy.env.extent = rangepixel
  arcpy.env.mask = rangepixel
  arcpy.env.cellSize = 250
  
  if arcpy.Exists("test"):
    arcpy.Delete_management("test")
    #range = Con(rangeland.RANGE_NUM == 1, 1)
  test = prod + 0
  test.save("test")
  outtable = "testtable"
  if arcpy.Exists("testtable"):
    arcpy.Delete_management("testtable")
  outZonalStatistics = ZonalStatisticsAsTable(county, "Value", test,outtable,"DATA","MEAN")
  #outZonalStatistics.save("test_zone")
  fields = ['Value','MEAN']
  with arcpy.da.SearchCursor("testtable", fields) as cursor:
    for row in cursor:
      #print(u'{0}, {1}, {2}'.format(row[0], row[1], row[2]))
      outtxt = str(year) + ',' + str(row[0]) + ',' + str(row[1]) + "\n"
      foutcsv.write(outtxt)
foutcsv.close()