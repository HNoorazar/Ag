# Water Rights

library(scales)
library(lattice)
# library(ggmap)
library(jsonlite)

library(data.table)
library(shiny)
library(shinydashboard)
library(shinyBS)
library(maps)
library(raster)
library(rgdal)    # for readOGR and others
library(sp)       # for spatial objects
library(leaflet)  # for interactive maps (NOT leafletR here)
library(dplyr)    # for working with data frames
library(ggplot2)  # for plotting
library(reshape2)
library(RColorBrewer)

###################################
#######
####### Reactive
#######
###################################
shinyServer(function(input, output, session) {

  curr_spatial <- reactive({
      water_resource <- input$WaRecRCWCl
      if (water_resource == "surfaceWater") {
           curr_spatial <- spatial_wtr_right %>% 
                           filter(WaRecRCWCl == "surfaceWater") %>% 
                           data.table()

           } else if (water_resource == "groundwater"){
            curr_spatial <- spatial_wtr_right %>% 
                            filter(WaRecRCWCl == "groundwater") %>% 
                             data.table()

           } else if (water_resource == "both_water_resource") {
            curr_spatial <- spatial_wtr_right %>% 
                            filter(WaRecRCWCl %in% c("surfaceWater", 
                                                     "groundwater")
                                  ) %>% 
                            data.table()
       }
    target_date <- as.Date(input$cut_date)
    curr_spatial[, colorr := ifelse(right_date < target_date, "#FF3333", "#0080FF")]

  })
  
  output$water_right_map <- renderLeaflet({
  target_date <- as.Date(input$cut_date)
  leaflet() %>%
  addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
           attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
           layerId = "Satellite",
           options= providerTileOptions(opacity = 0.9)) %>%
  setView(lat = 47, lng = -120, zoom = 7) %>%
  addCircleMarkers(data = curr_spatial(),
                   lng = ~long, lat = ~lat,
                   label = ~ popup,
                   layerId = ~ location,
                   radius = 3,
                   color = ~ colorr,
                   stroke  = FALSE,
                   fillOpacity = .95 
                  )
  })  
})


# shinyServer(function(input, output, session) {
  # This observer is responsible for maintaining map
  # according to the variables the user has chosen

  ###################################
  #######
  ####### Observe
  #######
  ###################################
  
  # observe({
       # water_resource <- input$WaRecRCWCl
       # if (water_resource == "surfaceWater") {
       #     curr_spatial <- spatial_wtr_right %>% 
       #                     filter(WaRecRCWCl == "surfaceWater")
       #     curr_spatial <- data.table(curr_spatial)

       #     } else if (water_resource == "groundwater"){
       #      curr_spatial <- spatial_wtr_right %>% 
       #                      filter(WaRecRCWCl == "groundwater")
       #      curr_spatial <- data.table(curr_spatial)

       #     } else if (water_resource == "both_water_resource") {
       #      curr_spatial <- spatial_wtr_right %>% 
       #                      filter(WaRecRCWCl %in% c("surfaceWater", 
       #                                               "groundwater")
       #                            )
       #      curr_spatial <- data.table(curr_spatial)
       # }
      
      # target_date <- as.Date(input$cut_date)
      # curr_spatial[, colorr := ifelse(right_date < target_date, "#FF3333", "#0080FF")]

      # leafletProxy("water_right_map", data = curr_spatial) %>%
      # clearShapes() %>%
      # addCircleMarkers(data = curr_spatial, 
      #                  lng = ~long, lat = ~lat,
      #                  label = ~ popup,
      #                  layerId = ~ location,
      #                  radius = 3,
      #                  color = ~ colorr,
      #                  stroke  = FALSE,
      #                  fillOpacity = .95 
      #                  )
  #})

  # output$water_right_map <- renderLeaflet({
  #    leaflet() %>%
  #    addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  #             attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
  #             layerId = "Satellite",
  #             options= providerTileOptions(opacity = 0.9)) %>%

  #    setView(lat = 47, lng = -120, zoom = 7)
  # })
# })


###################################
#######
####### original. before reactive and observe
#######
###################################
# shinyServer(function(input, output, session) {
 
  
  # output$water_right_map <- renderLeaflet({
  #    target_date <- as.Date(input$cut_date)

  #    spatial_wtr_right <- data.table(spatial_wtr_right)
  #    spatial_wtr_right[, colorr := ifelse(right_date < target_date, "#FF3333", "#0080FF")]

  #    leaflet() %>%
  #    addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  #             attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
  #             layerId = "Satellite",
  #             options= providerTileOptions(opacity = 0.9)) %>%

  #    # addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
  #    #          attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>') %>%
  #    setView(lat = 47, lng = -120, zoom = 7) %>%
  #    addCircleMarkers(data = spatial_wtr_right, 
  #                     lng = ~long, lat = ~lat,
  #                     label = ~ popup,
  #                     layerId = ~ location,
  #                     radius = 3,
  #                     color = ~ colorr,
  #                     stroke  = FALSE,
  #                     fillOpacity = .95 
  #                     )
   
  # })

# })

