# Per_MS Meeting

library(shiny)
library(leaflet)

# Define server logic required to draw a histogram
shinyServer(function(input, output, session) {
  # pal <- colorBin("YlOrRd", domain = curr_SF@data$Source)
  zoom_level = 9
  map_center_lat <- 47.3
  map_center_long <- -119

  output$mymap = renderLeaflet({
                        if (input$Field_type == "Double-cropped potentials"){
                            curr_SF <- Grant_2015_2018_correct_years_2_SF
                             } else {
                            curr_SF <- Grant_2015_2018_correct_years_all_SF
                        }
                        
                        ss <- as.numeric(input$Survey_Year)
                        curr_SF <- curr_SF[grepl(ss, curr_SF$LstSrvD), ]
                        factpal <- colorFactor(topo.colors(5), curr_SF@data$Source)
                        leaflet() %>%
                        addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                                 attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
                                 layerId = "Satellite",
                                 options= providerTileOptions(opacity = 0.8)) %>%
                        setView(lat = map_center_lat, lng = map_center_long, zoom = zoom_level) %>%
                        addPolygons(data = curr_SF,
                                    stroke = TRUE, 
                                    fillOpacity = 0.1, 
                                    smoothFactor = 0.9,
                                    color = ~factpal(Source)
                                    # fillColor = ~factpal(Source)  # ~pal(Source) - curr_SF@data$Source [Did not work]
                                    )  %>% 
                        addLegend(pal = factpal, values = curr_SF$Source, 
                                  position = "bottomleft", opacity = 1) 

                })
           
})



