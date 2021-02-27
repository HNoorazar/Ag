library(leaflet)
library(RColorBrewer)
library(scales)
library(lattice)
library(dplyr)
library(raster)
library(sp)
library(rgdal)
library(ggmap)
library(jsonlite)
library(ggplot2)
library(reshape2)

shinyServer(function(input, output, session) {
  
  # Create the map
  output$map <- renderLeaflet({
    m <- leaflet(options = leafletOptions(zoomControl = TRUE,
                                          minZoom = 4, maxZoom = 9,
                                          dragging = TRUE)) %>%
         addTiles() %>%
         setView(lng = -103.85, lat = 39.45, zoom = 4)
  })
  
  # Show page on click event...
  observeEvent(input$map_shape_click, 
               { p <- input$map_shape_click
                toggleModal(session, modalId = "Graphs", toggle = "open")
                if (input$boundaries == "District"){
                                      district <- readOGR("shp/district.shp", layer = "district")
                                      # get polygon of current selected district(boundary)
                                      dat <- data.frame(Longitude = c(p$lng),Latitude =c(p$lat))
                                      coordinates(dat) <- ~ Longitude + Latitude
                                      proj4string(dat) <- proj4string(district)        
                                      currentDistrictName = toString(over(dat,district)$GEOID)
                                      neDistricts <- subset(district, district$GEOID %in% c(currentDistrictName))
                                      # get data based on only that district
                                      ## Example RasterLayer
                                      r <- raster(nrow=1e3, ncol=1e3, crs=proj4string(neDistricts))
                                      r[] <- 1:length(r)
                                      
                                      ## crop and mask
                                      r2 <- crop(r, extent(neDistricts))
                                      r3 <- mask(r2, neDistricts)
                                      
                                      # plot(r3)
                                      # r3 is is the data in the shape file 
                                      # that the user selected.
                  } else if (input$boundaries == "County"){
                      county <- readOGR("shp/county.shp", layer = "county")
                      # get polygon of current selected county(boundary)
                      dat <- data.frame(Longitude = c(p$lng),Latitude =c(p$lat))
                      coordinates(dat) <- ~ Longitude + Latitude
                      proj4string(dat) <- proj4string(county)
                      currentCountyName = toString(over(dat,county)$NAME)
                      neCounties <- subset(county, county$NAME %in% c(currentCountyName))
                      # get data based on only that county
                      ## Example RasterLayer
                      r <- raster(nrow=1e3, ncol=1e3, crs=proj4string(neCounties))
                      r[] <- 1:length(r)
                      ## crop and mask
                      r2 <- crop(r, extent(neCounties))
                      r3 <- mask(r2, neCounties)
                      # plot(r3)
                      # r3 is is the data in the shape file that the user selected.
                  } else { #(input$boundaries == "State")
                      state <- readOGR("shp/state.shp", layer = "state")
                      # get polygon of current selected state(boundary)
                      dat <- data.frame(Longitude = c(p$lng),Latitude =c(p$lat))
                      coordinates(dat) <- ~ Longitude + Latitude
                      proj4string(dat) <- proj4string(state)
                      currentStateName = toString(over(dat,state)$NAME)
                      neStates <- subset(state, state$NAME %in% c(currentStateName))
                      # get data based on only that state
                      ## Example RasterLayer
                      r <- raster(nrow=1e3, ncol=1e3, crs=proj4string(neStates))
                      r[] <- 1:length(r)
                      ## crop and mask
                      r2 <- crop(r, extent(neStates))
                      r3 <- mask(r2, neStates)
                      # plot(r3)
                      # r3 is is the data in the shape file that the user selected.
                }

                output$Plot <- renderPlot({ # ggplot() + geom_boxplot()
                                           s <- stack(paste("tif/", input$climate, input$indicator, ".tif", sep=""))
                                            for (i in 1:nlayers(s)){
                                              if (i>1){
                                                if(input$indicator ==  "hsi"){
                                                  if(input$climate == "b2"){
                                                    paste("r", i, sep="")[s[[i]] == 157] = NA
                                                    s[s[[i]] == 157] = NA
                                                    s[s[[i]] == 255] = -1
                                                    s[s[[i]] == 254] = -2
                                                  } 
                                                  else{
                                                    s[s[[i]] == 157] = NA
                                                    s[s[[i]] == 0] = NA
                                                    s[s[[i]] == 255] = -1
                                                    s[s[[i]] == 254] = -2
                                                  }
                                                  
                                                }
                                                else if(input$indicator ==  "vulstk"){
                                                }
                                                else{
                                                  s[s[[i]]== 157] = NA
                                                  s[s[[i]]== 255] = -1
                                                  s[s[[i]]== 254] = -2
                                                }
                                                assign(paste("r", i, sep=""), projectRaster(s[[i]], r2))
                                              }
                                            }
                                            
                                            df = data.frame(r2=values(r2), r3=values(r3),
                                                            r4=values(r4), r5=values(r5),
                                                            r6=values(r6), r7=values(r7),
                                                            r8=values(r8), r9=values(r9),
                                                            r10=values(r10))
                                                          
                                            if (input$indicator == "vulstk"){
                                              lim <- c(-8,8)
                                             }
                                             else {
                                                lim <-c(-2,2)
                                            }
                                            
                                            if (input$indicator == "npp"){
                                              plot_title <- "Net Primary Productivity"
                                             }
                                             else if (input$indicator == "nppsd"){
                                              plot_title <- "Inter-annual Forage Variability"
                                             }
                                             else if (input$indicator == "mc2"){
                                              plot_title <- "Vegetation Type Trajectory"
                                             }
                                             else if (input$indicator == "hsi"){
                                              plot_title <- "Heat Stress Index"
                                             }
                                             else {
                                                plot_title <- "Vulnerability Index"
                                            }
                                            boxplot(df, main=plot_title, 
                                                    ylim=lim, 
                                                    names=seq(2010, 2090, by=10), 
                                                    xlab="Decade", 
                                                    ylab="Aggregate Values")    
                                          })
  })
  
  state = NULL
  county = NULL
  district = NULL
  FPtoState <- read_json('FPtoState.txt')

  loadCounty <- function(county){
    if(is.null(county)) {
      county <- readOGR("shp/county.shp", layer = "county")
      labels <- sprintf("<strong>%s</strong><br/><strong>%s</strong>",
                        county$NAME, county$STATEFP) %>% 
                lapply(htmltools::HTML)
      
      leafletProxy("map") %>% 
      addPolygons(data = county, 
                  weight=1,col = 'white', 
                  group = "County", 
                  fill=TRUE, 
                  fillOpacity = 0, 
                  highlight = highlightOptions(weight = 5,
                                               color = "#666",
                                               dashArray = "",
                                               fillOpacity = 0,
                                               bringToFront = TRUE),
                  label = labels,
                  labelOptions = labelOptions(style = list("font-weight" = "normal", 
                                                           padding = "3px 8px"),
                                              textsize = "15px",
                                              direction = "auto"))   
    }
  }

  loadDistrict <- function(district){
    if(is.null(district)) {
      district <- readOGR("shp/district.shp", layer = "district")
      
      labels <- sprintf("<strong>%s</strong><br/><strong>%s</strong>",
                        district$CD115FP, district$STATEFP) %>% 
                lapply(htmltools::HTML)
      
      leafletProxy("map") %>% 
      addPolygons(data = district, 
                  weight=1, 
                  col = 'white', 
                  group ="District", label = labels,
                  fill=TRUE, fillOpacity = 0, 
                  highlight = highlightOptions(weight = 5,
                                               color = "#666",
                                               dashArray = "",
                                               fillOpacity = 0,
                                               bringToFront = TRUE),
                  labelOptions = labelOptions(style = list("font-weight" = "normal", 
                                                            padding = "3px 8px"),
                                              textsize = "15px",
                                              direction = "auto"))
    }
  }

  loadState <- function(state){
    if(is.null(state)) {
      state <- readOGR("shp/state.shp", layer = "state")
      labels <- sprintf("<strong>%s</strong><br/>", state$NAME) %>% 
                lapply(htmltools::HTML)
      
      leafletProxy("map") %>% 
      addPolygons(data = state, weight=1, 
                  col = 'white', 
                  group = "State", fill=TRUE, 
                  fillOpacity = 0, 
                  highlight = highlightOptions(weight = 5,
                                               color = "#666",
                                               dashArray = "",
                                               fillOpacity = 0,
                                               bringToFront = TRUE),
                  label = labels,
                  labelOptions = labelOptions(style = list("font-weight" = "normal", 
                                                           padding = "3px 8px"),
                                              textsize = "15px",
                                              direction = "auto"))
    }
  }
  
    # check changes in Satellite/Topographic/Basic view 
  observe({
    if (input$tileSelect == "Satellite"){
      leafletProxy("map") %>%
        addTiles(urlTemplate = "http://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                 attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
                 layerId = "Satellite")
    }
    else if (input$tileSelect == "Topographic"){
      leafletProxy("map") %>%
        addTiles(urlTemplate = "http://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                 attribution = '&copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
                 layerId = "Topographic")
    }
    else #(input$tileSelect == "Basic")
    {
      leafletProxy("map") %>%
        addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
                 attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>',
                 layerId = "Basic")
    }
  })
  
  # check changes in county/districts
  observe({
    if (input$boundaries == "District"){  
      district <- loadDistrict(district)    
      leafletProxy("map") %>% hideGroup("County") %>% hideGroup("State") %>% showGroup("District")
     } else if (input$boundaries == "County"){     
      county <- loadCounty(county)
      leafletProxy("map") %>% 
      hideGroup("District") %>% 
      hideGroup("State") %>% 
      showGroup("County")
     } else { #(input$boundaries == "State")
      state <- loadState(state)   
      leafletProxy("map") %>% hideGroup("County") %>% hideGroup("District") %>% showGroup("State")      
    }
  })
  
  # change to time period or model
  observe({
      {
        foo <- input$tileSelect
        foo2 <- input$boundaries
        
        d <- raster(paste("tif/",input$ClimateModel,input$Indicators,".tif",sep=""), band=input$Decades)
        d <- crop(d, extent(d, 0, 294, 0, 381))
        
        # set data ranges and transform values from unsigned to signed
        if(input$Indicators ==  "hsi"){
          if(input$ClimateModel == "b2"){
            d[d == 157] = NA
            d[d == 255] = -1
            d[d == 254] = -2
           } 
           else{
            d[d == 157] = NA
            d[d == 0] = NA
            d[d == 255] = -1
            d[d == 254] = -2
          }
          v <- c(-2,2)
          b <- 5
         }
         else if(input$Indicators ==  "vulstk"){
          v <- c(-8,8)
          b <- 9
         }
         else {
           d[d== 157] = NA
           d[d== 255] = -1
           d[d== 254] = -2
           v <- c(-2,2)
           b <- 5
        }
        
        # emission scenario 'b2' is actually 'b1'
        if(input$ClimateModel == "b2" && input$Indicators == "mc2"){
          label <- "b1mc2"
         }
         else{
          label <- paste(input$ClimateModel,input$Indicators,sep=" ")
        }
        
        pal <- colorNumeric(c("#8b0000","#ffffff", "#006400"), v, na.color = "transparent")
        leafletProxy("map") %>%
        clearControls() %>%
        clearImages() %>%
        addRasterImage(d, colors = pal, opacity = 1, maxBytes=Inf) %>%
        addLegend(position = "bottomleft", pal = pal, values =  v,  title =  label, bins=b)
      }
    })
})