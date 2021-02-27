# Bloom Pruet

###  Shiny Server  ###
shinyServer(function(input, output, session) {
  ##################
  ##
  ##  Build Map
  ##
  ##################

  ## Observe Map Input ##
  observeEvent(input$precip_map_marker_click, {
    
    toggleModal(session, modalId = "precip_graphs", toggle = "open")
    
  })

  ###############################################################
  #######
  #######      Bloom PART
  #######
  ###############################################################
  spatial_bcf_data <- reactive({
    spatial_bcf
  })

  ##################
  ##
  ##  Build Map ####
  ##
  ##################

  output$bcf_map <- renderLeaflet({
    pal <- colorBin(palette = "plasma", reverse = TRUE,
                    domain = spatial_bcf_data()$lat, bins = 8, pretty=TRUE)
    leaflet() %>%
    addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
             attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>') %>%
    setView(lat = 47, lng = -120, zoom = 7) %>%
    addCircleMarkers(data = spatial_bcf_data(), 
                     lng = ~ long, lat = ~ lat,
                     label = ~ location,
                     layerId = ~ location,
                     radius = 4,
                     color = ~ pal(lat),
                     stroke  = FALSE,
                     fillOpacity = .95)
  })

  # plot part of bloom
  observeEvent(input$bcf_map_marker_click, 
             { p <- input$bcf_map_marker_click$id
               lat <- substr(as.character(p), 1, 8)
               long <- substr(as.character(p), 13, 21)
               print(p)
               file_dir_string <- paste0("/data/hnoorazar/", 
                                         "bloom_thresh_frost/plots/", 
                                         "CM_locs/bloom_thresh_in_one",
                                         "/no_obs/apple/thresh_75/")
               toggleModal(session, modalId = "bcf_graphs", toggle = "open")

               curr_emission <- gsub(" ", "_", input$bcf_plot_climate_proj)
               output$bcf_plot <- renderImage({
                     image_name <- paste0(lat, "_-", long, "_", 
                                          curr_emission, "_", 
                                          input$bcf_plot_fruit_type,
                                          ".png")

                     filename <- normalizePath(file.path(file_dir_string, image_name))
                     # Return a list containing the filename and alt text
                     list(src = filename, width = 800, height = 550)
                                              }, deleteFile = FALSE
                                              )
            })
})
