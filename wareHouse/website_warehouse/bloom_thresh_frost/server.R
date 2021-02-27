# Bloom Pruet

###  Shiny Server  ###
shinyServer(function(input, output, session) {
  ###############################################################
  #######
  #######      Precip PART
  #######
  ###############################################################

  ## Load Data ####
  precip_map_data <- reactive({
    
      filter(spatial_precip, 
             climate_proj == input$precip_map_climate_proj,
             group == input$precip_map_climate_group, 
             exceedance == input$precip_map_exceedance,
             time_scale == input$precip_map_time_scale)

  })
  
  precip_data_month <- reactive({

    # Test if location is selected
    validate(
      need(!is.null(input$precip_map_marker_click$id), 
            "Please select a location")
    )

    # load data
    readRDS(paste0("/data/pruett/precip/", 
                    input$precip_plot_time_scale, 
                    "_prob_month/", 
                    input$precip_map_marker_click$id, 
                    ".rds")) %>% 
      filter(climate_proj == input$precip_plot_climate_proj)
  })
  
  precip_data_octmar <- reactive({

    # Test if location is selected
    validate(
      need(!is.null(input$precip_map_marker_click$id), 
           "Please select a location")
    )
    
    # load data
    readRDS(paste0("/data/pruett/precip/", 
                   input$precip_plot_time_scale, 
                   "_prob_octmar/", 
                   input$precip_map_marker_click$id, 
                   ".rds")) %>% 
      filter(climate_proj == input$precip_plot_climate_proj)
    
  })
  ##################
  ##
  ##  Build Map ####
  ##
  ##################
  output$precip_map <- renderLeaflet({
    pal <- colorBin(palette = "plasma", reverse = TRUE,
                    domain = precip_map_data()$prob_median, 
                    bins = 8, pretty=TRUE)
    leaflet() %>%
    addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
             attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>') %>%
    addPolygons(data = skagit, fill = FALSE, stroke = 1, color = 'black') %>% 
    addPolygons(data = whatcom, fill = FALSE, stroke = 1, color = 'black') %>% 
    addPolygons(data = snohomish, fill = FALSE, stroke = 1, color = 'black') %>% 
    setView(lat = 48.35, lng = -121.5, zoom = 8) %>%
    addCircleMarkers(data = precip_map_data(), lng = ~ lng, lat = ~ lat,
                     label = ~ file_name,
                     layerId = ~ file_name,
                     radius = 6,
                     color = ~ pal(prob_median),
                     stroke  = FALSE,
                     fillOpacity = .95) # %>% 
    # addLegend("bottomleft", pal = pal, values = NULL, 
    #           title = "Difference from Exceedance Probability") 
      
  })

  ## Observe Map Input ##
  observeEvent(input$precip_map_marker_click, {
    
    toggleModal(session, modalId = "precip_graphs", toggle = "open")
    
  })
  ##########################
  ##
  ## Plot precip Output ####
  ##
  ##########################
  output$precip_plot <- renderPlot({
      p_month <- plot_monthly_prob(precip_data_month(), "Daily Probability")
      p_octmar <- plot_octmar_prob(precip_data_octmar())
      
      plot_grid(p_month, p_octmar, 
                nrow = 1, align = "vh", 
                rel_widths = c(4, 1), 
                axis='b')
    
  }, res = 140)

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
                                         "bloom/plots/", 
                                         "CM_locs/bloom_thresh_in_one",
                                         "/no_obs/apple/thresh_75/")
               toggleModal(session, modalId = "bcf_graphs", toggle = "open")

              
               output$bcf_plot <- renderImage({
                     image_name <- paste0(lat, "_-", long, "_", 
                                          gsub(" ", "_", input$bcf_plot_climate_proj), 
                                          "_", 
                                          input$bcf_plot_fruit_type,
                                          ".png")

                     filename <- normalizePath(file.path(file_dir_string, image_name))
                     # Return a list containing the filename and alt text
                     list(src = filename, width = 800, height = 550)
                                              }, deleteFile = FALSE
                                              )
            })
  ###############################################################
  #######
  #######      DRY PART
  #######
  ###############################################################
  # dry_days_map_data <- reactive({
  #   spatial_dry_days %>% 
  #     filter(climate_proj == input$dry_days_map_climate_proj,
  #            group == input$dry_days_map_climate_group, 
  #            exceedance == input$dry_days_map_exceedance)
  # })

  # dry_days_data <- reactive({

  #   # Test if location is selected
  #   validate(
  #             need(!is.null(input$dry_map_marker_click$id), 
  #             "Please select a location")
  #           )
  #   # load data
  #   readRDS(paste0("/data/pruett/combined/data/", 
  #                  input$dry_map_marker_click$id))
    
  # })
  
  # output$dry_map <- renderLeaflet({
  #   pal <- colorBin(palette = "plasma", reverse = TRUE,
  #                   domain = dry_days_map_data()$prob_median, bins = 8, pretty=TRUE)
    
  #   leaflet() %>%
  #     addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
  #              attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>') %>%
  #     addPolygons(data = skagit, fill = FALSE, stroke = 1, color = 'black') %>% 
  #     addPolygons(data = whatcom, fill = FALSE, stroke = 1, color = 'black') %>% 
  #     addPolygons(data = snohomish, fill = FALSE, stroke = 1, color = 'black') %>% 
  #     setView(lat = 48.35, lng = -121.5, zoom = 8) %>%
  #     addCircleMarkers(data = dry_days_map_data(), 
  #                      lng = ~ lng, lat = ~ lat,
  #                      label = ~ file_name,
  #                      layerId = ~ file_name,
  #                      radius = 6,
  #                      color = ~ pal(prob_median),
  #                      stroke  = FALSE,
  #                      fillOpacity = .95) %>% 
  #     addLegend("bottomleft", pal = pal, values = NULL, title = "Difference from Exceedance Probability") 
  # })

  # observeEvent(input$dry_map_marker_click, {
    
  #   toggleModal(session, modalId = "dry_days_graphs", toggle = "open")
    
  #              })
  # output$dry_days_plot <- renderPlot({
      
  #     # p_month <- plot_monthly_prob(dry_days_data_month(), "Monthly Probability")
  #     # p_octmar <- plot_octmar_prob(dry_days_data_octmar())
      
  #     # plot_grid(p_month, p_octmar, nrow = 1, align = "vh", rel_widths = c(4, 1), axis = 'b')
    
  #   plot_drydays_boxplot(dry_days_data(), input$dry_days_plot_climate_proj)
    
  # }, res = 70, width = 400)
  ###############################################################
  #######
  #######      SURFACE PART
  #######
  ###############################################################
  
  # surface_map_data <- reactive({
    
  #   spatial_surface %>% 
  #     filter(climate_proj == input$surface_map_climate_proj,
  #            group == input$surface_map_climate_group, 
  #            exceedance == input$surface_map_exceedance)
    
  # })
  
  # surface_data_month <- reactive({

  #   # Test if location is selected
  #   validate(
  #     need(!is.null(input$surface_map_marker_click$id), 
  #          "Please select a location")
  #   )

  #   # load data
  #   readRDS(paste0("/data/pruett/surface/daily_prob_month/", 
  #                  input$surface_map_marker_click$id)) %>% 
  #     filter(climate_proj == input$surface_plot_climate_proj)
    
  # })
  
  # surface_data_octmar <- reactive({

  #   # Test if location is selected
  #   validate(
  #     need(!is.null(input$surface_map_marker_click$id), 
  #          "Please select a location")
  #   )

  #   # load data
  #   readRDS(paste0("/data/pruett/surface/daily_prob_octmar/", 
  #                  input$surface_map_marker_click$id)) %>% 
  #     filter(climate_proj == input$surface_plot_climate_proj)
    
  # })
  
  # output$surface_map <- renderLeaflet({
  #   pal <- colorBin(palette = "plasma", reverse = TRUE,
  #                   domain = c(-.1, .5), bins = 6, pretty=TRUE)

  #   leaflet() %>%
  #     addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
  #              attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>') %>%
  #     addPolygons(data = skagit, fill = FALSE, stroke = 1, color = 'black') %>% 
  #     addPolygons(data = whatcom, fill = FALSE, stroke = 1, color = 'black') %>% 
  #     addPolygons(data = snohomish, fill = FALSE, stroke = 1, color = 'black') %>% 
  #     setView(lat = 48.35, lng = -121.5, zoom = 8) %>%
  #     addCircleMarkers(data = surface_map_data(), 
  #                      lng = ~ lng, lat = ~ lat,
  #                      label = ~ file_name,
  #                      layerId = ~ file_name,
  #                      radius = 6,
  #                      color = ~ pal(prob_median),
  #                      stroke  = FALSE,
  #                      fillOpacity = .95) %>% 
  #     addLegend("bottomleft", pal = pal, 
  #               values = NULL, 
  #               title = "Difference from Exceedance Probability") 
    
  # })
  
  # observeEvent(input$surface_map_marker_click, {
    
  #   toggleModal(session, modalId = "surface_graphs", toggle = "open")
    
  # })
  
  # output$surface_plot <- renderPlot({
      
  #     p_month <- plot_monthly_prob(surface_data_month(), "Monthly Probability")
  #     p_octmar <- plot_octmar_prob(surface_data_octmar())
      
  #     plot_grid(p_month, p_octmar, nrow = 1, align = "vh", rel_widths = c(4, 1), axis = 'b')
    
  # }, res = 140)


})
