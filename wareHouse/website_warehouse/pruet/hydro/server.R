
###  Shiny Server  ###
shinyServer(function(input, output, session) {
  
  ## Map Input ##
  data_of_click <- reactiveValues(clickedMarker = NULL)
  
  ## Build Map ##
  output$map <- renderLeaflet({
    pal <- colorBin(palette = "Reds", 
                   domain = as.numeric(map_df$max_precip), bins = 8, pretty=TRUE)
    leaflet() %>%
      addTiles(urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
               attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>') %>%
      addPolygons(data = skagit, fill = FALSE, stroke = 1, color = 'black') %>% 
      addPolygons(data = whatcom, fill = FALSE, stroke = 1, color = 'black') %>% 
      addPolygons(data = snohomish, fill = FALSE, stroke = 1, color = 'black') %>% 
      setView(lat = 48.35,
              lng = -122,
              zoom = 9) %>%
      addCircleMarkers(
        data = map_df, lng = ~ lng, lat = ~ lat,
        layerId = ~ file_name,
        radius = 6,
        color = ~ pal(max_precip),
        stroke  = FALSE,
        fillOpacity = 0.75) %>% 
      addLegend("topright", pal = pal, values = NULL, title = "Max Precipitation") 
  })
  
  ## Observe Map Input ##
  observeEvent(input$map_marker_click, {
    data_of_click$clickedMarker <- input$map_marker_click
    updateTabsetPanel(session, inputId = "main", selected = "Plots")
  })
  
  ## Plot Output ##
  output$plot = renderPlot({
    
    # Set Initial lat, long #
    if(is.null(data_of_click$clickedMarker$id)){data_of_click$clickedMarker$id <- map_df$file_name[1]}
    
    # Read Data #
    d <- read_RDS(data_of_click$clickedMarker$id, "rcp45") %>%
      mutate(time_stamp = ymd(paste(year, month, day, sep="-")),
             water_year = year(time_stamp + month(3)))
    
    # p1 <- d %>% filter(!is.na(group)) %>% 
    #   calc(1) %>%
    #   ggplot() +
    #   geom_line(aes(x = return_period, y = XT, color = group)) +
    #   labs(x = "Return Period (years)", y = "Storm Intensity (mm/hr)", title = "Max Precip") +
    #   theme_bw()
    
    # p2 <- d %>% filter(!is.na(group)) %>%
    #   calc(.999) %>%
    #   ggplot() +
    #   geom_line(aes(x = return_period, y = XT, color = group)) +
    #   labs(x = "Return Period (years)", y = "Storm Intensity (mm/hr)", title = "99.9 Percentile Precip") +
    #   theme_bw()

    p3 <- d %>% filter(!is.na(group)) %>%
      calc(.99) %>%
      ggplot() +
      geom_line(aes(x = return_period, y = XT, color = group)) +
      labs(x = "Return Period (years)", y = "Storm Intensity (mm/hr)", title = "A) 99 Percentile Precip") +
      theme_bw()

    p4 <- d %>%
      filter(precip >= quantile(precip, .999), !is.na(group)) %>%
      group_by(group) %>% 
      ggplot() +
      stat_ecdf(aes(x = precip, color = group), geom = "line") +
      labs(title = "B) Precip > 99.9 percentile") +
      theme_bw()

    p5 <- d %>%
      filter(precip >= quantile(precip, .95), !is.na(group)) %>%
      ggplot() +
      stat_ecdf(aes(x = precip, color = group), geom = "line") +
      labs(title = "C) Precip > 95th percentile") +
      theme_bw()

    p6 <- d %>%
      ggplot() +
      geom_line(aes(y = precip, x = time_stamp)) +
      geom_point(aes(y = precip, x = time_stamp)) +
      labs(title = "D)") +
      theme_bw()

    # p7 <- d %>% filter(month <= 4 | month >= 9) %>%
    #   group_by(water_year) %>%
    #   filter(precip > 0) %>%
    #   distinct(time_stamp, .keep_all = TRUE) %>%
    #   summarise(rain_days = n()) %>%
    #   ggplot() +
    #   geom_point(aes(x = water_year, y = rain_days)) +
    #   labs(x = "Water Year", y = "No. of days with Precip") +
    #   theme_bw()
    # 
    # p8 <- d %>% distinct(time_stamp, .keep_all = TRUE) %>%
    #   filter(!is.na(group)) %>% 
    #   group_by(month, group) %>%
    #   summarise(month_precip = mean(precip)) %>%
    #   ggplot() +
    #   geom_line(aes(x = month, y = month_precip, color = group)) +
    #   labs(x = "Month", y = "Mean Monthly Precip (mm)") +
    #   theme_bw()
    # 
    # p9 <- d %>% filter(!is.na(group)) %>%
    #   ggplot() +
    #   geom_boxplot(aes(y = precip, x = group)) +
    #   scale_y_log10() +
    #   theme_bw()
    # 
    p10 <- d %>% filter(precip >= quantile(precip, .995), !is.na(group)) %>% 
      group_by(precip, group) %>% 
      summarise(days = n()) %>% 
      arrange(desc(precip)) %>% 
      group_by(group) %>% 
      mutate(days_above = cumsum(days)) %>% 
      ggplot() +
      geom_line(aes(x = precip, y = days_above, color = group)) +
      labs(title = "E) Precip > 99.5%") +
      theme_bw()
    
    p11 <- d %>% filter(precip >= quantile(precip, .95), !is.na(group)) %>% 
      group_by(precip, group) %>% 
      summarise(days = n()) %>% 
      arrange(desc(precip)) %>% 
      group_by(group) %>% 
      mutate(days_above = cumsum(days)) %>% 
      ggplot() +
      geom_line(aes(x = precip, y = days_above, color = group)) +
      labs(title = "F) Precip > 95%") +
      theme_bw()
    

    multiplot(p3, p4, p5, p6, p10, p11, cols = 3)
    
  })
  
  ## Data Output ##
  output$table <- renderTable(d)
  
})
