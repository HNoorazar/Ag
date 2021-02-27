#
# from 
# https://shiny.rstudio.com/reference/shiny/latest/updateSelectInput.html
#
# useful in goddamn update reactivity thing:
# https://stackoverflow.com/questions/28379937/change-selectize-choices-but-retain-previously-selected-values
#


shinyServer(function(input, output, session) {
  curr_spatial <- spatial_wtr_right
  current_selection <- reactiveVal(NULL)

  observeEvent(input$subbasins_id, {
      curr_spatial <- curr_spatial %>% 
                      filter(subbasin %in% input$subbasins_id) %>% 
                      data.table()

      subbasin_to_plot <- unique(curr_spatial$subbasin)
      # print ("___________________________")
      # print ("subbasin_to_plot")
      # print (subbasin_to_plot)
      # print ("___________________________")
    })

  output$water_right_map <- renderLeaflet({
    target_date <- as.Date(input$cut_date)
    curr_spatial[, colorr := ifelse(right_date < target_date, 
                                    "#FF3333", "#0080FF")]

    #########################################################
    if (input$water_source_type == "surfaceWater") {
      curr_spatial <- spatial_wtr_right %>%
                      filter(WaRecRCWCl == "surfaceWater")%>%
                      data.table()

     } else if (input$water_source_type == "groundwater"){
      curr_spatial <- spatial_wtr_right %>%
                      filter(WaRecRCWCl == "groundwater")%>%
                      data.table()
       } else {
          curr_spatial <- data.table(spatial_wtr_right)
    }
    #########################################################
    #########################################################
    # observeEvent(input$countyType_id, {
      #######################
      curr_spatial <- curr_spatial %>% 
                      filter(WRIA_NM == input$countyType_id) %>% 
                      data.table()
      
      subbasins <- sort(unique(curr_spatial$subbasin))
      #######################
     
      ## Can also set the label and select items
      # updateSelectInput(session,
      #                   inputId = 'subbasins_id',
      #                   choices = subbasins,
      #                   selected = head(subbasins, 1)
      #                   )
    # })
    ############################################# 
    #############################################
    # current_selection <- reactiveVal(NULL)
    observeEvent(input$subbasins_id, {
             current_selection(input$subbasins_id)
             })

    # now if you are updating your menu
    # print(current_selection())
    # print(subbasins)
    # print (!(current_selection() %in% subbasins))
    # print(is.null(current_selection()))
    # print(is.null(subbasins))
    curr_s <- current_selection()
    print (curr_s)

    # if (length(curr_s)>1){
    #   curr_s <- sort(curr_s)[1]
    #   if (!(curr_s %in% subbasins)){
    #   sss <- subbasins[1]
    #   }
    #  } else {
    #     sss <- current_selection()
    # }

    if (!(curr_s %in% subbasins)){
      sss <- subbasins[1]
      } else {
        sss <- current_selection()
    }

    updateSelectInput(session, 
                      inputId = "subbasins_id", 
                      choices = subbasins,
                      selected = sss)
  #############################################
    water_map <- build_map(data_dt = curr_spatial, 
                           sub_bas = curr_s)
    water_map

  })

})

