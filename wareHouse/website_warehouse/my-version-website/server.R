#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)

# Define server logic required to draw a histogram
shinyServer(function(input, output, session) {
   
  output$distPlot <- renderPlot({
    
    # generate bins based on input$bins from ui.R
    x    = faithful[, 2] 
    bins = seq(min(x), max(x), length.out = input$bins + 1)
    
    # draw the histogram with the specified number of bins
    hist(x, breaks = bins, col = 'darkgray', border = 'white')
    
  })
  
  output$aboutText = renderText({
                              rawText = readLines('Data.txt')
                              return(rawText)
                            })
  
  output$PeopleNames = renderText({
                                "Introducing Shiny"
                                })
  
  # for the leaflet test
#  points = eventReactive(input$recalc, {cbind(rnorm(40) * 2 + 13, rnorm(40) + 48)}, ignoreNULL = FALSE)
#  output$mymap = renderLeaflet({ leaflet() %>% 
#      # providers$Stamen.TonerLite
#      addProviderTiles(providers$Esri.NatGeoWorldMap, options = providerTileOptions(noWrap = TRUE)) %>%
#      addMarkers(data = points())
#  })

#################################################### location group

  output$location_group <- renderImage({
    return(list(src = "./plots/location_group.png",
                contentType = "image/png",
                width=386.88,
                height=450,
                alt = "location_group")
          )
      }, deleteFile = FALSE)

#################################################### Bloom Plots
  output$bloom_45 <- renderImage({
    return(list(src = "./plots/bloom_rcp45_main.png",
                contentType = "image/png",
                width=525,
                height=525,
                alt = "bloom_45")
          )
      }, deleteFile = FALSE)

  output$bloom_85 <- renderImage({
    return(list(src = "./plots/bloom_rcp85_main.png",
                contentType = "image/png",
                width=525,
                height=525,
                alt = "bloom_85")
          )
      }, deleteFile = FALSE)
#################################################### Degree Day Plots
output$cumdd_45 <- renderImage({
    return(list(src = "./plots/cumdd_45.png",
                contentType = "image/png",
                width=525,
                height=525,
                alt = "cumdd_45")
          )
      }, deleteFile = FALSE)

output$cumdd_85 <- renderImage({
    return(list(src = "./plots/cumdd_85.png",
                contentType = "image/png",
                width=525,
                height=525,
                alt = "cumdd_85")
          )
      }, deleteFile = FALSE)
#################################################### Egg Hatch
output$eggHatch_45 <- renderImage({
    return(list(src = "./plots/eggHatch_45.png",
                contentType = "image/png",
                width=525,
                height=525,
                alt = "eggHatch_45")
          )
      }, deleteFile = FALSE)

output$eggHatch_85 <- renderImage({
    return(list(src = "./plots/eggHatch_85.png",
                contentType = "image/png",
                width=525,
                height=525,
                alt = "eggHatch_85")
          )
      }, deleteFile = FALSE)
#################################################### Generations by Aug 23
output$Adult_Gen_Aug23_rcp45 <- renderImage({
    return(list(src = "./plots/Adult_Gen_Aug23_rcp45.png",
                contentType = "image/png",
                width=525,
                height=425,
                alt = "Adult_Gen_Aug23_rcp45")
          )
      }, deleteFile = FALSE)


output$Adult_Gen_Aug23_rcp85 <- renderImage({
    return(list(src = "./plots/Adult_Gen_Aug23_rcp85.png",
                contentType = "image/png",
                width=525,
                height=425,
                alt = "Adult_Gen_Aug23_rcp85")
          )
      }, deleteFile = FALSE)



output$Larva_Gen_Aug23_rcp45 <- renderImage({
    return(list(src = "./plots/Larva_Gen_Aug23_rcp45.png",
                contentType = "image/png",
                width=525,
                height=425,
                alt = "Larva_Gen_Aug23_rcp45")
          )
      }, deleteFile = FALSE)


output$Larva_Gen_Aug23_rcp85 <- renderImage({
    return(list(src = "./plots/Larva_Gen_Aug23_rcp85.png",
                contentType = "image/png",
                width=525,
                height=425,
                alt = "Larva_Gen_Aug23_rcp85")
          )
      }, deleteFile = FALSE)
#################################################### Adult Emergence
output$emergence_rcp45 <- renderImage({
    return(list(src = "./plots/emergence_rcp45.png",
                contentType = "image/png",
                width=525,
                height=425,
                alt = "emergence_rcp45")
          )
      }, deleteFile = FALSE)

output$emergence_rcp85 <- renderImage({
    return(list(src = "./plots/emergence_rcp85.png",
                contentType = "image/png",
                width=525,
                height=425,
                alt = "emergence_rcp85")
          )
      }, deleteFile = FALSE)



#################################################### Diapause Populations
output$diapause_rel_rcp45 <- renderImage({
    return(list(src = "./plots/diapause_rel_rcp45.png",
                contentType = "image/png",
                width=525,
                height=425,
                alt = "diapause_rel_rcp45")
          )
      }, deleteFile = FALSE)


output$diapause_rel_rcp85 <- renderImage({
    return(list(src = "./plots/diapause_rel_rcp85.png",
                contentType = "image/png",
                width=525,
                height=425,
                alt = "diapause_rel_rcp85")
          )
      }, deleteFile = FALSE)



output$diapause_abs_rcp45 <- renderImage({
    return(list(src = "./plots/diapause_abs_rcp45.png",
                contentType = "image/png",
                width=525,
                height=425,
                alt = "diapause_abs_rcp45")
          )
      }, deleteFile = FALSE)


output$diapause_abs_rcp85 <- renderImage({
    return(list(src = "./plots/diapause_abs_rcp85.png",
                contentType = "image/png",
                width=525,
                height=425,
                alt = "diapause_abs_rcp85")
          )
      }, deleteFile = FALSE)


})

