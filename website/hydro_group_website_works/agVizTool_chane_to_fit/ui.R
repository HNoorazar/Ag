library(leaflet)
library(shinyBS)
library(shiny)
library(plotly)
library(shinydashboard)
# Map menu
overlays <- c( "Satellite"="Satellite", 
               "Topographic"="Topographic", 
               "Basic"="Basic"
              )
boundaries <- c("State"="State",
                "County"="County", 
                "District"="District"
               )

indicators <- c("Net Primary Productivity" = "npp",
                "Inter-annual Forage Variability" = "nppsd",
                "Vegetation Type Trajectory" = "mc2",
                "Heat Stress Index" = "hsi",
                "Vulnerability Index" = "vulstk"
                )

decades <- c( "2010-2020" = 2,
              "2020-2030" = 3,
              "2030-2040" = 4,
              "2040-2050" = 5,
              "2050-2060" = 6,
              "2060-2070" = 7,
              "2070-2080" = 8,
              "2080-2090" = 9,
              "2090-2100" = 10
            )
climateModels <- c("A1B" = "a1b",
                   "A2" = "a2",
                   "B2" = "b2"
                  )

heatStress <- "Heat Stress - Negative physiological effects on cattle due to temperature"
netPrimaryProductivity <- "Net Primary Productivity - Carbon uptake after subtracting Plant Respiration from Gross Primary Primary Productivity"
forageVariability <- "Inter-annual Forage Variability - Standard deviation in annual average forage quantity"
VegetationType <- "Vegeation Type Trajectory - Ratio of edible to inedible vegetation"
vulnerabilityIndex <- "Vulnerability Index - Aggregate of four indicators"

shinyUI(
         navbarPage( title = div( ""), 
                     id="nav", windowTitle = "Rangelands",
                     tabPanel( "Map",
                               fluidPage( id = "nav", inverse=FALSE, fluid=FALSE, title="Tool",
                                          div( class="outer",
                                               tags$head( includeCSS("styles.css"),
                                                          includeScript("gomap.js")),
                                               leafletOutput("map", width="100%", height="100%"),
                                               absolutePanel( id="menuPanel", draggable=FALSE, width=330, height="auto",
                                                              left="auto", right=20, bottom="auto", top=60, fixed=TRUE,
                                                              inverse=TRUE, h2("Map Key"),
                                                              selectInput(inputId = "tileSelect", label = "Map Overlay", choices = overlays),
                                                              selectInput(inputId = "boundaries", label = "Map Boundaries", choices = boundaries),
                                                              selectInput(inputId = "Indicators", label = "Indicators", choices = indicators),
                                                              selectInput(inputId = "Decades", label = "Time Period", choices = decades),
                                                              selectInput(inputId = "ClimateModel", label = "Climate Model", choices = climateModels)
                                                            )
                                              )
                                        ),
                               fluidPage(bsModal( "Graphs", trigger=NULL, title = "", size="large",
                                                  dashboardPage( dashboardHeader(title = "Plots"),
                                                                 dashboardSidebar( radioButtons(inputId = "climate", label = "Scenarios", choices = climateModels),
                                                                                   radioButtons(inputId = "indicator", label = "Indicators", choices = indicators)),
                                                                 dashboardBody( plotOutput("Plot"), 
                                                                                p(heatStress), 
                                                                                p(netPrimaryProductivity),
                                                                                p(forageVariability), 
                                                                                p(VegetationType), 
                                                                                p(vulnerabilityIndex)
                                                                              )
                                                                )
                                                 )
                                        )
                              )
                    )
)
