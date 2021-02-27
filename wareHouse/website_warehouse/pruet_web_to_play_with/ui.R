# Bloom Pruet
####  Header  ####
header <- dashboardHeader(
  title = tags$div(tags$img(src='WSU_logo.png', height = 30), "CAHNRS"))

####  Sidebar  ####
sidebar <- dashboardSidebar(
  sidebarMenu(
    
    ## Home ##
    menuItem("Home", tabName = "home", icon = icon("home"),
             menuSubItem("About", 
                         tabName = "about", 
                         icon = icon("info-circle")),
             
             menuSubItem("People", 
                         tabName = "people", 
                         icon = icon("users")),
             
             menuSubItem("Data Source", 
                         tabName = "data", 
                         icon = icon("table")),
             
             menuSubItem("Contact", 
                         tabName = "contact", 
                         icon = icon("envelope"))
             ),
    
    ## Map ##
    menuItem("Precipitation", 
             tabName = "precip_map", 
             icon = icon("tint")),

    menuItem("Bloom, CPs, and 1st frost", 
             tabName = "bcf_map", 
             icon = icon("tint"))
    # menuItem("Surface Flow", tabName = "surface_map", icon = icon("tint")),
    # menuItem("Dry Days", tabName = "dry_map", icon = icon("tint"))
    
  )
)
####################################
####        ########################
####  Body  ########################
####        ########################
####################################
body <- dashboardBody(
  ##################
  ####         #####
  ####   CSS   #####
  ####         #####
  ##################
  tags$head(
    tags$link(rel = "stylesheet", 
              type = "text/css", 
              href = "custom.css")),
  
  ## Tabs ##
  tabItems(
    
    # Home Tab #
    tabItem(tabName = "home"),
    tabItem(tabName = "about", fluidPage(fluidRow(column(8, offset = 2, includeMarkdown("pages/about.Rmd"))))),
    tabItem(tabName = "people", fluidPage(fluidRow(column(8, offset = 2, includeMarkdown("pages/people.Rmd"))))),
    tabItem(tabName = "data", fluidPage(fluidRow(column(8, offset = 2, includeMarkdown("pages/data.Rmd"))))),
    tabItem(tabName = "contact", fluidPage(fluidRow(column(8, offset = 2, includeMarkdown("pages/contact.Rmd"))))),
    
    # bloom_vs_CP tab
    tabItem(tabName = "bcf_map",
            box(id = "bcf", width = NULL,
                #  Main Map  #
                tabPanel("Map",
                         div(class = "outer",
                             tags$style(type = "text/css", 
                                        "#bcf_map {height: calc(100vh - 125px) !important;}"),
                             leafletOutput("bcf_map")
                            )
                        )
                )
            )
  
    # plot Modal
    bsModal(title = "Bloom vs. CP, and frost", 
            id = "bcf_graphs", trigger = NULL, size = "large",
            fluidPage(fluidRow(column(2, 
                                      radioButtons("bcf_plot_climate_proj", 
                                                   label = h4("Climate Projection"), 
                                                   choices = list("RCP 8.5" = "RCP 8.5",
                                                                  "RCP 4.5" = "RCP 4.5" ),
                                                   selected = "RCP 8.5"),

                                      radioButtons("bcf_plot_fruit_type", 
                                                   label = h4("Fruit type"), 
                                                   choices = list("Cripps Pink" = "Cripps_Pink", 
                                                                  "Gala" = "Gala", 
                                                                  "Red Delicious" = "Red_Deli"),
                                                   selected = "Cripps_Pink")),
                                column(10, offset = 0, plotOutput("bcf_plot", height = 550))
                               )
                      )
            )
  )

####  Combine Dashboard Elements  ####
dashboardPage(
  title="Bloom",
  header,
  sidebar,
  body
)