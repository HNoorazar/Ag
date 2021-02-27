# Bloom Pruet
####  Header  ####
header <- dashboardHeader(
  title = tags$div(tags$img(src='WSU_logo.png', height = 30), "CAHNRS"))

####  Sidebar  ####
sidebar <- dashboardSidebar(
  sidebarMenu(
    
    ## Home ##
    menuItem("Home", tabName = "home", icon = icon("home"),
             menuSubItem("About", tabName = "about", icon = icon("info-circle")),
             menuSubItem("People", tabName = "people", icon = icon("users")),
             menuSubItem("Data Source", tabName = "data", icon = icon("table")),
             menuSubItem("Contact", tabName = "contact", icon = icon("envelope"))),
    
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
    tags$link(rel = "stylesheet", type = "text/css", href = "custom.css")),
  
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
        ),
    
    # Precip Tab #
    tabItem(tabName = "precip_map",
            box(id = "precip",
                width = NULL,
      
                #  Main Map  #
                tabPanel("Map",
                         div(class = "outer",
                             tags$style(type = "text/css", "#precip_map {height: calc(100vh - 125px) !important;}"),
                             leafletOutput("precip_map"),
                             absolutePanel(id = "controls", 
                                           class = "panel panel-default", fixed = TRUE,
                                           draggable = TRUE, 
                                           top = 90, left = "auto", 
                                           right = 40, bottom = "auto",
                                           width = 250, height = "auto" ,
                                           h3(tags$b("Explorer")),
                                           radioButtons("precip_map_climate_proj", 
                                                        label = "Climate Projection", 
                                                        choices = list("RCP 4.5" = "rcp45", 
                                                                       "RCP 8.5" = "rcp85"),
                                                        selected = "rcp85"),
                                           radioButtons("precip_map_exceedance", 
                                                        label = "Exceedance Value", 
                                                        choices = list("20th Percentile" = "prob_80", 
                                                                       "10th Percentile" = "prob_90",
                                                                       "5th Percentile" = "prob_95"),
                                                        selected = "prob_80"),
                                           radioButtons("precip_map_climate_group", 
                                                        label = "Climate Group", 
                                                        choices = list("2040s" = "2040s", 
                                                                       "2060s" = "2060s", 
                                                                       "2080s" = "2080s"),
                                                        
                                                        selected = "2040s"),
                                           radioButtons("precip_map_time_scale", 
                                                        label = "Time Scale", 
                                                        choices = list("Daily" = "day", 
                                                                       "Weekly" = "week", 
                                                                       "Monthly" = "month"),
                                                        selected = "day")
                                           )
                            
                            )
                        )
                )
            ) # ,

    # ## Surface Water Tab 
    # tabItem(tabName = "surface_map",
    #         box(id = "surface",
    #             width = NULL,
              
    #             #  Main Map  #
    #             tabPanel("Map",
    #                      div(class = "outer",
    #                          tags$style(type = "text/css", "#surface_map {height: calc(100vh - 125px) !important;}"),
    #                          leafletOutput("surface_map"),
    #                          absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
    #                                        draggable = TRUE, top = 90, left = "auto", right = 40, bottom = "auto",
    #                                        width = 250, height = "auto",
    #                                        h3(tags$b("Explorer")),
    #                                        radioButtons("surface_map_climate_proj", label = "Climate Projection", 
    #                                                     choices = list("A1B" = "A1B", "B1" = "B1"),
    #                                                     selected = "A1B"),
    #                                        radioButtons("surface_map_exceedance", label = "Exceedance Value", 
    #                                                     choices = list("20th Percentile" = "prob_80", 
    #                                                                    "10th Percentile" = "prob_90",
    #                                                                    "5th Percentile" = "prob_95"),
    #                                                     selected = "prob_80"),
    #                                        radioButtons("surface_map_climate_group", label = "Climate Group", 
    #                                                     choices = list("2040s" = "2040s", "2060s" = "2060s", "2080s" = "2080s"),
    #                                                     selected = "2040s"))
    #                           )
    #                      )
    #             )
    #        )# , 
    # ## Dry Days Tab 
    # tabItem(tabName = "dry_map",
    #         box(id = "dry_days",
    #             width = NULL,
              
    #             #  Main Map  #
    #             tabPanel("Map",
    #                      div(class = "outer",
    #                          tags$style(type = "text/css", "#dry_map {height: calc(100vh - 125px) !important;}"),
    #                          leafletOutput("dry_map"),
    #                          absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
    #                                        draggable = TRUE, top = 90, left = "auto", right = 40, bottom = "auto",
    #                                        width = 250, height = "auto",
    #                                        h3(tags$b("Explorer")),
    #                                        radioButtons("dry_days_map_climate_proj", label = "Climate Projection", 
    #                                                     choices = list("RCP 4.5" = "rcp45", "RCP 8.5" = "rcp85")),
    #                                        radioButtons("dry_days_map_exceedance", label = "Exceedance Value", 
    #                                                     choices = list("20th Percentile" = "prob_80", 
    #                                                                    "10th Percentile" = "prob_90",
    #                                                                    "5th Percentile" = "prob_95")),
    #                                        radioButtons("dry_days_map_climate_group", label = "Climate Group", 
    #                                                     choices = list("2040s" = "2040s", "2060s" = "2060s", "2080s" = "2080s")))
    #                         )
    #                     )
    #             )
    #         )
    
    ),
  
  # plot Modal
  bsModal(title = "Bloom vs. CP, and frost", 
          id = "bcf_graphs", trigger = NULL, size = "large",
          fluidPage(fluidRow(column(2, 
                                    radioButtons("bcf_plot_climate_proj", 
                                                 label = h4("Climate Projection"), 
                                                 choices = list("RCP 4.5" = "RCP 4.5", 
                                                                "RCP 8.5" = "RCP 8.5"),
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
         ),

  bsModal(title = "Precipitation", 
          id = "precip_graphs", 
          trigger = NULL, 
          size = "large",
          fluidPage(fluidRow(column(2, 
                                    radioButtons("precip_plot_climate_proj", 
                                                 label = h4("Climate Projection"), 
                                                 choices = list("RCP 4.5" = "rcp45", 
                                                                "RCP 8.5" = "rcp85"),
                                                 selected = "rcp85"),

                                    radioButtons("precip_plot_time_scale", 
                                                 label = h4("Time Scale"), 
                                                 choices = list("Daily" = "day", 
                                                                "Weekly" = "week", 
                                                                "Monthly" = "month"),
                                                 selected = "day")),
                             column(10, offset = 0, plotOutput("precip_plot", height = 550))
                            )
                    )
         ) # ,
  
  # bsModal(title = "Surface Flow Summary", id = "surface_graphs", trigger = NULL, size = "large",
  #         fluidPage(fluidRow(column(2, radioButtons("surface_plot_climate_proj", label = h3("Climate Projection"), 
  #                                                   choices = list("A1B" = "A1B", "B1" = "B1"), 
  #                                                   selected = "A1B")),
  #                           column(10, offset = 0, plotOutput("surface_plot", height = 550))
  #                           )
  #                  )
  #        ),
  
  # bsModal(title = "Dry Days Summary", id = "dry_days_graphs", trigger = NULL, size = "large",
  #         fluidPage(fluidRow(column(2, radioButtons("dry_days_plot_climate_proj", label = h3("Climate Projection"), 
  #                                                   choices = list("RCP 4.5" = "rcp45", 
  #                                                                  "RCP 8.5" = "rcp85"),
  #                                                   selected = "rcp85")),
  #                            column(10, offset = 0, plotOutput("dry_days_plot", height = 550))
  #                           )
  #                  )
  #         )
  )

####  Combine Dashboard Elements  ####
dashboardPage(
  title="Bloom",
  header,
  sidebar,
  body
)