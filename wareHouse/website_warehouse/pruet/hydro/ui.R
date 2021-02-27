
###  Header Code  ###
header <- dashboardHeader(
  title = tags$div(tags$img(src='WSU_logo.png', height = 40),
                   "Climate Visualization"), titleWidth = 300
)

###  Body Code  ###
body <- dashboardBody(
  
  ## CSS ##
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "custom.css")
  ),
  
  ## Main Tab Box##
  tabBox(
  id = "main", width = NULL,
  
  #  Main Map  #
  tabPanel("Map",
           div(
             class = "outer",
             tags$style(type = "text/css",
                        "#map {height: calc(100vh - 165px) !important;}"),
             leafletOutput("map")
           )),
  
  # Plots #
  tabPanel("Plots",
           value = "Plots",
           div(
             class = "outer",
             tags$style(type = "text/css",
                        "#plot {height: calc(100vh - 165px) !important;}"),
             plotOutput("plot")
           )
           ),
  # Data #
  tabPanel("Data",
           value = "Data",
           div(
             class = "outer",
             tags$style(type = "text/css",
                        "#table {height: calc(100vh - 165px) !important;}"),
             tableOutput('table')
           )
  )
)
)

###  Combine Dashboard Elements  ###
dashboardPage(
  header,
  dashboardSidebar(disable = TRUE),
  body
)
