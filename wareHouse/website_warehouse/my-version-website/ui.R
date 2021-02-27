# pull up the packages
library(ggplot2)
library(shiny)
library(leaflet)


ui = navbarPage("Hydro Lab", inverse=TRUE, collapsible=TRUE,
                
                # Home Tab starts here
                tabPanel(title = "Home", 
                         fluidRow( theme = "slate.css", 
                           column(6, offset = 0,  
                                 navlistPanel(
                                   
                                   tabPanel(title = "About", 
                                            br(), 
                                            bootstrapPage(htmlTemplate("home_page/about-template.html", name = "About"))
                                            ),
                                   
                                   tabPanel(title = "People", 
                                            br(),
                                            bootstrapPage(htmlTemplate("home_page/people-template.html", name = "People"))
                                            ),
                                   
                                   tabPanel(title = "Codling Moth Life Cycle and Management",
                                            br(),
                                            bootstrapPage(htmlTemplate("home_page/Codling_Moth_Template.html", name = "Moth"))
                                            ),
                                   
                                   tabPanel(title = "climate Data",
                                            br(),
                                            bootstrapPage(htmlTemplate("home_page/climate-data-template.html", name = "Climate"))
                                            ),
                                   
                                   tabPanel(title = "What's the Story?", 
                                            br(),
                                            bootstrapPage(htmlTemplate("home_page/story-template.html", name = "Story"))
                                            ),
                                   
                                   tabPanel(title = "Contact",
                                            br(),
                                            bootstrapPage(htmlTemplate("home_page/contact-template.html", name = "Story"))
                                            ),
                                   
                                   tabPanel(title = "Take a Tour (video)",
                                            br(),
                                            bootstrapPage(htmlTemplate("home_page/video-template.html", name = "Video"))
                                            )
                                              ) # navlistPanel ends here
                                 )
                                )
                         ), # tabPanel of Home Page ends here
                
                navbarMenu(title = "Bloom",
                           tabPanel(title = "Median Day of Year"),
                           tabPanel(title = "Difference from Historical")),
                
                navbarMenu(title = "CM Flight",
                           tabPanel(title = "Median Day of Year (1st Flight)"),
                           tabPanel(title = "Difference from Historical (1st Flight)"),
                           tabPanel(title = "Median Day of Year (By Generation)"),
                           tabPanel(title = "Difference from Historical (By Generation)")
                           ),
                
                navbarMenu(title = "CM Egg Hatch",
                           tabPanel(title = "Pest Risk"),
                           tabPanel(title = "Median Day of Year (By Generation)"),
                           tabPanel(title = "Difference from Historical (By Generation)")
                           ),
                
                tabPanel(title = "CM Diapause"),
#                tabPanel(title = "Regional",
#                  fixedPage(
#                          fixedRow(
#                            column(7,
#                              "sidebar",
#                              navlistPanel(
#                                                      tabPanel("Regional Groups"),
#                                                      "rcp 4.5",
#                                                      tabPanel("Bloom"),
#                                                      tabPanel("Degree Days"),
#                                                      tabPanel("Adult Flight"),
#                                                      tabPanel("Diapause"),
#                                                      tabPanel("Egg Hatch into Larva"),
#                                                      "rcp 8.5",
#                                                      tabPanel("Bloom"),
#                                                      tabPanel("Degree Days"),
#                                                      tabPanel("Adult Flight"),
#                                                      tabPanel("Diapause"),
#                                                      tabPanel("Egg Hatch into Larva")
#                                                    )
#                            ),column(5,
#                              "main"
#                           )
#                          )
#                            )),
                tabPanel(title = "Regional Plots",
                  fluidPage(theme = "slate.css",
                    column(6, offset = 0, 
                            navlistPanel(
                              tabPanel("Location Groups", imageOutput("location_group")),

                              "rcp 4.5",
                              tabPanel("Bloom", imageOutput("bloom_45")),
                              tabPanel("Degree Days", imageOutput("cumdd_45")),
                              tabPanel("Adult Flight",
                                tabsetPanel(tabPanel("Emergence", imageOutput("emergence_rcp45")),
                                            tabPanel("Adult Flight Day of Year"),
                                            tabPanel("Number of Generations", imageOutput("Adult_Gen_Aug23_rcp45"))
                                            )
                                      , style='width: 1350px; height: 930px'),
                              tabPanel("Diapause",
                                tabsetPanel(tabPanel("Absolute Population", imageOutput("diapause_abs_rcp45")),
                                            tabPanel("Relative Population", imageOutput("diapause_rel_rcp45"))
                                            )
                                      , style='width: 1500px; height: 1500px'),
                              tabPanel("Egg Hatch into Larva",
                                tabsetPanel(tabPanel("Cumulative Larva Population Fraction", imageOutput("eggHatch_45")),
                                            tabPanel("Egg Hatch Day of Year"),
                                            tabPanel("Number of Generations", imageOutput("Larva_Gen_Aug23_rcp45"))
                                            )
                                      , style='width: 1000px; height: 1000px'
                                      ),

                              "rcp 8.5",
                              tabPanel("Bloom", imageOutput("bloom_85")),
                              tabPanel("Degree Days", imageOutput("cumdd_85")),
                              tabPanel("Adult Flight", 
                                tabsetPanel(tabPanel("Emergence", imageOutput("emergence_rcp85")),
                                            tabPanel("Adult Flight Day of Year"),
                                            tabPanel("Number of Generations", imageOutput("Adult_Gen_Aug23_rcp85"))
                                            )
                                      , style='width: 1350px; height: 930px'),
                              tabPanel("Diapause",
                                tabsetPanel(tabPanel("Absolute Population", imageOutput("diapause_abs_rcp85")),
                                            tabPanel("Relative Population", imageOutput("diapause_rel_rcp85"))
                                            )
                                      , style='width: 1500px; height: 1500px'),
                              tabPanel("Egg Hatch into Larva",
                                tabsetPanel(tabPanel("Cumulative Larva Population Fraction", imageOutput("eggHatch_85")),
                                            tabPanel("Egg Hatch Day of Year"),
                                            tabPanel("Number of Generations", imageOutput("Larva_Gen_Aug23_rcp85"))
                                           )
                                      , style='width: 1000px; height: 1000px' 
                                      )
                            ))
                          )
                        )
                
#                tabPanel(title = "Test",
#                         # CSS options so far:
#                         # bootstrap.css, paper.css, slate.css, superhero.css
#                         #
#                         fluidPage(theme = "slate.css",
#                           titlePanel("Tabs!"),
#                           sidebarLayout(
#                             sidebarPanel(
#                               textInput(inputId = "box_1", label = "Enter Tab 1 Text:", value="Tab 1!"),
#                               textInput(inputId = "box_2", label = "Enter Tab 2 Text:", value="Tab 2!"),
#                               textInput(inputId = "box_3", label = "Enter Tab 3 Text:", value="Tab 3!")
#                                          ),
#                             
#                             mainPanel(
#                               tabsetPanel(type="tabs",
#                                           tabPanel(title = "Tab 1", br(), textOutput("out_1")),
#                                           tabPanel(title = "Tab 2", br(), textOutput("out_2")),
#                                           tabPanel(title = "Tab 3", br(), textOutput("out_3"))
#                                           )
#                                       )
#                                        )
#                                     ) 
#                         ),
#                
#                ## Leaflet test tab
#                tabPanel(title = "Leaflet",
#                         fluidPage(leafletOutput("mymap"), 
#                                   p(), 
#                                   actionButton("recalc", "New points")
#                                   )
#                         )


                ) # navbarPage Ends here




