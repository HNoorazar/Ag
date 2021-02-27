navbarPage(title = div( 
			"",#tags$b(" Codling Moth"),
			img(src='csanr_logo.png', style='width:100px;height:35px;'), 
			#img(src='DAS-Icon.png', style='width:60px;height:35px;'),
			img(src='WSU-DAS-log.png', style='width:100px;height:35px;'),
			img(src='NW-Climate-Hub-Logo.jpg', style='width:100px;height:35px;')
		   ), id="nav", windowTitle = "Codling Moth",
		#title = div(img(src='csanr_logo.png', style='width:60px;height:30px;'), tags$b(" Codling Moth")), id="nav", windowTitle = "Codling Moth",
           #selected = "Median Day of Year",
	   tabPanel(tags$b("Home"),
		navlistPanel(
			tabPanel(tags$b("About"), tags$div(style="width:950px", includeHTML("home-page/about.html"))),
			tabPanel(tags$b("People"), tags$div(style="width:950px", includeHTML("home-page/people.html"))),
			tabPanel(tags$b("Codling Moth Life Cycle and Management"), tags$div(style = "width: 950px", includeHTML("home-page/life-cycle.html"))),
			tabPanel(tags$b("Climate Data"), tags$div(style="width:950px", includeHTML("home-page/climate-change-projections.html"))),
			tabPanel(tags$b("What's the story?"), tags$div(style="width: 950px", includeHTML("home-page/changing-pest-pressures.html"))),
			tabPanel(tags$b("Contact"), tags$div(style="width:950px", includeHTML("home-page/contact.html"))),
			tabPanel(tags$b("Take a tour! (video)"), tags$div(style="width:950px", includeHTML("home-page/take-a-tour.html"))),
			widths = c(2,10)
		)
	   ),
           navbarMenu(tags$b("Bloom"),
           #tabPanel(tags$b("Full Bloom"),
                      tabPanel("Median Day of Year", 
                               div(class="outer",
                                   tags$head(
                                     # Include our custom CSS
                                     includeCSS("styles.css")
                                     #includeScript("gomap.js")
                                   ),
                                   leafletOutput("map_bloom_doy", width="100%", height="100%"),
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 
                                                 h3(tags$b("Explorer")),
				   		 gsub("cg", "cg_bloom", includeHTML("explorer_climate_group.html")),
				      		 #includeHTML("explorer_climate_group.html"),
                                                 #selectInput("fver_bloom", label = h5("Select Future Data Version"),
						 #	     choices = list("rcp45" = "rcp45", "rcp85" = "rcp85"),
                                                 #           selected = "rcp85"),	
						 selectInput("apple_type", 
						 	         label = h4(tags$b("Select Apple Variety")),
                              		 choices = list( "Cripps Pink" = "cripps_pink",
								 	                 "Gala" = "gala", 
               							             "Red Delicious" = "red_deli"),
									 selected = "cripps_pink")))),
                      tabPanel("Difference from Historical", 
                               div(class="outer",
                                   tags$head(
                                     # Include our custom CSS
                                     includeCSS("styles.css")
                                     #includeScript("gomap.js")
                                   ),
                                   leafletOutput("map_bloom_diff", width="100%", height="100%"),
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 
                                                 h3(tags$b("Explorer")),
				   		 gsub("cg", "cg_bloom_diff", includeHTML("explorer_climate_group_diff.html")),
				      		 #includeHTML("explorer_climate_group.html"),
                                                 #selectInput("fver_bloom", label = h5("Select Future Data Version"),
						 #	     choices = list("rcp45" = "rcp45", "rcp85" = "rcp85"),
                                                 #           selected = "rcp85"),	
						 selectInput("apple_type_diff", label = h4(tags$b("Select Apple Variety")),
                              		                     choices = list( "Cripps Pink" = "cripps_pink",
								 	     "Gala" = "gala", 
               							             "Red Delicious" = "red_deli"),
									     selected = "cripps_pink"))))
                      ),
           navbarMenu(tags$b("CM Flight"),
                      tabPanel("Median Day of Year (First Flight)", 
                               div(class="outer",
                                   tags$head(
                                     # Include our custom CSS
                                     includeCSS("styles.css")
                          	     #includeScript("script.js")
                                     #includeScript("gomap.js")
                                   ),
                                   leafletOutput("map_emerg_doy", width="100%", height="100%"),
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 #draggable = TRUE, top = 120, left = "auto", right = 170, bottom = "auto",
                                                 draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 
                                                 h3(tags$b("Explorer")),
				      		 gsub("cg", "cg_emerg_doy", includeHTML("explorer_climate_group.html")))
				      		 #includeHTML("explorer_climate_group.html"))
                                                 #selectInput("fver_emerg_doy", label = h5("Select Future Data Version"),
						 #	     choices = list("rcp45" = "rcp45", "rcp85" = "rcp85"),
                                                 #            selected = "rcp45"))	
				)),
                      tabPanel("Difference from Historical (First Flight)",
                               div(class="outer",
                                   tags$head(
                                     # Include our custom CSS
                                     includeCSS("styles.css")
                          	     #includeScript("script.js")
                                     #includeScript("gomap.js")
                                   ),
                                   leafletOutput("map_diff_emerg", width="100%", height="100%"),
                                   
                                   # Shiny versions prior to 0.11 should use class="modal" instead.
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 #draggable = TRUE, top = 60, left = "auto", right = 260, bottom = "auto",
                                                 draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 
                                                 h3(tags$b("Explorer")),
				      		 gsub("cg", "cg_diff_emerg", includeHTML("explorer_climate_group_diff.html"))
                                                 #selectInput("fver_emerg_diff", label = h5("Select Future Data Version"),
						 #	     choices = list("rcp45" = "rcp45", "rcp85" = "rcp85"),
                                                 #            selected = "rcp45"),
                                                 #selectInput("emerg_diff_type", label = h5("Select Difference Quantile"),
						 #	     choices = list("Median" = 3),
                                                             #choices = list("Minimum" = 1, "First Quantile" = 2, 
                                                             #               "Median" = 3, "Third Quantile" = 4,
                                                             #               "Maximum" = 5),
                                                 #            selected = 3)
					)) ),
		   tabPanel("Median Day Of Year (By Generation)", 
				div(class="outer",
				  tags$head(
				  # Include our custom CSS
				  includeCSS("styles.css")
				  #includeScript("gomap.js")
				  ),
				  leafletOutput("map_adult_med_doy", width="100%", height="100%"),
			    
				  # Shiny versions prior to 0.11 should use class="modal" instead.
				  absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
					      #draggable = TRUE, top = 60, left = "auto", right = 170, bottom = "auto",
					      draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
					      width = 250, height = "auto",
					      
					      h3(tags$b("Explorer")),
					      gsub("cg", "cg_adult_med_doy", includeHTML("explorer_climate_group.html")),
				      	      #includeHTML("explorer_climate_group.html"),
                                              #selectInput("fver_med_doy", label = h5("Select Future Data Version"),
							     #choices = list("rcp45" = "rcp45", "rcp85" = "rcp85"),
                                                             #selected = "rcp45"),	
					      #radioButtons("adult_type", label = h5("Select Growth Stage"),
						#	   choices = list("Adult Flight" = "A",
						#			  "Egg Hatch into Larvae" = "L"),
						#	   selected = "A", inline = FALSE),
					      selectInput("adult_gen", label = h4(tags$b("Select Generation")),
							  choices = list("Generation 1" = "Gen1", "Generation 2" = "Gen2",
									 "Generation 3" = "Gen3", "Generation 4" = "Gen4"),
							  selected = "Gen1"),
					      radioButtons("adult_percent", label = h4(tags$b("Select % Population that has completed the growth stage")), #h5(textOutput("growthPercent")), #h5("Select Cumulative Percentage"),
							   choices = list("25 %" = "0.25", "50 %" = "0.5", "75 %" = "0.75"),
							   selected = "0.25", inline = TRUE)))), 
                      tabPanel("Difference from Historical (By Generation)",
                               div(class="outer",
                                   tags$head(
                                     # Include our custom CSS
                                     includeCSS("styles.css")
                                     #includeScript("gomap.js")
                                   ),
                                   leafletOutput("map_adult_diff_doy", width="100%", height="100%"),
                                   
                                   # Shiny versions prior to 0.11 should use class="modal" instead.
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 
                                                 h3(tags$b("Explorer")),
				      		 gsub("cg", "cg_adult_doy_diff", includeHTML("explorer_climate_group_diff.html")),
                                                 #selectInput("fver_doy_diff", label = h5("Select Future Data Version"),
						 #	     choices = list("rcp45" = "rcp45", "rcp85" = "rcp85"),
                                                 #            selected = "rcp45"),	
                                                 #radioButtons("type_diff", label = h5("Select Growth Stage"),
                                                 #             choices = list("Adult Flight" = "A",
                                                 #                            "Egg Hatch into Larvae" = "L"),
                                                 #             selected = "A", inline = FALSE),
                                                 #selectInput("diff_type", label = h5("Select Difference Quantile"),
						 #	     choices = list("Median" = 3),
                                                             #choices = list("Minimum" = 1, "First Quantile" = 2, 
                                                             #               "Median" = 3, "Third Quantile" = 4,
                                                             #               "Maximum" = 5),
                                                 #            selected = 1),
                                                 selectInput("adult_gen_diff", label = h4(tags$b("Select Generation")),
                                                             choices = list("Generation 1" = "Gen1", "Generation 2" = "Gen2", 
                                                                            "Generation 3" = "Gen3", "Generation 4" = "Gen4"),
                                                             selected = "Gen1"),
                                                 radioButtons("adult_percent_diff", label = h4(tags$b("Select % Population that has completed the growth stage")), #h5(textOutput("growthPercentDiff")), #h5("Select Cumulative Percentage"),
                                                              choices = list("25 %" = "0.25", "50 %" = "0.5", "75 %" = "0.75"),
                                                              selected = "0.25", inline = TRUE))))
			),
           navbarMenu(tags$b("CM Egg Hatch"),
		   tabPanel("Pest Risk",
			    #tags$b("Pest Risk", id="pest_risk"),
			    #bsTooltip(id="pest_risk", title="something to add", placement = "bottom", trigger = "hover",options = NULL),
			    #bsPopover(id="pest_risk", title="", content="something to add", placement = "bottom", trigger = "hover",options = NULL),
			    div(class="outer",
				tags$head(
				  # Include our custom CSS
				  includeCSS("styles.css")
				  #includeScript("script.js")
				  #includeScript("gomap.js")
				),
				leafletOutput("map_risk", width="100%", height="100%"),
				
				# Shiny versions prior to 0.11 should use class="modal" instead.
				absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
					      draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
					      #draggable = TRUE, top = 60, left = "auto", right = 170, bottom = "auto",
					      width = 250, height = "auto",
					      
					      h3(tags$b("Explorer")),
					      gsub("cg", "cg_risk", includeHTML("explorer_climate_group.html")),
					      #selectInput("fver_risk", label = h5("Select Future Data Version"),
							   #choices = list("rcp45" = "rcp45", "rcp85" = "rcp85"),
							   #selected = "rcp45"),
					      #radioButtons("type_risk", label = h5(" "), #Select Adult/Larvae"),
					      #             choices = list(#"Adult Generation" = "A",
					      #                            "Larvae Generation" = "L"),
					      #             selected = "L", inline = FALSE),
					      selectInput("gen_risk", label = h4(tags$b("Select Generation")),
							  choices = list("Generation 3" = "Gen3", "Generation 4" = "Gen4"),
							  selected = "Gen3"),
					      radioButtons("percent_risk", label = h4(tags$b("Select Proportion of Eggs Hatched")), #Cumulative Percentage"),
							   choices = list("25 %" = "0.25", "50 %" = "0.5", "75 %" = "0.75"),
							   selected = "0.75", inline = TRUE)))),
		   tabPanel("Median Day Of Year (By Generation)", 
				div(class="outer",
				  tags$head(
				  # Include our custom CSS
				  includeCSS("styles.css")
				  #includeScript("gomap.js")
				  ),
				  leafletOutput("map_larvae_med_doy", width="100%", height="100%"),
			    
				  # Shiny versions prior to 0.11 should use class="modal" instead.
				  absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
					      #draggable = TRUE, top = 60, left = "auto", right = 170, bottom = "auto",
					      draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
					      width = 250, height = "auto",
					      
					      h3(tags$b("Explorer")),
					      gsub("cg", "cg_larvae_med_doy", includeHTML("explorer_climate_group.html")),
				      	      #includeHTML("explorer_climate_group.html"),
                                              #selectInput("fver_med_doy", label = h5("Select Future Data Version"),
							     #choices = list("rcp45" = "rcp45", "rcp85" = "rcp85"),
                                                             #selected = "rcp45"),	
					      #radioButtons("type", label = h5("Select Growth Stage"),
						#	   choices = list("Adult Flight" = "A",
						#			  "Egg Hatch into Larvae" = "L"),
						#	   selected = "A", inline = FALSE),
					      selectInput("larvae_gen", label = h4(tags$b("Select Generation")),
							  choices = list("Generation 1" = "Gen1", "Generation 2" = "Gen2",
									 "Generation 3" = "Gen3", "Generation 4" = "Gen4"),
							  selected = "Gen1"),
					      radioButtons("larvae_percent", label = h4(tags$b("Select Proportion of Eggs hatched")), #h5(textOutput("growthPercent")), #h5("Select Cumulative Percentage"),
							   choices = list("25 %" = "0.25", "50 %" = "0.5", "75 %" = "0.75"),
							   selected = "0.25", inline = TRUE)))),
             
                      tabPanel("Difference from Historical (By Generation)",
                               div(class="outer",
                                   tags$head(
                                     # Include our custom CSS
                                     includeCSS("styles.css")
                                     #includeScript("gomap.js")
                                   ),
                                   leafletOutput("map_larvae_diff_doy", width="100%", height="100%"),
                                   
                                   # Shiny versions prior to 0.11 should use class="modal" instead.
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 
                                                 h3(tags$b("Explorer")),
				      		 gsub("cg", "cg_larvae_doy_diff", includeHTML("explorer_climate_group_diff.html")),
                                                 #selectInput("fver_doy_diff", label = h5("Select Future Data Version"),
						 #	     choices = list("rcp45" = "rcp45", "rcp85" = "rcp85"),
                                                 #            selected = "rcp45"),	
                                                 #radioButtons("type_diff", label = h5("Select Growth Stage"),
                                                 #             choices = list("Adult Flight" = "A",
                                                 #                            "Egg Hatch into Larvae" = "L"),
                                                 #             selected = "A", inline = FALSE),
                                                 #selectInput("diff_type", label = h5("Select Difference Quantile"),
						 #	     choices = list("Median" = 3),
                                                             #choices = list("Minimum" = 1, "First Quantile" = 2, 
                                                             #               "Median" = 3, "Third Quantile" = 4,
                                                             #               "Maximum" = 5),
                                                 #            selected = 1),
                                                 selectInput("larvae_gen_diff", label = h4(tags$b("Select Generation")),
                                                             choices = list("Generation 1" = "Gen1", "Generation 2" = "Gen2", 
                                                                            "Generation 3" = "Gen3", "Generation 4" = "Gen4"),
                                                             selected = "Gen1"),
                                                 radioButtons("larvae_percent_diff", label = h4(tags$b("Select Proportion of Eggs hatched")), #h5(textOutput("growthPercentDiff")), #h5("Select Cumulative Percentage"),
                                                              choices = list("25 %" = "0.25", "50 %" = "0.5", "75 %" = "0.75"),
                                                              selected = "0.25", inline = TRUE))))
			),

#	  tabPanel("Risk Map",
#                    div(class="outer",
#                        tags$head(
#                          # Include our custom CSS
#                          includeCSS("styles.css")
#                          #includeScript("gomap.js")
#                        ),
#                        leafletOutput("map_risk1", width="100%", height="100%"),
#                        
#                        # Shiny versions prior to 0.11 should use class="modal" instead.
#                        absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
#                                      draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
#                                      width = 250, height = "auto",
#                                      
#                                      h2("Explorer"),
#                                      radioButtons("type_risk1", label = h5("Select Adult/Larvae"),
#                                                   choices = list("Adult Generation" = "Adult",
#                                                                  "Larvae Generation" = "Larva"),
#                                                   selected = "Adult", inline = FALSE),
#                                      selectInput("gen_risk1", label = h5("Select Generation"),
#                                                  choices = list("Generation 3" = "Gen3", "Generation 4" = "Gen4"),
#                                                  selected = "Gen3"),
#                                      selectInput("clim_scen", label = h5("Select Climate Scenario"),
#                                                  choices = list("historical" = "historical", "BNU-ESM" = "BNU-ESM", "CanESM2" = "CanESM2", 
#                                                                 "GFDL-ESM2G" = "GFDL-ESM2G",	"bcc-csm1-1-m" = "bcc-csm1-1-m", "CNRM-CM5" = "CNRM-CM5",
#                                                                 "GFDL-ESM2M" = "GFDL-ESM2M"),
#                                                  selected = "GFDL-ESM2M"),
#                                      radioButtons("percent_risk1", label = h5("Select Percentage"),
#                                                   choices = list("25 %" = "0.25", "50 %" = "0.5", "75 %" = "0.75"),
#                                                   selected = "0.25", inline = TRUE)))), 
           
           #navbarMenu(tags$b("Adult First Flight  Day of Year")
           #           ),
           
#           tabPanel("Median Population", 
#                    div(class="outer",
#                        tags$head(
#                          # Include our custom CSS
#                          includeCSS("styles.css")
#                          #includeScript("gomap.js")
#                        ),
#                        leafletOutput("map_med_pop", width="100%", height="100%"),
#                        
#                        # Shiny versions prior to 0.11 should use class="modal" instead.
#                        absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
#                                      draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
#                                      width = 250, height = "auto",
#                                      
#                                      h2("Explorer"),
#                                      radioButtons("type_pop", label = h5("Select Adult/Larvae"),
#                                                   choices = list("Adult Population" = "Adult",
#                                                                  "Larvae Population" = "Larva"),
#                                                   selected = "Adult", inline = FALSE),
#                                      selectInput("type_pop_gen", label = h5("Select Generation"),
#                                                  choices = list("Generation 1" = "Gen1", 
#                                                                 "Generation 2" = "Gen2",
#                                                                 "Generation 3" = "Gen3", 
#                                                                 "Generation 4" = "Gen4"),
#                                                  selected = "Gen1"),
#                                      #selectInput("pop_month", label = h5("Population at the end of (Month)"),
#                                      #            choices = list("January" = "Feb1", "February" = "Mar1", "March" = "Apr1",
#                                      #                           "April" = "May1", "May" = "June1", "June" = "Jul1",
#                                      #                           "July" = "Aug1", "August" = "Sep1", "September" = "Oct1",
#                                      #                           "October" = "Nov1", "November" = "Dec1"),
#                                      #            selected = "June1"),
#				      selectInput("pop_month", label = h5("Population at the end of (Month)"),
#                                                  choices = list("January" = "January", "February" = "February", "March" = "March",
#                                                                 "April" = "April", "May" = "May", "June" = "June",
#                                                                 "July" = "July", "August" = "August", "September" = "September",
#                                                                 "October" = "October", "November" = "November", "December" = "December"),
#                                                  selected = "May")))),
           
           #navbarMenu(tags$b("Growth Stage Day of Year")
           #           ),
           
#                      tabPanel("Population",
#                               div(class="outer",
#                                   tags$head(
#                                     # Include our custom CSS
#                                     includeCSS("styles.css")
#                                     #includeScript("gomap.js")
#                                   ),
#                                   leafletOutput("map_diff_pop", width="100%", height="100%"),
#                                   
#                                   # Shiny versions prior to 0.11 should use class="modal" instead.
#                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
#                                                 draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
#                                                 width = 250, height = "auto",
#                                                 
#                                                 h2("Explorer"),
#                                                 radioButtons("type_pop_diff", label = h5("Select Adult/Larvae"),
#                                                              choices = list("Adult Population" = "Adult",
#                                                                             "Larvae Population" = "Larva"),
#                                                              selected = "Adult", inline = FALSE),
#                                                 selectInput("pop_diff_type", label = h5("Select Difference Quantile"),
#							     choices = list("Median" = 3),
#                                                             #choices = list("Minimum" = 1, "First Quantile" = 2, 
#                                                             #               "Median" = 3, "Third Quantile" = 4,
#                                                             #               "Maximum" = 5),
#                                                             selected = 3),
#						 selectInput("pop_diff_type_gen", label = h5("Select Generation"),
#                              		                     choices = list("Generation 1" = "Gen1", 
#               							        "Generation 2" = "Gen2",
#									"Generation 3" = "Gen3", 
#									"Generation 4" = "Gen4"),
#									selected = "Gen1"),
#                                                 #selectInput("pop_diff_month", label = h5("Population at the end of (Month)"),
#                                                 #            choices = list("January" = "Feb1", "February" = "Mar1", "March" = "Apr1",
#                                                 #                           "April" = "May1", "May" = "June1", "June" = "Jul1",
#                                                 #                           "July" = "Aug1", "August" = "Sep1", "September" = "Oct1",
#                                                 #                           "October" = "Nov1", "November" = "Dec1"),
#                                                 #            selected = "June1"))))
#						 selectInput("pop_diff_month", label = h5("Population at the end of (Month)"),
#							  choices = list("January" = "January", "February" = "February", "March" = "March",
#									 "April" = "April", "May" = "May", "June" = "June",
#									 "July" = "July", "August" = "August", "September" = "September",
#									 "October" = "October", "November" = "November", "December" = "December"),
#							  selected = "May"))))
           #navbarMenu(tags$b("Diapause"),
           tabPanel(tags$b("CM Diapause"),
                      #tabPanel("Mean Population", 
                               div(class="outer",
                                   tags$head(
                                     # Include our custom CSS
                                     includeCSS("styles.css")
                                     #includeScript("gomap.js")
                                   ),
                                   leafletOutput("map_diap_pop", width="100%", height="100%"),
                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
                                                 draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
                                                 width = 250, height = "auto",
                                                 
                                                 h3(tags$b("Explorer")),
				      		 gsub("cg", "cg_diap", includeHTML("explorer_climate_group.html")),
				      		 #includeHTML("explorer_climate_group.html"),
                                                 #selectInput("fver_diap", label = h5("Select Future Data Version"),
						 #	     choices = list("rcp45" = "rcp45", "rcp85" = "rcp85"),
                                                 #            selected = "rcp45"),	
                                                 #radioButtons("diap_pop", label = h5("Select Population"),
                                                 #             choices = list("Absolute Percentage" = "AbsPct",
                                                 #                            "Relative Percentage" = "RelPct"),
                                                 #             selected = "RelPct", inline = FALSE),
                                                 radioButtons("diapaused", label = h4(tags$b("Select Diapause/Non-Diapause")),
							     choices = list("Diapause Escaped" = "NonDiap",
									    "Diapause Induced" = "Diap"),
                                                             selected = "NonDiap", inline = FALSE),
						 selectInput("diap_gen", label = h4(tags$b("Select Generation")),
                              		                     choices = list(
									"Generation 1" = "Gen1", 
               							        "Generation 2" = "Gen2",
									"Generation 3" = "Gen3", 
									"Generation 4" = "Gen4",
									"All" = "all"),
									selected = "Gen1")))),
#                      tabPanel("Difference from Historical",
#                               div(class="outer",
#                                   tags$head(
#                                     # Include our custom CSS
#                                     includeCSS("styles.css")
#                                     #includeScript("gomap.js")
#                                   ),
#                                   leafletOutput("map_diff_diap", width="100%", height="100%"),
#                                   
#                                   # Shiny versions prior to 0.11 should use class="modal" instead.
#                                   absolutePanel(id = "controls", class = "panel panel-default", fixed = TRUE,
#                                                 draggable = TRUE, top = 60, left = "auto", right = 20, bottom = "auto",
#                                                 width = 250, height = "auto",
#                                                 
#                                                 h2("Explorer"),
#                                                 selectInput("diap_diff_type", label = h5("Select Difference Quantile"),
#							     choices = list("Median" = 3),
#                                                             #choices = list("Minimum" = 1, "First Quantile" = 2, 
#                                                             #               "Median" = 3, "Third Quantile" = 4,
#                                                             #               "Maximum" = 5),
#                                                             selected = 3))) )
                     # ),
#           navbarMenu("Plots",
#                      tabPanel("Adult Generation",
#                        fluidRow(tabBox("Adult Generation",
#                          tabPanel("Time Frame Vs Day of Year", imageOutput("ag_vplot"), height = 900),
#                          tabPanel("Time Frame Vs Percentage of Years", imageOutput("ag_bplot"), height = 900)
#                        ))),
#                      tabPanel("Larvae Generation",
#                        fluidRow(tabBox("Larvae Generation",
#                          tabPanel("Time Frame Vs Day of Year", imageOutput("lg_vplot"), height = 900),
#                          tabPanel("Time Frame Vs Percentage of Years", imageOutput("lg_bplot"), height = 900)
#                        ))),
#                      #tabPanel("Adult Population",
#                      #  fluidRow(tabBox("Adult Population",
#                      #    tabPanel("Time Frame Vs Population(%)", imageOutput("ap_vplot"), height = 900),
#                      #    tabPanel("Time Frame Vs Percentage of Years", imageOutput("ap_bplot"), height = 900)
#                      #  ))),
#                      #tabPanel("Larvae Population",
#                      #  fluidRow(tabBox("Larvae Population",
#                      #    tabPanel("Time Frame Vs Population(%)", imageOutput("lp_vplot"), height = 900),
#                      #    tabPanel("Time Frame Vs Percentage of Years", imageOutput("lp_bplot"), height = 900)
#                      #  ))),
#		      #tabPanel("Adult/Larvae Population",
#				#imageOutput("gen_pop_plot"), height = 900),
#		      #tabPanel("Adult/Larvae Population - GFDL-ESM2M",
#			#	imageOutput("gen_pop_plot1"), height = 900),
#                      tabPanel("Cumulative Larva Population Fraction", imageOutput("cum_larva_pop")),
#                      tabPanel("Emergence Day of Year",
#                               imageOutput("e_vplot"), height = 700),
#                      #tabPanel("Diapause Population",
#                      #         imageOutput("d_vplot"), height = 700)
#		      tabPanel("Degree Days",
#				fluidRow(
#				tabBox(
#					tabPanel("Cumulative Degree Days", imageOutput("cumdd")),
#					tabPanel("CumDD Magnitude Difference - Month Groups", imageOutput("cumDD_mongrps_magdiff")),
#					tabPanel("CumDD Magnitude Difference - Months", imageOutput("cumDD_mons_magdiff")),
#					tabPanel("DD Magnitude Difference - Month Groups", imageOutput("DD_mongrps_magdiff1")),
#					tabPanel("DD - Month Groups", imageOutput("DD_mongrps")),
#					width = 12
#				))),
#		      tabPanel("Diapause",
#				fluidRow(
#				tabBox(
#					#side="right", height = "1000px",
#					tabPanel("Relative Population Vs Cumulative DD", imageOutput("rel_pop_cumdd")), #imageOutput("county_groups")),
#					tabPanel("Relative Population Vs DOY", imageOutput("rel_pop_doy")),
#					tabPanel("Absolute Population Vs Cumulative DD", imageOutput("abs_pop_cumdd")),
#					tabPanel("Absolute Population Vs DOY", imageOutput("abs_pop_doy"))
#				))),
#                      tabPanel("Full Bloom", imageOutput("full_bloom"))
#			),
	tabPanel(tags$b("Regional Plots"),
		 navlistPanel(
			tabPanel("Location Groups", imageOutput("location_group")),
			HTML("<b>Rcp 8.5</b>"),
			tabPanel("Bloom", imageOutput("full_bloom")),
			tabPanel("Degree Days", imageOutput("cumdd")),
			#tabPanel("Figure 1", h5("Figure 1")),
			tabPanel("Adult Flight",
			  fluidRow(tabBox(
			    tabPanel("Adult Flight", imageOutput("e_vplot"), height = 700),
			    tabPanel("Adult Flight Day of Year", imageOutput("ag_vplot"), height = 900),
			    tabPanel("Number of Generations", imageOutput("ag_bplot"), height = 900),
				width = 12
			))),
			tabPanel("Egg Hatch into Larva",
			  fluidRow(tabBox(
			    tabPanel("Cumulative Larva Population Fraction", imageOutput("cum_larva_pop")),
			    tabPanel("Egg Hatch Day of Year", imageOutput("lg_vplot"), height = 900),
			    tabPanel("Number of Generations", imageOutput("lg_bplot"), height = 900),
				width = 12
			))),
				#fluidRow(
				#tabBox(
				#	tabPanel("Cumulative Degree Days", imageOutput("cumdd")),
					#tabPanel("CumDD Magnitude Difference - Month Groups", imageOutput("cumDD_mongrps_magdiff")),
					#tabPanel("CumDD Magnitude Difference - Months", imageOutput("cumDD_mons_magdiff")),
					#tabPanel("DD Magnitude Difference - Month Groups", imageOutput("DD_mongrps_magdiff1")),
					#tabPanel("DD - Month Groups", imageOutput("DD_mongrps")),
				#	width = 12
				#))),
			tabPanel("Diapause",
				fluidRow(
				tabBox(
					#side="right", height = "1000px",
					tabPanel("Relative Population Vs Cumulative DD", imageOutput("rel_pop_cumdd")), #imageOutput("county_groups")),
					#tabPanel("Relative Population Vs DOY", imageOutput("rel_pop_doy")),
					tabPanel("Absolute Population Vs Cumulative DD", imageOutput("abs_pop_cumdd")),
					#tabPanel("Absolute Population Vs DOY", imageOutput("abs_pop_doy")),
					width = 12
				))),
			HTML("<b>Rcp 4.5</b>"),
			#tabPanel("Figure 2", h5("Figure 2")),
			tabPanel("Bloom", imageOutput("full_bloom_rcp45")),
			tabPanel("Degree Days", imageOutput("cumdd_rcp45")),
			tabPanel("Adult Flight",
			  fluidRow(tabBox(
			    tabPanel("Adult Flight", imageOutput("e_vplot_rcp45"), height = 700),
			    tabPanel("Adult Flight Day of Year", imageOutput("ag_vplot_rcp45"), height = 900),
			    tabPanel("Number of Generations", imageOutput("ag_bplot_rcp45"), height = 900),
				width = 12
			))),
			tabPanel("Egg Hatch into Larva",
			  fluidRow(tabBox(
			    tabPanel("Cumulative Larva Population Fraction", imageOutput("cum_larva_pop_rcp45")),
			    tabPanel("Egg Hatch Day of Year", imageOutput("lg_vplot_rcp45"), height = 900),
			    tabPanel("Number of Generations", imageOutput("lg_bplot_rcp45"), height = 900),
				width = 12
			))),
				#fluidRow(
				#tabBox(
				#	tabPanel("Cumulative Degree Days", imageOutput("cumdd_rcp45")),
					#tabPanel("CumDD Magnitude Difference - Month Groups", imageOutput("cumDD_mongrps_magdiff_rcp45")),
					#tabPanel("CumDD Magnitude Difference - Months", imageOutput("cumDD_mons_magdiff_rcp45")),
					#tabPanel("DD Magnitude Difference - Month Groups", imageOutput("DD_mongrps_magdiff1_rcp45")),
					#tabPanel("DD - Month Groups", imageOutput("DD_mongrps_rcp45")),
				#	width = 12
				#))),
			tabPanel("Diapause",
				fluidRow(
				tabBox(
					#side="right", height = "1000px",
					tabPanel("Relative Population Vs Cumulative DD", imageOutput("rel_pop_cumdd_rcp45")), #imageOutput("county_groups")),
					#tabPanel("Relative Population Vs DOY", imageOutput("rel_pop_doy_rcp45")),
					tabPanel("Absolute Population Vs Cumulative DD", imageOutput("abs_pop_cumdd_rcp45")),
					#tabPanel("Absolute Population Vs DOY", imageOutput("abs_pop_doy_rcp45")),
					width = 12
				))),
			widths = c(2,10)
		))
)
