rm(list=ls())
library(xlsx)
library( readxl )

merge_water_supply <- function(data_dir){
############################################################
########## This is not memory efficient, you can 
########## count number of rows needed and initialize
########## the data frame of proper size, to avoid
########## making copies of data tables!
############################################################
	# list of files to be merged (which ends in .xlsx here)
	file_list = list.files(path = data_dir, pattern = ".xlsx", 
	                       all.files = FALSE, 
	                       full.names = FALSE, 
	                       recursive = FALSE)

	# initialize the (empty) data table
	merged_data <- data.table( year   = character(),
		                       supply = numeric(),
		                       ENVFLW = numeric(),
		                       available = numeric(),
		                       state   = character(),
		                       district= character(),
		                       model   = character()
		                       )
	for (file in file_list){
		# extract the name of the state. Files are named
		# as "Compiled [Georgia] Water Supply.xlsx". 
		# The second word is the state name
		state_name = unlist(strsplit(file, split=" "))[2]
		
		# extract number of sheets in the xlsx
	    number_of_sheets = length( excel_sheets(paste0(data_dir, file) ))

	    # extract the names of the sheets
	    sheet_names = excel_sheets(paste0(data_dir, file))

	    for (sheet in sheet_names){
	    	district_code = unlist(strsplit(sheet, split="_"))[1]
	    	model_type = unlist(strsplit(sheet, split="_"))[2]

	    	# the default of R is to truncate numbers, change it to 20 digits(!)
	    	options(digits=20)
	    	current_file = read.xlsx(paste0(data_dir, file), sheetName = sheet, skipEmptyRows=TRUE)
	    	current_file = current_file[, 1:4]
	    	colnames(current_file) <- c("year", "supply", "ENVFLW", "available")
	    	current_file$state <- state_name
	    	current_file$district <- district_code
	    	current_file$model <- model_type
	    	merged_data = rbind(merged_data, current_file)
	    }
	}
	merged_data$state <- factor(merged_data$state)
	merged_data$district <- factor(merged_data$district)
	merged_data$model <- factor(merged_data$model)

	saveRDS(merged_data, paste0(data_dir, "compiled_water_supply.rds"))
	write.csv(merged_data, file = paste0(data_dir, "compiled_water_supply.csv"))
}

# the direcotry including the files to be merged:
data_dir = "/Users/hn/Documents/GitHub/Kirti/merge_irrigation_Monday/"
merge_water_supply(data_dir)
