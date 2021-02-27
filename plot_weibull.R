param_dir = "/Users/hn/Documents/GitHub/Kirti/Codling_moth_Code/parameters/"
param_name = "CodlingMothparameters.txt"
params = read.table (paste0(param_dir, param_name), header=TRUE, sep=",")

colors_1 = c("black", "blue", "red", "gray")

for (i in 1:16){
  shap = params[i, 3]
  scal = params[i, 4]
  if (i %in% 1:4)  {
    colors = "blue"
    }
  else if (i %in% 5:8) {
    colors = "black"
  } 
  else if  (i %in% 9:12){
    colors = "red"
  }
  else {colors = "green"}
    
  curve(dweibull(x, shape=shap, scale=scal), xlim = c(-100, 6000), 
                                             ylim = c(0, .004), 
                                             col=colors, 
                                             lty=1,
                                             add=TRUE)
  
}


for (i in 1:16){
  shap = params[i, 3]
  scal = params[i, 4] * (1.2)
  if (i %in% 1:4)  {
    colors = "blue"
  }
  else if (i %in% 5:8) {
    colors = "black"
  } 
  else if  (i %in% 9:12){
    colors = "red"
  }
  else {colors = "green"}
  
  curve(dweibull(x, shape=shap, scale=scal), xlim = c(-100, 6000), 
        ylim = c(0, .004), 
        col=colors, 
        lty=2,
        add=TRUE)
  
}