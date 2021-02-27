print("Hello World !!!")

M <- replicate(2, runif(10e5, 0, 1))
d <- data.frame(M)
colnames(d) <- c("y1", "y2")
head(d)
str(d)

library(tidyverse)

iris2 <- iris %>%
  as_tibble() %>%
  filter(Sepal.Length > 2.5) %>%
  mutate(Sepal.Area = Sepal.Width * Sepal.Length) %>%
  arrange(desc(Sepal.Area))
iris2
glimpse(iris2)

scatter <- ggplot(data = iris2, aes(x = Sepal.Length, y = Sepal.Width)) +
  geom_point(aes(color = Species, shape = Species)) +
  xlab("Sepal Length") + ylab("Sepal Width") +
  ggtitle("Sepal Length-Width")

dir.create(path = file.path("./img"), recursive = TRUE, showWarnings = FALSE)
file2save <- file.path("img", "ggplot_iris.png")
ggsave(
  filename = file2save,
  plot = scatter,
  type = "cairo",
  height = 6,
  width = 6,
  dpi = 150
)

library(MESS)
library(chillR)
library(hydroTSM)
library(hydroGOF)
library(EGRET)

devtools::session_info()

