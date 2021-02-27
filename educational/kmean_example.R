df <- scaled_clusters
mapgilbert <- get_map(location = c(lon = mean(df$lon), 
                      lat = mean(df$lat)), zoom = 4,
                      maptype = "satellite", scale = 2)

ggmap(df) + 
geom_point(data = df, 
           mapping = aes(x = long, y = lat, color = clusters))


ggplot(df, aes(long, lat, group = group)) + 
geom_polygon(data = cnty2, color="black", size=0.05, fill="lightgrey")+
geom_point(data = df, aes(long, lat, color=clusters))


ggplot() + 
geom_polygon(data = cnty2, aes(x=long, y=lat), color="black", size=0.05, fill="lightgrey") + 
geom_point(data = df, aes(x=long, y=lat, color=clusters))



ggplot() + 
geom_point(data = df, aes(x=long, y=lat, color=clusters)) +
coord_quickmap()


states <- map_data("state")
usamap <- ggplot(states, aes(long, lat, group = group)) +
          geom_polygon(fill = "white", colour = "black") +
          geom_point(data = df, aes(x=long, y=lat, color=clusters), inherit.aes = FALSE )

df$clusters <- factor(df$clusters)
gg <- ggplot() +
      geom_polygon(data=cnty2, 
                   aes(x=long, y=lat, group=group, fill=NA), 
                   color = "black", fill=NA, size=0.5) +
      geom_point(data=df, aes(x=long, y=lat, color=df$clusters)) +
      coord_map()





gg <- ggplot()+
      geom_map(data=cnty2, map=cnty2, 
               aes(x=long, y=lat, map_id=fips, group=group),
                   fill=NA, color="black")

