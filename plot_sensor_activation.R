# MEG activations

library(tidyverse)
library(ggplot2)
library(doParallel)

registerDoParallel(7)

##### Sensor names #####
sensor_names <- c('MEG0111', 'MEG0112', 'MEG0113', 'MEG0121', 'MEG0122', 'MEG0123', 'MEG0131', 'MEG0132', 'MEG0133', 
                  'MEG0141', 'MEG0142', 'MEG0143', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0221', 'MEG0222', 'MEG0223', 
                  'MEG0231', 'MEG0232', 'MEG0233', 'MEG0241', 'MEG0242', 'MEG0243', 'MEG0311', 'MEG0312', 'MEG0313', 
                  'MEG0321', 'MEG0322', 'MEG0323', 'MEG0331', 'MEG0332', 'MEG0333', 'MEG0341', 'MEG0342', 'MEG0343', 
                  'MEG0411', 'MEG0412', 'MEG0413', 'MEG0421', 'MEG0422', 'MEG0423', 'MEG0431', 'MEG0432', 'MEG0433', 
                  'MEG0441', 'MEG0442', 'MEG0443', 'MEG0511', 'MEG0512', 'MEG0513', 'MEG0521', 'MEG0522', 'MEG0523', 
                  'MEG0531', 'MEG0532', 'MEG0533', 'MEG0541', 'MEG0542', 'MEG0543', 'MEG0611', 'MEG0612', 'MEG0613', 
                  'MEG0621', 'MEG0622', 'MEG0623', 'MEG0631', 'MEG0632', 'MEG0633', 'MEG0641', 'MEG0642', 'MEG0643', 
                  'MEG0711', 'MEG0712', 'MEG0713', 'MEG0721', 'MEG0722', 'MEG0723', 'MEG0731', 'MEG0732', 'MEG0733', 
                  'MEG0741', 'MEG0742', 'MEG0743', 'MEG0811', 'MEG0812', 'MEG0813', 'MEG0821', 'MEG0822', 'MEG0823', 
                  'MEG0911', 'MEG0912', 'MEG0913', 'MEG0921', 'MEG0922', 'MEG0923', 'MEG0931', 'MEG0932', 'MEG0933', 
                  'MEG0941', 'MEG0942', 'MEG0943', 'MEG1011', 'MEG1012', 'MEG1013', 'MEG1021', 'MEG1022', 'MEG1023', 
                  'MEG1031', 'MEG1032', 'MEG1033', 'MEG1041', 'MEG1042', 'MEG1043', 'MEG1111', 'MEG1112', 'MEG1113', 
                  'MEG1121', 'MEG1122', 'MEG1123', 'MEG1131', 'MEG1132', 'MEG1133', 'MEG1141', 'MEG1142', 'MEG1143', 
                  'MEG1211', 'MEG1212', 'MEG1213', 'MEG1221', 'MEG1222', 'MEG1223', 'MEG1231', 'MEG1232', 'MEG1233', 
                  'MEG1241', 'MEG1242', 'MEG1243', 'MEG1311', 'MEG1312', 'MEG1313', 'MEG1321', 'MEG1322', 'MEG1323', 
                  'MEG1331', 'MEG1332', 'MEG1333', 'MEG1341', 'MEG1342', 'MEG1343', 'MEG1411', 'MEG1412', 'MEG1413', 
                  'MEG1421', 'MEG1422', 'MEG1423', 'MEG1431', 'MEG1432', 'MEG1433', 'MEG1441', 'MEG1442', 'MEG1443', 
                  'MEG1511', 'MEG1512', 'MEG1513', 'MEG1521', 'MEG1522', 'MEG1523', 'MEG1531', 'MEG1532', 'MEG1533', 
                  'MEG1541', 'MEG1542', 'MEG1543', 'MEG1611', 'MEG1612', 'MEG1613', 'MEG1621', 'MEG1622', 'MEG1623', 
                  'MEG1631', 'MEG1632', 'MEG1633', 'MEG1641', 'MEG1642', 'MEG1643', 'MEG1711', 'MEG1712', 'MEG1713', 
                  'MEG1721', 'MEG1722', 'MEG1723', 'MEG1731', 'MEG1732', 'MEG1733', 'MEG1741', 'MEG1742', 'MEG1743', 
                  'MEG1811', 'MEG1812', 'MEG1813', 'MEG1821', 'MEG1822', 'MEG1823', 'MEG1831', 'MEG1832', 'MEG1833', 
                  'MEG1841', 'MEG1842', 'MEG1843', 'MEG1911', 'MEG1912', 'MEG1913', 'MEG1921', 'MEG1922', 'MEG1923', 
                  'MEG1931', 'MEG1932', 'MEG1933', 'MEG1941', 'MEG1942', 'MEG1943', 'MEG2011', 'MEG2012', 'MEG2013', 
                  'MEG2021', 'MEG2022', 'MEG2023', 'MEG2031', 'MEG2032', 'MEG2033', 'MEG2041', 'MEG2042', 'MEG2043', 
                  'MEG2111', 'MEG2112', 'MEG2113', 'MEG2121', 'MEG2122', 'MEG2123', 'MEG2131', 'MEG2132', 'MEG2133', 
                  'MEG2141', 'MEG2142', 'MEG2143', 'MEG2211', 'MEG2212', 'MEG2213', 'MEG2221', 'MEG2222', 'MEG2223', 
                  'MEG2231', 'MEG2232', 'MEG2233', 'MEG2241', 'MEG2242', 'MEG2243', 'MEG2311', 'MEG2312', 'MEG2313', 
                  'MEG2321', 'MEG2322', 'MEG2323', 'MEG2331', 'MEG2332', 'MEG2333', 'MEG2341', 'MEG2342', 'MEG2343', 
                  'MEG2411', 'MEG2412', 'MEG2413', 'MEG2421', 'MEG2422', 'MEG2423', 'MEG2431', 'MEG2432', 'MEG2433', 
                  'MEG2441', 'MEG2442', 'MEG2443', 'MEG2511', 'MEG2512', 'MEG2513', 'MEG2521', 'MEG2522', 'MEG2523', 
                  'MEG2531', 'MEG2532', 'MEG2533', 'MEG2541', 'MEG2542', 'MEG2543', 'MEG2611', 'MEG2612', 'MEG2613', 
                  'MEG2621', 'MEG2622', 'MEG2623', 'MEG2631', 'MEG2632', 'MEG2633', 'MEG2641', 'MEG2642', 'MEG2643')



##### Read precomputed files #####

data_path <- "/Users/ludvigolsen/Documents/Programmering/PythonLudvig/Machine learning/MEG-fMRI-group2/data/"
subjects <- paste0("group_", c(1, 3:7))

# NOTE: Load the saved file instead of rerunning 
all_timepoints <- plyr::ldply(subjects, .parallel = TRUE, function(s){
  precomputed_path <- paste0(data_path, s, "/precomputed/")
  time_point_df_paths <- list.files(precomputed_path, pattern=".csv")
  plyr::ldply(time_point_df_paths, function(tp){
    read.csv(paste0(precomputed_path, tp), stringsAsFactors = FALSE)
  }) %>% dplyr::mutate(
    Subject = s
  )
}) %>% 
  dplyr::as_tibble()

all_timepoints["X"] <- NULL
colnames(all_timepoints) <- c(sensor_names, "Trial", "Time.Point", "Subject")
# write.csv(all_timepoints, paste0(data_path, "all_timepoints_all_groups.csv"))

# NOTE:
# The ones ending with the number 1 are magnetometers, the rest are gradiometers. 

info_cols <- all_timepoints %>% 
  dplyr::select(Trial, Time.Point, Subject)

magnetometers <- all_timepoints %>% 
  dplyr::select_if(grepl("1$",names(.)))
magnetometers_names <- colnames(magnetometers)
magnetometers <- magnetometers %>%
  dplyr::bind_cols(info_cols)

gradiometers <- all_timepoints %>% 
  dplyr::select_if(!grepl("1$",names(.))) %>% 
  dplyr::select(-c(Trial, Time.Point, Subject))
gradiometers_names <- colnames(gradiometers)
gradiometers <- gradiometers %>%
  dplyr::bind_cols(info_cols)

# First averaging

magnetometers_averages <- magnetometers %>% 
  dplyr::group_by(Subject, Time.Point) %>% 
  dplyr::summarise_at(magnetometers_names, 
                      .f = mean)

gradiometers_averages <- gradiometers %>% 
  dplyr::group_by(Subject, Time.Point) %>% 
  dplyr::summarise_at(gradiometers_names, 
                      .f = mean)

mean_abs <- function(x){mean(abs(x))}

magnetometers_average_absolutes <- magnetometers %>% 
  dplyr::group_by(Subject, Time.Point) %>% 
  dplyr::summarise_at(magnetometers_names, 
                      .f = mean_abs)

gradiometers_average_absolutes <- gradiometers %>% 
  dplyr::group_by(Subject, Time.Point) %>% 
  dplyr::summarise_at(gradiometers_names, 
                      .f = mean_abs)

magnetometers_average_absolute_sensors <- magnetometers_average_absolutes %>% 
  tidyr::gather(key="Sensor", value="Activation", 3:length(magnetometers_average_absolutes))

gradiometers_average_absolute_sensors <- gradiometers_average_absolutes %>% 
  tidyr::gather(key="Sensor", value="Activation", 3:length(gradiometers_average_absolutes))


magnetometers_average_absolute_sensors %>% 
  ggplot(aes(x=Time.Point-500, y=Activation, color=Subject)) +
  facet_wrap(.~Sensor) +
  geom_line() + 
  theme_light() +
  labs(x="Time Point (ms)", y = "Average Absolute Sensor Activation", 
       title="Average Absolute Activation at Magnetometers") + 
  theme(axis.text.y = element_text(size=9),
        axis.text.x = element_text(size=9, hjust = 1, angle = 90),
        axis.title.x = element_text(size=15),
        axis.title.y = element_text(size=15),
        title = element_text(size=18)) 

gradiometers_average_absolute_sensors %>% 
  ggplot(aes(x=Time.Point-500, y=Activation, color=Subject)) +
  facet_wrap(.~Sensor) +
  geom_line() + 
  theme_light() +
  labs(x="Time Point (ms)", y = "Average Absolute Sensor Activation", 
       title="Average Absolute Activation at Gradiometers") + 
  theme(axis.text.y = element_text(size=9),
        axis.text.x = element_text(size=9, hjust = 1, angle = 90),
        axis.title.x = element_text(size=15),
        axis.title.y = element_text(size=15),
        title = element_text(size=18)) 

magnetometers_average_absolute_overall <- magnetometers_average_absolute_sensors %>% 
  dplyr::group_by(Subject, Time.Point) %>%
  dplyr::summarise(Activation = mean(Activation))

gradiometers_average_absolute_overall <- gradiometers_average_absolute_sensors %>% 
  dplyr::group_by(Subject, Time.Point) %>%
  dplyr::summarise(Activation = mean(Activation))

magnetometers_average_absolute_overall %>% 
  ggplot(aes(x = Time.Point-500, y=Activation, color = Subject)) +
  geom_line() +
  theme_light() +
  labs(x="Time Point (ms)", y = "Average Absolute Activation", 
       title="Average Absolute Magnetometer Activation") + 
  scale_x_continuous(breaks = round(seq(-500, 1000, by = 100),1))

gradiometers_average_absolute_overall %>% 
  ggplot(aes(x = Time.Point-500, y=Activation, color = Subject)) +
  geom_line() +
  theme_light() +
  labs(x="Time Point (ms)", y = "Average Absolute Activation", 
       title="Average Absolute Magnetometer Activation") + 
  scale_x_continuous(breaks = round(seq(-500, 1000, by = 100),1))


