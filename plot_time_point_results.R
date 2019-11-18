##---------------------------------------------------------------------------
## Plotting time point results
##---------------------------------------------------------------------------

library(ggplot2)
library(tidyverse)

# Set working dir
setwd("~/Development/python/MEG-fMRI-group2")
# setwd("~/Documents/Programmering/PythonLudvig/Machine learning/MEG-fMRI-group2")

group_name = "group_4"
model_name = "svm_3"
path_to_result <- paste0("results/time_point_models/single/",
                         group_name,
                         "/results/",
                         model_name, "/",
                         group_name, "_",
                         model_name, "_",
                         "results_at_all_time_points.csv")

# Read results file
results <- read.csv(path_to_result) %>% 
  select(-X) %>% 
  mutate(Time.Point = as.integer(Time.Point),
         Repetition = as.factor(Repetition)) %>% 
  dplyr::as_tibble()

results_averaged_by_repetition <- results %>% 
  select(-c(Repetition)) %>%  
  group_by(Time.Point) %>% 
  dplyr::summarize_all(.funs = list(~mean(.)))

# Plot 

# Here we plot each repetition (repeated cross-validation) by themselves

(time_point_results_plot <- ggplot(results, aes(x=Time.Point-500, y=F1, color=Repetition)) +
    geom_line(size=0.2) +
    labs(title = "F1 by Time Point per CV Repetition", x = "Time (ms)", y="F1",subtitle = paste0(group_name, ": ", model_name)) +
    # geom_smooth(data=results_averaged_by_repetition, size=1, aes(color = NULL)) + 
    theme_light()
  )

# Here we plot the average repetition over time

max_f1_data <- results_averaged_by_repetition[results_averaged_by_repetition$F1 == max(results_averaged_by_repetition$F1),]
max_f1_tp <- max_f1_data$Time.Point-500
max_f1 <- round(max_f1_data$F1, 3)

tiff(paste0(group_name, "_", model_name, "_F1_by_time_point.tif"), units="in", width = 4, height = 4, res = 300)
(time_point_averaged_results_plot <- ggplot(results_averaged_by_repetition, aes(x=Time.Point-500, y=F1)) +
    geom_line(aes(y=0.5), color="grey", size=0.5) +
    geom_line(aes(x=max_f1_tp), color="grey", size=0.5) +
    geom_line(size=0.25) +
    geom_text(data = max_f1_data, aes(label = paste0("F1=",max_f1," at TP=",max_f1_tp)),nudge_x=0, nudge_y = 0.02) +
    # geom_text(data = max_f1, aes(label = paste0("",max_f1$Time.Point-500)),nudge_x=30, nudge_y = -0.48) +
    coord_cartesian(ylim = c(0.2,1)) +
    labs(title = "Average F1 per Time Point", x = "Time (ms)", y="F1",subtitle = paste0(group_name, ": ", model_name)) +
    # geom_smooth(data=results_averaged_by_repetition, size=1, aes(color = NULL)) + 
    theme_light()
)
dev.off()
