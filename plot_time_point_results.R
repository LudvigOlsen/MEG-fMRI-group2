##---------------------------------------------------------------------------
## Plotting time point results
##---------------------------------------------------------------------------

library(ggplot2)
library(tidyverse)

# Set working dir
setwd("~/Development/python/MEG-fMRI-group2")
# setwd("~/Documents/Programmering/PythonLudvig/Machine learning/MEG-fMRI-group2")

group_name = "group_3"
model_name = "svm_2"
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

tiff(paste0(group_name, "_", model_name, "_F1_by_time_point.tif"), units="in", width = 4, height = 4, res = 300)
(time_point_averaged_results_plot <- ggplot(results_averaged_by_repetition, aes(x=Time.Point-500, y=F1)) +
    geom_line(size=0.25) +
    labs(title = "Average F1 per Time Point", x = "Time (ms)", y="F1",subtitle = paste0(group_name, ": ", model_name)) +
    # geom_smooth(data=results_averaged_by_repetition, size=1, aes(color = NULL)) + 
    theme_light()
)
dev.off()
