##---------------------------------------------------------------------------
## Plotting time point results
##---------------------------------------------------------------------------

library(ggplot2)
library(tidyverse)

# Set working dir
setwd("~/Documents/Programmering/PythonLudvig/Machine learning/MEG-fMRI-group2")

path_to_result <- "results/main_results_backup/svm_2_group_1_results_at_all_time_points.csv"

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

(time_point_results_plot <- ggplot(results, aes(x=Time.Point, y=F1, color=Repetition)) +
    geom_line(size=0.1) +
    # geom_smooth(data=results_averaged_by_repetition, size=1, aes(color = NULL)) + 
    theme_light()
  )

# Here we plot the average repetition over time

tiff("svm_2_group_1_F1_by_time_point.tif", units="in", width = 4, height = 4, res = 600)
(time_point_averaged_results_plot <- ggplot(results_averaged_by_repetition, aes(x=Time.Point, y=F1)) +
    geom_line(size=0.15) +
    # geom_smooth(data=results_averaged_by_repetition, size=1, aes(color = NULL)) + 
    theme_light()
)
dev.off()
