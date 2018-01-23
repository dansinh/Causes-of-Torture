# Specify the number of digits
options(digits = 3)
options(dplyr.width = Inf)
options("scipen" = 4)

# Load packages
library(dplyr) # Upload dplyr to process data
library(ggplot2) # visualize data
library(xgboost) # algorithm for XGBoost
library(foreach) # do parallel loop
library(doParallel) # do parallel loop
library(RhpcBLASctl) #multicore
library(SuperLearner)

# Setup parallel computation - use all cores on our computer.
num_cores = RhpcBLASctl::get_num_cores()

# Use all of those cores for parallel SuperLearner.
options(mc.cores = num_cores)

# Check how many parallel workers we are using: 
getOption("mc.cores")

# We need to set a different type of seed that works across cores.
set.seed(11, "L'Ecuyer-CMRG")

# Read stacked data sets and process data
data <- read.csv("midata1.csv") %>%
  dplyr::select(c(year, ccode, latent, latent_lag, gdppc, pop,
                  cwar, ythblgap, injud, trade_gdp,
                  parcomp, polity2, ingo_uia, xconst,
                  rentspc, aibr_lag, final_decision, britcol,
                  xrcomp, fair_trial, cpr_ratify, common,
                  public_trial, military, execrlc, legislative_ck,
                  fdi_net_in, ainr_lag, cat_ratify, hr_law,
                  avmdia_lag, structad, wbimfstruct, xropen,
                  imfstruct, iwar, hro_shaming_lag)) %>% na.omit()

# Total 2,096 observations across 154 countries over 18 years
Y <- data$latent
X <- data.frame(data[, 4:37])

# 3*6*3 = 54 different configurations.
tune = list(ntrees = c(500, 1000, 2000),
            max_depth = c(3:8),
            shrinkage = c(0.01, 0.05, 0.1))

# Set detailed names = T so we can see the configuration for each function.
learners = create.Learner("SL.xgboost", tune = tune,
                          detailed_names = T, name_prefix = "XGB")

# Fit the SuperLearner using ICCPR, CEDAW, and CAT first imputed datasets
SL.library <- c(learners$names)

# Tune XGBoost using cross-validated super learner
set.seed(2)
sl = CV.SuperLearner(Y = Y, X = X,
                     family = gaussian(), SL.library = SL.library,
                     method = "method.NNLS", verbose = TRUE,
                     cvControl = list(V = 4L, shuffle = TRUE),
                     parallel = "multicore")
plot.CV.SuperLearner(sl)
tuning_result <- summary.CV.SuperLearner(sl)$Table
tuning_result[order(tuning_result$Ave), ]

# Plot tuning results
tuning_result$Algorithm <- factor(tuning_result$Algorithm,
                          levels = tuning_result$Algorithm[order(tuning_result$Ave,
                                                                 decreasing = TRUE)])
tuning <- ggplot(data = tuning_result,
                 aes(x = Algorithm, y = Ave, ymin = Min, ymax = Max)) +
  geom_pointrange(size = 1.2) + coord_flip() +
  ggtitle("XGBoost configurations by (mean, min, and max) cross-validated MSE in predicting state repression") +
  theme_fivethirtyeight() + theme(axis.text = element_text(size = 10))
tuning

save.image("XGBtuning.RData")