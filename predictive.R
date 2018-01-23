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
set.seed(2, "L'Ecuyer-CMRG")

# Tuning XGB
XGB1 = create.Learner("SL.xgboost",
                      tune = list(ntrees = 2000, max_depth = 4, shrinkage = 0.01),
                      detailed_names = T, name_prefix = "XGB")
XGB2 = create.Learner("SL.xgboost",
                     tune = list(ntrees = 1000, max_depth = 3, shrinkage = 0.05),
                     detailed_names = T, name_prefix = "XGB")
XGB3 = create.Learner("SL.xgboost",
                     tune = list(ntrees = 500, max_depth = 4, shrinkage = 0.05),
                     detailed_names = T, name_prefix = "XGB")

# Create Super Learner library
SL.library <- c("SL.glm", "SL.glmnet", "SL.ridge",
                "SL.gam", "SL.loess", "SL.polymars",
                "SL.randomForest", "SL.cforest",
                "SL.xgboost", XGB1$names, XGB2$names, XGB3$names)

# Read first imputed dataset and transform data
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

set.seed(3)
SLfull = CV.SuperLearner(Y = Y, X = X,
                         family = gaussian(), SL.library = SL.library,
                         method = "method.NNLS", verbose = TRUE,
                         cvControl = list(V = 5L, shuffle = TRUE),
                         parallel = "multicore")
plot.CV.SuperLearner(SLfull)
result_CVfull <- summary.CV.SuperLearner(SLfull)$Table

# Plot algorithm performance
result_CVfull <- result_CVfull %>%
  mutate(algorithms = c("Super Learner", "Discrete Super Learner",
                        "Linear model", "Linear model with lasso",
                        "Ridge regression", "Generalized additive model",
                        "Local regression", "Polynomial spline regression",
                        "Random forest", "Conditional random forest",
                        "XGBoost (default)", "XGBoost 2000_4_0.01",
                        "XGBoost 1000_3_0.05", "XGBoost 500_4_0.05"))
result_CVfull$algorithms <- factor(result_CVfull$algorithms,
                          levels = result_CVfull$algorithms[order(result_CVfull$Ave,
                                                                 decreasing = TRUE)])
algo <- ggplot(data = result_CVfull,
               aes(x = algorithms, y = Ave, ymin = Min, ymax = Max)) +
  geom_pointrange(size = 1.2) + 
  coord_flip() + ggtitle("Algorithms by cross-validated MSE in predicting state repression") +
  theme_fivethirtyeight() + theme(axis.text = element_text(size = 20))
algo

save.image("predictive.RData")
