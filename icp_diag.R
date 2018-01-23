# Specify the number of digits
options(digits = 4)
options(dplyr.width = Inf)
options("scipen" = 4)

# Load packages
library(dplyr) # Upload dplyr to process data
library(tidyr) # Data processing
library(reshape2) # Convert data form
library(ggplot2) # visualize data
library(ggthemes) # visualize data
library(SuperLearner) # Super Learner prediction
library(FSA) # Summarize the data
library(rcompanion) # Permutation test
library(coin) # Permutation test of independence

# Read first imputed dataset and transform data
data <- read.csv("midata1.csv") %>%
  dplyr::select(c(year, ccode,
                  final_decision, fair_trial, public_trial, legislative_ck,
                  britcol, common, latent, latent_lag,
                  pop, ythblgap, gdppc, rentspc, iwar,
                  military, execrlc, xconst, xropen,
                  injud, cpr_ratify, cat_ratify,
                  trade_gdp, fdi_net_in, structad, wbimfstruct, imfstruct, hr_law,
                  ingo_uia, aibr_lag, ainr_lag, avmdia_lag, hro_shaming_lag)) %>%
  na.omit()

# Tuning XGB
XGB = create.Learner("SL.xgboost",
                     tune = list(ntrees = 500, max_depth = 4, shrinkage = 0.05),
                     detailed_names = T, name_prefix = "XGB")
# Create Super Learner library
SL.library <- c("SL.glm", "SL.glmnet", "SL.ridge",
                "SL.gam", "SL.polymars", "SL.loess",
                "SL.randomForest", "SL.cforest", "SL.xgboost", XGB$names)
SL.library <- c(XGB$names)

# Define target variable and predictive covariates
Y <- data$latent
X <- dplyr::select(data, c(3:8, 10:33))

# XGBoost prediction function and compute residuals
SL.fit <- SuperLearner(Y = Y, X = X, SL.library = SL.library,
                       family = "gaussian", method = "method.NNLS",
                       verbose = TRUE, cvControl = list(V = 5L, shuffle = TRUE))
pred <- predict.SuperLearner(SL.fit)$pred[, 1]
data$resid <- (Y - pred)
data$s.resid <- scale(data$resid)[, 1]

# Histogram of residuals
histplot <- ggplot(data = data, aes(x = s.resid)) +
  geom_histogram(binwidth = 0.175, alpha = 0.75, aes(y = ..density..)) +
  stat_function(fun = dnorm, colour = "blue", lwd = 1.5,
                arg = list(mean = mean(data$s.resid),
                           sd = sd(data$s.resid))) +
  theme_wsj() + theme(axis.text = element_text(size = 18)) +
  labs(title = "Histogram of XGBoost-predicted scaled residuals",
       subtitle = "Overlayed empirical normal curve")
histplot

# Permutation test for independence between time indices and residuals
perm_data <- data %>% dplyr::select(year, ccode, resid, s.resid) %>% arrange(year)
summarystates <- Summarize(s.resid ~ factor(year), data = perm_data)
boxp <- ggplot(data = perm_data, aes(x = factor(year), y = s.resid)) +
  geom_boxplot() + geom_jitter(alpha = 0.5) +
  theme_wsj() + theme(axis.text = element_text(size = 18)) +
  ggtitle("Scaled XGBoost-predicted residuals by year")
boxp

scatterp <- ggplot(data = perm_data, aes(x = year, y = s.resid)) +
  geom_point(alpha = 0.3) + geom_smooth(method = "loess") +
  theme_wsj() + theme(axis.text = element_text(size = 18)) +
  ggtitle("Scaled XGBoost-predicted residuals scatterplot")
scatterp

# Independence test
indtest <- independence_test(s.resid ~ factor(year), data = perm_data)

# Pairwise permutation tests
PM <- pairwisePermutationMatrix(s.resid ~ factor(year), data = perm_data, method = "fdr")

# Plot pairwise permutation test pvalues Unadjusted
PMunadj <- data.frame(log(PM$Unadjusted))
colnames(PMunadj) <- as.character(c(1982:1999))
PMunadj <- PMunadj %>% mutate(id = c(1982:1999))
gatherPMunadj <- gather(PMunadj, key, value, -id) %>% na.omit() %>%
  arrange(id)
pvalue <- ggplot(data = gatherPMunadj, aes(x = factor(id), y = value)) +
  geom_point(size = 4, alpha = 0.75) + geom_jitter() +
  geom_hline(yintercept = -3, col = "red", lwd = 1.5) + 
  theme_wsj() + theme(axis.text = element_text(size = 18)) +
  labs(title = "Unadjusted (log) p-values of pairwise permutation tests")
pvalue

# Plot pairwise permutation test pvalues Adjusted
PMadj <- PM$Adjusted
PMadj[lower.tri(PMadj, diag = TRUE)] <- NA
PMadj <- data.frame(log(PMadj))
colnames(PMadj) <- as.character(c(1982:1999))
PMadj <- PMadj %>% mutate(id = c(1982:1999))
gatherPMadj <- gather(PMadj, key, value, -id) %>% na.omit() %>%
  arrange(id)
pvalue_ad <- ggplot(data = gatherPMadj, aes(x = factor(id), y = value)) +
  geom_point(size = 4, alpha = 0.75) + geom_jitter() +
  theme_wsj() + theme(axis.text = element_text(size = 18)) +
  labs(title = "Adjusted (log) p-values of pairwise permutation tests")
pvalue_ad

# Kruskal-Wallis test of all medians are equal
kwtest <- kruskal_test(s.resid ~ factor(year),
                       distribution = approximate(B = 10000), data = perm_data)

# Pairwise group-comparisons using Wilcoxon's rank-sum test
wilcontest <- pairwise.wilcox.test(perm_data$s.resid,
                                   factor(perm_data$year),
                                   p.adjust.method = "holm")

# Save results
save.image("IndTest_XGB.RData")
