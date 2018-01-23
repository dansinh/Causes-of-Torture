# Set digits options
options(digits = 4)
options(dplyr.width = Inf)
options("scipen" = 4)

# Load packages
library(dplyr) # Manage data
library(ggplot2) # visualize data
library(ggthemes) # visualize data
library(SuperLearner) # use Super Learner predictive method
library(xtable) # create LaTeX tables
library(RhpcBLASctl) #multicore
library(doParallel) # parallel computing
library(foreach) #  parallel computing

# Tuning XGB
XGB = create.Learner("SL.xgboost",
                     tune = list(ntrees = 500, max_depth = 4, shrinkage = 0.05),
                     detailed_names = T, name_prefix = "XGB")
# Create Super Learner library
SL.library <- c(XGB$names)

# Parallelization
cluster = parallel::makeCluster(2)

# Load the SuperLearner package on all workers so they can find
# SuperLearner::All(), the default screening function which keeps all variables.
parallel::clusterEvalQ(cluster, library(SuperLearner))

# We need to explictly export our custom learner functions to the workers.
parallel::clusterExport(cluster, XGB$names)
parallel::clusterSetRNGStream(cluster, 2)

# Create function rescaling outcome into 0-1
std <- function(x) {
  x = (x - min(x))/(max(x) - min(x))
}

# Read first imputed dataset and transform data
data <- read.csv("midata1.csv") %>%
  dplyr::select(c(year, ccode,
                  final_decision, fair_trial, public_trial, legislative_ck,
                  britcol, common, physint, physint_lag,
                  pop, ythblgap, gdppc, rentspc, iwar,
                  military, execrlc, xconst, xropen,
                  injud, cpr_ratify, cat_ratify,
                  trade_gdp, fdi_net_in, structad, wbimfstruct, imfstruct, hr_law,
                  ingo_uia, aibr_lag, ainr_lag, avmdia_lag, hro_shaming_lag)) %>%
  group_by(ccode) %>% mutate(pop2 = lag(pop, 1), ythblgap2 = lag(ythblgap, 1),
                             gdppc2 = lag(gdppc, 1), rentspc2 = lag(rentspc, 1),
                             iwar2 = lag(iwar, 1), military2 = lag(military, 1),
                             execrlc2 = lag(execrlc, 1), xconst2 = lag(xconst, 1),
                             xropen2 = lag(xropen, 1),
                             injud2 = lag(injud, 1), cpr_ratify2 = lag(cpr_ratify, 1),
                             cat_ratify2 = lag(cat_ratify, 1),
                             trade_gdp2 = lag(trade_gdp, 1),
                             fdi_net_in2 = lag(fdi_net_in, 1),
                             structad2 = lag(structad, 1),
                             wbimfstruct2 = lag(wbimfstruct, 1),
                             imfstruct2 = lag(imfstruct, 1),
                             hr_law2 = lag(hr_law, 1),
                             ingo_uia2 = lag(ingo_uia, 1)) %>% na.omit()

for (i in c(3:6, 9:14, 17:20, 23:24, 29:37, 40:43, 46:47, 52)){
  data[, i] <- std(data[, i])
}

# Set multicore compatible seed.
set.seed(1, "L'Ecuyer-CMRG")
# Setup parallel computation - use all cores on our computer.
num_cores = RhpcBLASctl::get_num_cores()
# Use all of those cores for parallel SuperLearner.
options(mc.cores = num_cores)
# Check how many parallel workers we are using: 
getOption("mc.cores")

# For bootstrap-based inference, use stochastic imputation with 1 imputed dataset
# Take quantile for CI, no need for normality assumption
cl <- makeCluster(2)
registerDoParallel(cl)
B <- 500
psi_boot <- data.frame(matrix(NA, nrow = B, ncol = 23))

foreach(b = 1:B, .packages = c("dplyr", "xgboost", "SuperLearner"),
        .verbose = TRUE) %do% {
          
          # Create bootstrap indices
          bootIndices <- sample(1:nrow(data), replace = TRUE)
          bootData <- data[bootIndices, ]
          ndata <- nrow(bootData)
          
          # Causal effect of each time-varying covariate using resample dataset
          psi <- data.frame(matrix(NA, nrow = 1, ncol = 23))
          
          for (i in 1:19) {
            Y <- bootData$physint
            X <- data.frame(bootData[, c(i + 10, 3:8, 10, 30:52)])
            X1 <- X0 <- X
            X1[, 1] <- 1
            X0[, 1] <- 0
            newdata <- rbind(X, X1, X0)
            Qi <- snowSuperLearner(Y = Y, X = X, SL.library = SL.library,
                                   newX = newdata,
                                   family = "gaussian", method = "method.NNLS",
                                   verbose = TRUE, cluster = cluster,
                                   cvControl = list(V = 5L, shuffle = TRUE))
            predX1 <- Qi$SL.predict[(ndata + 1):(2*ndata)]
            predX0 <- Qi$SL.predict[(2*ndata + 1):(3*ndata)]
            psi[, i] <- mean(predX1 - predX0)
            print(c(b, i))
          }
          
          for (j in 1:4){
            Y <- bootData$physint
            X2 <- data.frame(bootData[, c(30:52, 3:8, 10)])
            X2_1 <- X2_0 <- X2
            X2_1[, j] <- 1
            X2_0[, j] <- 0
            newdata2 <- rbind(X2, X2_1, X2_0)
            Qj <- snowSuperLearner(Y = Y, X = X2, SL.library = SL.library,
                                   newX = newdata2,
                                   family = "gaussian", method = "method.NNLS",
                                   verbose = TRUE, cluster = cluster,
                                   cvControl = list(V = 5L, shuffle = TRUE))
            predX2_1 <- Qj$SL.predict[(ndata + 1):(2*ndata)]
            predX2_0 <- Qj$SL.predict[(2*ndata + 1):(3*ndata)]
            psi[, 19 + j] <- mean(predX2_1 - predX2_0)
          }
          # Combine bootstrap estimates
          psi_boot[b, 1:23] <- psi
        }

lower_quantile <- function(x, prob){quantile(x, prob = 0.025)}
upper_quantile <- function(x, prob){quantile(x, prob = 0.975)}
mean_boot <- apply(psi_boot, 2, mean)
lower_boot <- apply(psi_boot, 2, lower_quantile)
upper_boot <- apply(psi_boot, 2, upper_quantile)

cvim2_xgb <- data.frame(cbind(mean = mean_boot[1:23],
                             lower = lower_boot[1:23],
                             upper = upper_boot[1:23])) %>%
  mutate(covariate = c("Population", "Youth Budge",
                       "GDP per capita", "log Oil Rents",
                       "International War", "Military Regime", "Left Executive",
                       "Executive Constraint", "Executive Recruit Open",
                       "Judicial Independence", "ICCPR", "CAT",
                       "log Trade/GDP", "FDI Net",
                       "WB Struct. Adj.", "WB/IMF Struc. Adj.",
                       "IMF Struc. Adj.", "PTA w/ HR Clause",
                       "log INGOs", "AI Background (lag)", "AI Press Release (lag)",
                       "Western Media Shaming (lag)", "HRO Shaming (lag)")) %>%
  dplyr::select(c(covariate, mean, lower, upper)) %>%
  dplyr::arrange(mean)
# Plot variable importance
cvim2_xgb$covariate <- factor(cvim2_xgb$covariate,
                             levels = cvim2_xgb$covariate[order(cvim2_xgb$mean)])
fplot <- ggplot(data = cvim2_xgb,
                aes(x = covariate, y = mean, ymin = lower, ymax = upper)) +
  geom_pointrange(size = 1.2) + geom_hline(yintercept = 0, lty = 2, size = 1) +
  coord_flip() + ggtitle("Causal effects of covariates on physical integrity rights") +
  theme_fivethirtyeight() + theme(axis.text = element_text(size = 20))
fplot

# Create LaTex table
colnames(cvim2_xgb) <- c("Covariate", "Mean Causal Effect", "Lower", "Upper")
xtable(cvim2_xgb, digits = c(rep(3, 5)))

# Save results
save.image("cvim2_XGB.RData")
