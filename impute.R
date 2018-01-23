options(digits = 2)
options(dplyr.width = Inf)
rm(list = ls())
cat("\014")

library(dplyr) # Upload dplyr to process data
library(tidyr) # tinyr package to tidy data
library(foreign) # Read Stata data
library(ggplot2) # ggplot graphics
library(Amelia) # Multiple imputation
library(lubridate) # Handle dates data

## load data
data <- read.csv("rep.csv") %>%
  select(c(ccode, year,
           latent, latent_lag,
           gdppc, pop,
           cwar, ythblgap, injud, trade_gdp,
           xconst, rentspc, ingo_uia,
           cat_ratify, cpr_ratify))

# Multiple imputation using Amelia package
set.seed(123)
mi.data <- amelia(data, m = 5, ts = "year", cs = "ccode",
                  p2s = 2, polytime = 3,
                  logs = c("gdppc", "pop", "trade_gdp", "rentspc"),
                  ords = c( "injud", "xconst"),
                  emburn = c(50, 500), boot.type = "none",
                  bounds = rbind(c(3, -3, 4), c(4, -3, 4), c(8, 12, 45)))

# Write imputed data sets into CSV files
save(mi.data, file = "midata.RData")
write.amelia(obj = mi.data, file.stem = "midata", row.names = FALSE)

# Create missingness map and diagnostics
missmap(mi.data)
summary(mi.data)
