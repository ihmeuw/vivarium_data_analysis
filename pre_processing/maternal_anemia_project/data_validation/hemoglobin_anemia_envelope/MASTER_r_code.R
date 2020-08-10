
-----------------------------------------------------------------------------------------------
# clear memory
rm(list=ls())

# disable scientific notation
options(scipen = 999)

# load data from python output
library(readxl)
pacman::p_load(data.table,actuar)
setwd('H:/notebooks/vivarium_data_analysis/pre_processing/maternal_anemia_project/')

mean <- read_excel("mean.xlsx")
sd <- read_excel("sd.xlsx")
sb <- read_excel("sb.xlsx")
asfr <- read_excel("asfr.xlsx")
thresholds <- read_excel("thresholds.xlsx")


#***********************************************************************************************************************

#Distribution Functions
XMAX = 220
#Load ensemble weights - in future this will be automated but this was from dx optimization 
w = c(0.4,0.6)

#----MODEL-------------------------------------------------------------------------------------------------------------

###################
### LOAD DRAWS ####
###################
id_vars <- c("measure_id", "location_id", "year_id", "age_group_id", "sex_id")
by_vars <- c("measure_id", "location_id", "year_id", "age_group_id", "sex_id", "draw")
id_vars.p <- id_vars[id_vars != "measure_id"]

means = as.data.table(mean)
stdev = as.data.table(sd)

means[, c("modelable_entity_id", "model_version_id", "metric_id") := NULL]
stdev[, c("modelable_entity_id", "model_version_id", "metric_id") := NULL]

means.l <- melt(means, id.vars = id_vars, variable.name = "draw", value.name = "mean")
stdev.l <- melt(stdev, id.vars = id_vars, variable.name = "draw", value.name = "stdev")

df <- merge(means.l, stdev.l, by = by_vars)
df[, variance := stdev ^ 2]


### CALCULATE PREGNANCY RATE 
### No uncertainty captured because ASFR covariate doesn't have any 

#Age-spec-preg-prev = (ASFR + stillbirth) * 46/52
setnames(asfr, "mean_value", "asfr")

#stillbirths 
setnames(sb,"mean_value","sbr_mean")
sb <- as.data.table(sb)
sb[,age_group_id := NULL]
sb[,sex_id := NULL]

#Merge - stillbirths are only location-year specific 
df.p <- merge(asfr, sb, by = c("location_id", "year_id"))

#Stillbirth_mean is still births per live birth
df.p <- as.data.table(df.p)
df.p[, prev_pregnant := (asfr + (sbr_mean * asfr)) * 46/52  ]
print(min(df.p$prev_pregnant))
#if(max(df.p$prev_pregnant) > 0.5) stop("PREGNANCY PREV OVER 50% - CHECK MATH?") 


#Subset to pregnant 

#Anemia threshold for <15 does NOT depend on pregnaancy!
df.p <- df.p[age_group_id >= 8]

#Pregnant prev 
preg_prev <- copy(df.p)
preg_prev <- preg_prev[, c(id_vars.p, "prev_pregnant"), with = FALSE]

#Merge to mean df
df.p <- df.p[, c("location_id", "year_id", "age_group_id", "sex_id", "prev_pregnant"), with = FALSE]
df.p <- merge(df, df.p, by = c("location_id", "year_id", "age_group_id", "sex_id"))

df[, pregnant := 0]
df.p[, pregnant := 1]

#Calc pregnant mean & stdev - sorry for hard coding! This comes from crosswalk in data prep 
df.p[, mean := mean/0.919325]
df.p[, variance := variance * (1.032920188 ^ 2)]
df.p[, stdev := sqrt(variance)]

df.preg <- copy(df.p)
df.preg[, prev_pregnant := NULL]

df <- rbind(df, df.preg)




#MAP THRESHOLDS

df <- merge(df, thresholds, by = c("age_group_id", "sex_id", "pregnant"))

# define relevant constants
EULERS_CONSTANT <- 0.57721566490153286060651209008240243104215933593992
XMAX <- 220
gamma_w <- 0.4
m_gum_w <- 0.6

# define functions from GBD modelers
gamma_mv2p = function(mn, vr){list(shape = mn^2/vr,rate = mn/vr)}

mgumbel_mv2p = function(mn, vr){
  list(
    alpha = XMAX - mn - EULERS_CONSTANT*sqrt(vr)*sqrt(6)/pi,
    scale = sqrt(vr)*sqrt(6)/pi
  ) 
}

# note: pgamma is a standard R function, does not need defining here

pmgumbel = function(q, alpha, scale, lower.tail) 
{ 
  #NOTE: with mirroring, take the other tail
  pgumbel(XMAX-q, alpha, scale, lower.tail=ifelse(lower.tail,FALSE,TRUE)) 
}

###ABBREVIATED - FOR JUST GAMMA AND MGUMBEL
ens_mv2prev <- function(q, mn, vr, w){
  x = q
  
  ##parameters
  params_gamma = gamma_mv2p(mn, vr)
  params_mgumbel = mgumbel_mv2p(mn, vr)
  
  ##weighting
  prev = sum(
    w[1] * pgamma(x, data.matrix(params_gamma$shape),data.matrix(params_gamma$rate)), 
    w[2] * pmgumbel(x,data.matrix(params_mgumbel$alpha),data.matrix(params_mgumbel$scale), lower.tail=T)
  )
  prev
  
}

w = c(0.4,0.6)


### CALCULATE PREVALENCE 
print("CALCULATING MILD")
df[, mild := ens_mv2prev(hgb_upper_mild, mean, variance, w = w) - ens_mv2prev(hgb_lower_mild, mean, variance, w = w)
   , by = 1:nrow(df)]
print("CALCULATING MOD")
df[, moderate := ens_mv2prev(hgb_upper_moderate, mean, variance, w = w) - ens_mv2prev(hgb_lower_moderate, mean, variance, w = w)
   , by = 1:nrow(df)]
print("CALCULATING SEV")
df[, severe := ens_mv2prev(hgb_upper_severe, mean, variance, w = w) - ens_mv2prev(hgb_lower_severe, mean, variance, w = w)
   , by = 1:nrow(df)]
#Anemic is the sum
df[, anemic := mild + moderate + severe]

sevs <- c("mild", "moderate", "severe", "anemic")
###PREGNANCY ADJUSTMENT!
df.p <- df[pregnant == 1]
df.p <- merge(df.p, preg_prev, by = id_vars.p, all = TRUE)
df.p <- df.p[, c(by_vars, sevs, "prev_pregnant"), with = FALSE]
lapply(sevs, function(s) setnames(df.p, s, paste(s, "preg", sep = "_")))


df <- df[pregnant == 0]

nrow(df)
df <- merge(df, df.p, by = by_vars, all = TRUE)
nrow(df)


#weighted sum
test <- df[!is.na(prev_pregnant)]
lapply(sevs, function(sev) df[ !is.na(prev_pregnant) , c(sev) := get(sev) * (1 - prev_pregnant) + get(paste0(sev, "_preg")) * prev_pregnant ]) 
test2 <- df[!is.na(prev_pregnant)]


#Everything is prevalence (means/stdev entered as continuous)
df[,measure_id := 5]

#RESHAPE
mild <- df[,c(by_vars, "mild"), with = F]
mild <- dcast(mild, ... ~ draw, value.var = "mild", drop = T, fill = NA)
moderate <- df[,c(by_vars, "moderate"), with = F]
moderate <- dcast(moderate, ... ~ draw, value.var = "moderate", drop = T, fill = NA)
severe <- df[,c(by_vars, "severe"), with = F]
severe <- dcast(severe, ... ~ draw, value.var = "severe", drop = T, fill = NA)
anemic <- df[,c(by_vars, "anemic"), with = F]
anemic <- dcast(anemic, ... ~ draw, value.var = "anemic", drop = T, fill = NA)


fwrite(df, "updated_anemia_prevalence_from_r.csv")