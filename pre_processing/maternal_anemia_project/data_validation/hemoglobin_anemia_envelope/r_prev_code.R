# load data from python output
library(readxl)
pacman::p_load(data.table,actuar)
setwd('H:/notebooks/vivarium_data_analysis/pre_processing/maternal_anemia_project/')

ids_vars <- c("location_id","age_group_id")
by_vars <- c("location_id","age_group_id","draw")

hb_pop <- read_excel("hb_pop.xlsx")
hb_var <- read_excel("hb_var.xlsx")
hb_preg <- read_excel("hb_preg.xlsx")
hb_non_preg <- read_excel("hb_non_preg.xlsx")
preg_var <- read_excel("preg_var.xlsx")
preg_cuts <- read_excel("preg_cuts.xlsx")
non_preg_cuts <- read_excel("non_preg_cuts.xlsx")
thresholds <- read_excel('thresholds.xlsx')


df <- rbind(df, df.p)
df <- merge(df, thresholds, by=c('age_group_id','pregnant'))

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

df=setDT(df)
df[,mild := ens_mv2prev(hgb_upper_mild, mean, variance, w = w) - ens_mv2prev(hgb_lower_mild, mean, variance, w = w)
   , by = 1:nrow(df)]
df[,moderate := ens_mv2prev(hgb_upper_moderate, mean, variance, w = w) - ens_mv2prev(hgb_lower_moderate, mean, variance, w = w)
   , by = 1:nrow(df)]
df[,severe := ens_mv2prev(hgb_upper_severe, mean, variance, w = w) 
   , by = 1:nrow(df)]
df[,total := mild+moderate+severe]

# great! now save this :) 
fwrite(df, "anemia_prevalence_from_r.csv")

