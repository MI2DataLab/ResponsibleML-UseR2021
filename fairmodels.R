################# imports #################

library(magrittr)
library(ggplot2)
library(DALEX)
library(fairmodels)
library(gbm)
library(ranger)

################# data #################

head(fairmodels::compas)

compas2 <- read.csv('https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv')
head(compas2)



df <- compas2[,c('age', 'c_charge_degree', 'race', 'age_cat',
                 'score_text', 'sex', 'priors_count', 'juv_misd_count',
                 'v_decile_score', 'days_b_screening_arrest',
                 'decile_score', 'two_year_recid', 'c_jail_in', 'c_jail_out')]

df <- df[df$days_b_screening_arrest <= 30, ]
df <- df[df$days_b_screening_arrest >= -30, ]
df <- df[df$c_charge_degree != "O", ]
df <- df[df$score_text != 'N/A', ]
df$jail_time = as.numeric(difftime(df$c_jail_out,
                                   df$c_jail_in, units = c('days')))
df <- na.omit(df)
df <- df[ -c(4:5, 13:14)]

head(df)

# Here we change the order of the recidivism, so that the model
# predicts positive outcome
df$two_year_recid <- ifelse(df$two_year_recid == 1, 0, 1)


################# Fairness Check #################

# Classification task - will defendants become recidivist?

lr_model <- glm(two_year_recid ~.,
                data = df,
                family = binomial())

lr_explainer <- DALEX::explain(lr_model, data = df, y = df$two_year_recid)

# lets check the performance
model_performance(lr_explainer)

# let's do fairness check and quickly check if the model is fair
fairness_check(lr_explainer,
               protected = df$race,
               privileged = 'Caucasian')

# assigning to variable
fobject <- fairness_check(lr_explainer,
                          protected = df$race,
                          privileged = 'Caucasian')
class(fobject)

print(fobject)
plot(fobject)


# Insides
fobject$groups_confusion_matrices

fobject$groups_data
fobject$privileged
fobject$protected
fobject$cutoff
fobject$epsilon

# Let's make it more visible

df <- df[df$race %in% c('Caucasian', 'African-American'), ]
head(df)

lr_model <- glm(two_year_recid ~., data = df, family = binomial())
lr_explainer <- DALEX::explain(lr_model, data = df, y = df$two_year_recid)

fobject <- fairness_check(lr_explainer,
                          protected = df$race,
                          privileged = 'Caucasian')

plot(fobject)
### How to read it?

# what happens if i tweak this?
fobject <- fairness_check(lr_explainer,
                          protected = df$race,
                          privileged = 'Caucasian',
                          epsilon = 0.95)

plot(fobject)

# ok but lets revert now
fobject <- fairness_check(lr_explainer,
                          protected = df$race,
                          privileged = 'Caucasian')
plot(fobject)

################# More Models #################

rf_model <- ranger::ranger(as.factor(two_year_recid) ~.,
                           data=df,
                           probability = TRUE,
                           num.trees = 100,
                           max.depth = 7,
                           seed = 123)

rf_explainer <- DALEX::explain(rf_model, data = df, y = df$two_year_recid)

model_performance(rf_explainer)

# we have a few options to compare the models


# 1. explainer and fairness object
## 1.1
fobject2 <- fairness_check(rf_explainer, fobject, # with fobject
                           protected = df$race,
                           privileged = "Caucasian")
## 1.2 (recommended)
fobject2 <- fairness_check(rf_explainer, fobject) # with fobject

# 2. fairness objects
fobject2 <- fairness_check(rf_explainer,
                           protected = df$race,
                           privileged = "Caucasian")

fobject2 <- fairness_check(fobject, fobject2)


# 3. 2 explainers
fobject2 <- fairness_check(rf_explainer, lr_explainer,
                           protected = df$race,
                           privileged = "Caucasian")

plot(fobject2)
fobject2
# more than 2 is ok.

# what if the race in the data is the case?
df2 <- df[c('two_year_recid',
            'age',
            'c_charge_degree',
            'priors_count',
            'decile_score')]

rf_model2 <- ranger::ranger(as.factor(two_year_recid) ~.,
                           data=df2,
                           num.trees = 100,
                           max.depth = 7,
                           seed = 123,
                           probability = TRUE)

rf_explainer2 <- DALEX::explain(rf_model2,
                                data = df2,
                                y = df2$two_year_recid)
model_performance(rf_explainer2)

# can I add this like that?
fobject3 <- fairness_check(rf_explainer2, fobject2,
                           protected = df$race,
                           privileged = "Caucasian")

# no.. rather like this
fobject3 <- fairness_check(rf_explainer2, fobject2,
                           protected = df$race,
                           privileged = "Caucasian",
                           label = 'ranger_without_race')

plot(fobject3)

################# Other Plots #################

# shows raw scores of metrics
fobject3 %>% metric_scores() %>% plot()

# parity loss - back to presentation
fobject3$parity_loss_metric_data


fobject3 %>% fairness_radar() %>% plot()

?fairness_radar

fobject3 %>% fairness_radar(fairness_metrics = c("TPR", "STP", "FNR")) %>%
  plot()

# all metrics?

fobject3 %>% fairness_heatmap() %>% plot()

# summarised metrics?
# default metrics - those in fairness_check
fobject3 %>% stack_metrics() %>% plot()


# metric and performance? No problem
fobject3 %>%
  performance_and_fairness(fairness_metric = 'FPR',
                           performance_metric = 'accuracy') %>%
  plot()

# hard to remember? No problem!
?plot_fairmodels()

fobject3 %>% plot_fairmodels("stack_metrics")

# okay we have bias? What can we do?
# A lot!


################# Mitigation methods #################

##### Pre processing

# Let's construct a model previously used.

fobject <- fairness_check(rf_explainer,
                          protected = df$race,
                          privileged = 'Caucasian')

# resampling
indices <- resample(protected = df$race, df$two_year_recid)
df_resampled <- df[indices,]

rf_model_resampled <- ranger::ranger(as.factor(two_year_recid) ~.,
                           data=df_resampled,
                           num.trees = 100,
                           max.depth = 7,
                           seed = 123,
                           probability = TRUE)

rf_explainer_resampled <- DALEX::explain(rf_model_resampled,
                               data = df,
                               y = df$two_year_recid,
                               label = 'resampled')


fobject <- fairness_check(fobject, rf_explainer_resampled)
plot(fobject)

# reweight
weights <- reweight(protected = as.factor(df$race), y=df$two_year_recid)

rf_model_reweighted <- ranger::ranger(as.factor(two_year_recid) ~.,
                                     data=df,
                                     num.trees = 100,
                                     max.depth = 7,
                                     seed = 123,
                                     case.weights = weights,
                                     probability  = TRUE)

rf_explainer_reweighted <- DALEX::explain(rf_model_reweighted,
                                         data = df,
                                         y = df$two_year_recid,
                                         label = 'reweighted')


fobject <- fairness_check(fobject, rf_explainer_reweighted)
plot(fobject)

##### Post-Processing

# ROC pivot
rf_explainer_roc2 <- roc_pivot(rf_explainer,
                              protected =  df$race,
                              privileged = "Caucasian",
                              theta = 0.05)


fobject <- fairness_check(fobject, rf_explainer_roc2, label = "roc2")

plot(fobject)

# Cutoff manipulation

rf_explainer %>%
  fairness_check(protected = df$race, privileged = 'Caucasian') %>%
  ceteris_paribus_cutoff("African-American") %>%
  plot()

fobject <- fairness_check(fobject, rf_explainer,
                          cutoff = list('African-American'= 0.36),
                          label = 'ranger_cutoff')


plot(fobject)

# checking FPR and accuracy
fobject %>% performance_and_fairness(fairness_metric = 'FPR',
                                     performance_metric = 'accuracy') %>% plot()

################# Exercise #################

# As for the exercise: check fairness for the same
# model (ranger) but with protected vector equal to df$sex.
# Add to this a linear model (lr) and plot ceteris peribus cutoff
# for subgroup male

# SOLUTION

fc <- fairness_check(rf_explainer, protected = df$sex, privileged = "Female")
plot(fc)

fc <- fairness_check(fc, lr_explainer)
plot(fc)

fc %>% ceteris_paribus_cutoff("Male") %>% plot()

################# Regression Module #################

# Now we will try to predict the decile score of the system.
# It is not "true" value so have it in mind
# It look like this


ggplot(df[c('race', 'decile_score')], aes(x=decile_score)) +
  geom_bar() +
  facet_grid(~race)

# lets build a model to predict such scores decile scores

df_reg <- df[, ! colnames(df) %in% c('two_year_recid', 'v_decile_score')]
head(df_reg)


rf_reg <- ranger::ranger(decile_score ~., data = df_reg)

rf_reg_explainer <- DALEX::explain(rf_reg,
                                   data = df_reg,
                                   y = df$decile_score)

fairness_check(rf_reg_explainer,
               protected = df$race,
               privileged = "Caucasian")


fairness_check_regression(rf_reg_explainer,
                          protected = df$race,
                          privileged = "Caucasian")

fobject_reg <- fairness_check_regression(rf_reg_explainer,
                                         protected = df$race,
                                         privileged = "Caucasian")

plot(fobject_reg)












