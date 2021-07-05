library("DALEX")
library("modelStudio")
library("arenar")
library("tidyr")
library("ranger")
library("rpart")
library("gbm")
library("gam")
library("e1071")
library("earth") 

# load data
train <- read.csv("data/happiness_train.csv", row.names = 1)
test <- read.csv("data/happiness_test.csv",  row.names = 1)

# fit models
model_rf <- ranger(score~., data = train)
model_gbm <- gbm(score~., data = train, interaction.depth = 2)
model_gam <- gam(score ~ 
                   s(gdp_per_capita) +
                   s(social_support) +
                   s(healthy_life_expectancy) +
                   s(freedom_life_choices) +
                   s(generosity) +
                   s(perceptions_of_corruption),
                 data = train)
model_svm <- svm(score~., data = train)
model_dt  <- rpart(score~., data = train)
model_mars <- earth(score~., data = train)

# create explainers for the models
explainer_rf <- explain(model_rf,
                        data = test[,-1],
                        y = test$score)
explainer_gbm <- explain(model_gbm,
                         data = test[,-1],
                         y = test$score)
explainer_gam <- explain(model_gam,
                         data = test[,-1],
                         y = test$score,
                         label = "gam")
explainer_svm <- explain(model_svm,
                         data = test[,-1],
                         y = test$score)
explainer_dt  <- explain(model_dt,
                         data = test[,-1],
                         y = test$score)
explainer_mars  <- explain(model_mars,
                           data = test[,-1],
                           y = test$score)

plot(model_performance(explainer_rf),
     model_performance(explainer_gbm),
     model_performance(explainer_svm),
     model_performance(explainer_gam),
     model_performance(explainer_dt),
     model_performance(explainer_mars),
     geom = "boxplot")

plot(model_parts(explainer_rf),
     model_parts(explainer_gbm),
     model_parts(explainer_svm),
     model_parts(explainer_gam),
     model_parts(explainer_dt),
     model_parts(explainer_mars),
     bar_width = 4)

plot(model_profile(explainer_rf),
     model_profile(explainer_gbm),
     model_profile(explainer_svm),
     model_profile(explainer_gam),
     model_profile(explainer_dt),
     model_profile(explainer_mars))

# make an Arena for the models
arena <- create_arena(live=TRUE) %>%
  push_model(explainer_rf) %>%
  push_model(explainer_gbm) %>%
  push_model(explainer_svm) %>%
  push_model(explainer_gam) %>%
  push_model(explainer_dt) %>%
  push_model(explainer_mars) %>%
  push_observations(test) %>%
  push_dataset(train, "score", "train")

# explain!
run_server(arena)

# create train explainers for the models
exp_rf_train <- explain(model_rf,
                        data = train[,-1],
                        y = train$score,
                        label = "ranger-train")
exp_gam_train <- explain(model_gam,
                         data = train[,-1],
                         y = train$score,
                         label = "gam-train")
exp_svm_train <- explain(model_svm,
                         data = train[,-1],
                         y = train$score,
                         label = "svm-train")

plot(model_performance(exp_rf_train),
     model_performance(exp_gam_train),
     model_performance(exp_svm_train),
     geom = "boxplot")

plot(model_parts(exp_rf_train),
     model_parts(exp_gam_train),
     model_parts(exp_svm_train),
     bar_width = 4)

plot(model_profile(exp_rf_train),
     model_profile(exp_gam_train),
     model_profile(exp_svm_train))


# long computations for later

arena_saved <- create_arena(
  fi_B = 20, shap_B = 20,
  grid_points = 31, max_points_number = 300
) %>%
  push_model(explainer_rf) %>%
  push_model(explainer_svm) %>%
  push_model(explainer_gam) %>%
  push_model(exp_rf_train) %>%
  push_model(exp_svm_train) %>%
  push_model(exp_gam_train) %>%
  push_observations(test) %>%
  push_dataset(train, "score", "train") %>%
  push_dataset(test, "score", "test")

# save for later
save_arena(arena_saved, filename = "arena_happiness.json")
