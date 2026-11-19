library(nnet)
library(MASS)
library(caret)
library(ggplot2)
library(NeuralNetTools)
library(gganimate)
library(gifski)


df <- read.csv("D:\\data 477\\three potentisl data sets\\three potentisl data sets\\WA_Fn-UseC_-Telco-Customer-Churn - Copy.csv",
               stringsAsFactors = TRUE)

df$TotalCharges <- as.numeric(as.character(df$TotalCharges))  # 11 blanks become NA
df <- na.omit(df)                                  # drop the 11 rows with missing TotalCharges
df$target <- factor(ifelse(df$Churn == "Yes", 1, 0), levels = c(0, 1))  # 1 = churned
df$Churn <- NULL                                   # remove the answer column
df$customerID <- NULL                              # remove the ID column

set.seed(11072026)
train_index <- createDataPartition(df$target, p = 0.7, list = FALSE)
train   <- df[train_index, ]
holdout <- df[-train_index, ]

val_idx    <- createDataPartition(holdout$target, p = 0.5, list = FALSE)
Validation <- holdout[val_idx, ]
Test       <- holdout[-val_idx, ]

#SIZES OF each DS
sizes <- data.frame(
    set = c("Training","Validation","Test"),
    rows = c(nrow(train), nrow(Validation),nrow(Test)),
    percent = round(100* c(nrow(train), nrow(Validation),nrow(Test))/nrow(df),1)




)
print(sizes)

#fit LDA MODEl
 fit_lda <- lda(target ~ ., data = train)
#fit QDA model
fit_qda <- qda(target ~ SeniorCitizen + tenure + MonthlyCharges + TotalCharges, data = train)




#VAlidation Acccuracy
val_pred <- predict(fit_lda, newdata = Validation)$class
val_pred_qda <- predict(fit_qda, newdata = Validation)$class

#Test Accuracy
test_pred <- predict(fit_lda, newdata = Test)$class
test_pred_qda <- predict(fit_qda, newdata = Test)$class

#Confusion matrix (shows how many churners it actually caught)
confusionMatrix(val_pred_qda, Validation$target, positive = "1")


#Neural Network Model
pp <- preProcess(train, method = c("center", "scale"))
train_scaled <- predict(pp, train)
validation_scaled <- predict(pp, Validation)
test_scaled <- predict(pp, Test)

fit_nn <- nnet(target ~ ., data = train_scaled, 
size = 5, 
maxit = 200, 
trace = FALSE)


val_pred_nn  <- factor(predict(fit_nn, newdata = validation_scaled,  type = "class"), levels = levels(Validation$target))
test_pred_nn <- factor(predict(fit_nn, newdata = test_scaled, type = "class"), levels = levels(Test$target))













results <- data.frame(
  model      = c("LDA", "QDA", "NN"),
  validation = c(mean(val_pred == Validation$target),  
                mean(val_pred_qda == Validation$target),
                mean(val_pred_nn == Validation$target)),
  test       = c(mean(test_pred == Test$target),        
                mean(test_pred_qda == Test$target),
                mean(test_pred_nn == Test$target))
)
print(results)


#Nueral net animation graphs
step    <- 5    # iterations per snapshot
n_steps <- 40   # 40 x 5 = 200 iterations total

y_train <- as.numeric(as.character(train_scaled$target))
y_val   <- as.numeric(as.character(validation_scaled$target))
logloss <- function(p, y) { p <- pmin(pmax(p, 1e-7), 1 - 1e-7); -mean(y*log(p) + (1-y)*log(1-p)) }

set.seed(11142026)
fit <- nnet(target ~ ., data = train_scaled, size = 5, maxit = 0, trace = FALSE)  # random starting weights

curve_list <- list(); prob_list <- list(); wt_list <- list()

for (i in 1:n_steps) {
  # keep training from where it left off
  fit <- nnet(target ~ ., data = train_scaled, size = 5, maxit = step,
              Wts = fit$wts, trace = FALSE)
  it <- i * step

  p_tr <- as.numeric(predict(fit, train_scaled,      type = "raw"))
  p_va <- as.numeric(predict(fit, validation_scaled, type = "raw"))

  # error + how many churners it catches
  curve_list[[i]] <- data.frame(
    iter   = it,
    set    = c("Training", "Validation"),
    loss   = c(logloss(p_tr, y_train), logloss(p_va, y_val)),
    recall = c(mean(p_tr[y_train == 1] > 0.5), mean(p_va[y_val == 1] > 0.5))
  )

  # predicted probabilities on validation
  prob_list[[i]] <- data.frame(iter = it, prob = p_va, actual = factor(y_val))

  # input -> hidden weights
  n_in <- fit$n[1]; h <- fit$n[2]
  W <- matrix(fit$wts[1:(h * (n_in + 1))], nrow = n_in + 1)
  wt_list[[i]] <- data.frame(
    iter   = it,
    input  = rep(c("bias", fit$coefnames), times = h),
    hidden = rep(paste0("H", 1:h), each = n_in + 1),
    weight = as.vector(W)
  )
}

curve_df <- do.call(rbind, curve_list)
prob_df  <- do.call(rbind, prob_list)
wt_df    <- do.call(rbind, wt_list)

# Animation 1: learning curve drawing itself
a1 <- ggplot(curve_df, aes(iter, loss, colour = set)) +
  geom_line(linewidth = 1) + geom_point() +
  labs(title = "Neural net learning curve", x = "Iteration", y = "Log loss", colour = NULL) +
  transition_reveal(iter)
animate(a1, nframes = 100, fps = 10, renderer = gifski_renderer())
anim_save("d:/data 477/nn_learning_curve.gif")

# Animation 2: predicted probabilities shifting as it learns
a2 <- ggplot(prob_df, aes(prob, fill = actual)) +
  geom_histogram(bins = 30, boundary = 0) +
  facet_wrap(~ actual, ncol = 1, scales = "free_y",
             labeller = labeller(actual = c(`0` = "Stayed", `1` = "Churned"))) +
  geom_vline(xintercept = 0.5, linetype = "dashed") +
  labs(title = "Predicted P(churn) - iteration {closest_state}", x = "P(churn)", y = "Count") +
  theme(legend.position = "none") +
  transition_states(iter, transition_length = 1, state_length = 0)
animate(a2, nframes = 120, fps = 10, renderer = gifski_renderer())
anim_save("d:/data 477/nn_probabilities.gif")

# Animation 3: the weights changing (the actual "brain")
a3 <- ggplot(wt_df, aes(hidden, input, fill = weight)) +
  geom_tile() +
  scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b") +
  labs(title = "Input -> hidden weights - iteration {closest_state}", x = "Hidden node", y = NULL) +
  transition_states(iter, transition_length = 1, state_length = 0)
animate(a3, nframes = 120, fps = 10, renderer = gifski_renderer())
anim_save("d:/data 477/nn_weights.gif")