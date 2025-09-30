vinos <- read.csv2("Wine.csv",dec=".")


"Regresión Lineal Multiple "
#Muestra los campos que son significativos estadisticamente para nuestra variable respuesta
modelo_lineal <- lm(vinos$quality~.,data=vinos) 
summary(modelo_lineal)

par(mfrow = c(2, 2))
plot(modelo_lineal)

#Calculo LASSO
library(glmnet)

x <- as.matrix(vinos[, -which(names(vinos) == "quality")])  # Todas las variables predictoras/explicativas
y <- vinos$quality                                           # Variable respuesta

# Dividir en train/test (ejemplo: 80% train, 20% test)
set.seed(123)
train <- sample(1:nrow(vinos), 0.8 * nrow(vinos))
test <- setdiff(1:nrow(vinos), train)

x_train <- x[train, ]
y_train <- y[train]
x_test <- x[test, ]
y_test <- y[test]

#Definir grilla de lambda (misma que en tu ejemplo)
grilla <- 10^seq(10, -2, length = 100)

# Ajustar modelos Ridge y LASSO en la grilla
ridge.mod <- glmnet(x[train,], y[train], alpha = 0, lambda = grilla)
lasso.mod <- glmnet(x[train,], y[train], alpha = 1, lambda = grilla)

# Validación cruzada para elegir mejor lambda
set.seed(123)
cv.ridge <- cv.glmnet(x[train,], y[train], alpha = 0, lambda = grilla)
cv.lasso <- cv.glmnet(x[train,], y[train], alpha = 1, lambda = grilla)

# Lambda que minimiza error de validación
best.lambda.ridge <- cv.ridge$lambda.min
best.lambda.lasso <- cv.lasso$lambda.min

best.lambda.ridge
best.lambda.lasso

# Coeficientes finales en el lambda óptimo
coef(cv.ridge, s = "lambda.min")
coef(cv.lasso, s = "lambda.min")

# Evaluar error de predicción en test (opcional)
pred.ridge <- predict(cv.ridge, s = best.lambda.ridge, newx = x[test,])
pred.lasso <- predict(cv.lasso, s = best.lambda.lasso, newx = x[test,])

mean((y[test] - pred.ridge)^2)  # MSE Ridge
mean((y[test] - pred.lasso)^2)  # MSE Lasso



# Gráficos de trayectorias de coeficientes
plot(ridge.mod , xvar = "lambda", label = TRUE,
     main = "Trayectoria de coeficientes - Ridge")

plot(lasso.mod, xvar = "lambda", label = TRUE,
     main = "Trayectoria de coeficientes - Lasso")
ahb-oowv-twg