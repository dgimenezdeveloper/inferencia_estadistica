# Archivo de modelos y entrenamiento para la app integrada

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# LDA

def entrenar_lda(X, y, store_covariance=True):
    model = LinearDiscriminantAnalysis(store_covariance=store_covariance)
    model.fit(X, y)
    return model

def predecir_lda(model, X):
    return model.predict(X), model.predict_proba(X) if hasattr(model, 'predict_proba') else None

# QDA

def entrenar_qda(X, y, store_covariance=True):
    model = QuadraticDiscriminantAnalysis(store_covariance=store_covariance)
    model.fit(X, y)
    return model

def predecir_qda(model, X):
    return model.predict(X), model.predict_proba(X) if hasattr(model, 'predict_proba') else None

# Bayes Ingenuo

def entrenar_bayes(X, y):
    model = GaussianNB()
    model.fit(X, y)
    return model

def predecir_bayes(model, X):
    return model.predict(X), model.predict_proba(X) if hasattr(model, 'predict_proba') else None

# PCA (solo para proyección, no para clasificación)
def entrenar_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_proj = pca.fit_transform(X)
    return pca, X_proj


def predecir(model, X):
    """Devuelve (y_pred, y_prob) manejando ausencia de predict_proba."""
    y_pred = model.predict(X)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X)
        except Exception:
            y_prob = None
    return y_pred, y_prob
