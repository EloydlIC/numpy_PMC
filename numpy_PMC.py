import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class PMC(BaseEstimator, ClassifierMixin):
    """
    Perceptrón Multicapa (MLP) implementado desde cero con NumPy.
    Compatible con la API de scikit-learn para GridSearchCV, pipelines, etc.

    Parámetros:
    -----------
    layers : lista de tuplas
        Arquitectura de la red: [(neuronas_salida_capa1, neuronas_entrada_capa1), ...]
        Ejemplo: [(100, 784), (10, 100)] = 2 capas: 784->100->10
    iterations : int
        Número de épocas/iteraciones de entrenamiento
    alpha : float
        Tasa de aprendizaje (learning rate) para descenso de gradiente
    initial_params : tupla (weights, bias) o None
        Pesos pre-entrenados para continuar entrenamiento o inferencia
    show : bool
        Mostrar progreso durante entrenamiento
    show_iter : int
        Frecuencia de impresión de accuracy durante entrenamiento
    """

    def __init__(self,
                 layers=[(10, 784), (10, 10)],
                 iterations: int = 1000,
                 alpha: float = 0.1,
                 initial_params=None,
                 show: bool = True,
                 show_iter: int = 10):
        self.layers = layers
        self.n_layers = len(layers)
        self.alpha = alpha
        self.iterations = iterations
        self.show = show
        self.show_iter = show_iter
        self._initial_params = initial_params

        # Inicializar pesos si no se proporcionan
        if initial_params is not None:
            self.weights, self.bias = initial_params[0], initial_params[1]
        else:
            self.weights, self.bias = [], []

    def init_params(self):
        """
        Inicialización de pesos usando He Initialization (He et al., 2015).
        Óptima para redes con activación ReLU: pesos ~ N(0, sqrt(2/n_entradas))
        Los sesgos se inicializan a cero.
        """
        weights = []
        bias = []
        for elem in self.layers:
            # elem = (n_salida, n_entrada)
            # He init: std = sqrt(2 / n_entradas)
            w = np.random.randn(*elem) * np.sqrt(2.0 / elem[1])
            b = np.zeros((elem[0], 1))  # Sesgo: (n_salida, 1)
            weights.append(w)
            bias.append(b)
        return weights, bias

    def forward_propagation(self, X: np.ndarray):
        """
        Propagación hacia adelante (forward pass).
        
        Entrada X: (n_muestras, n_características) -> Transpuesta a (n_características, n_muestras)
        Para cada capa: Z = W @ A_prev + b, luego A = activación(Z)
        Capas ocultas: ReLU, Capa de salida: Softmax
        
        Salidas: Lista de tuplas [(Z1, A1), (Z2, A2), ...] para backpropagation
        """
        # X entra como (n_muestras, n_features), transponemos a (n_features, n_muestras)
        X = X.T
        forward = []  # Almacena (Z, A) de cada capa

        for i in range(self.n_layers):
            if i == 0:
                # Primera capa: entrada viene de X
                Z = self.weights[i] @ X + self.bias[i]
            else:
                # Capas siguientes: entrada viene de activación anterior
                Z = self.weights[i] @ forward[i-1][1] + self.bias[i]

            # Aplicar activación: Softmax en última capa, ReLU en ocultas
            if i == self.n_layers - 1:
                A = softmax(Z)  # Capa de salida: probabilidades
            else:
                A = ReLU(Z)     # Capa oculta: no-linealidad

            forward.append((Z, A))

        return forward

    def back_propagation(self, forward, X: np.ndarray, Y: np.ndarray):
        """
        Retropropagación (backpropagation) - cálculo de gradientes.
        
        Usa la regla de la cadena para calcular dL/dW y dL/db para cada capa.
        Función de pérdida: Entropía cruzada categórica.
        Derivada de CE + Softmax: dL/dZ = A - Y_one_hot
        
        Retorna: Lista de gradientes [(dW1, db1), (dW2, db2), ...] en orden directo
        """
        X = X.T  # (n_features, n_muestras)
        _, m = X.shape  # m = número de muestras
        one_hot_Y = one_hot(Y)  # (10, n_muestras)

        diffe = []  # Gradientes de cada capa

        # Gradiente inicial: dL/dZ_output = A_output - Y_one_hot
        # (Derivada de Cross-Entropy + Softmax combinadas)
        dZ = forward[-1][1] - one_hot_Y

        # Recorremos capas en orden inverso (de salida a entrada)
        for i in range(self.n_layers - 1, -1, -1):
            if i == self.n_layers - 1:
                # Capa de salida: dZ ya calculado arriba
                pass
            else:
                # Capa oculta: dZ = W_{i+1}^T @ dZ_{i+1} * dReLU(Z_i)
                # Regla de la cadena: gradiente fluye hacia atrás a través de pesos
                # y se multiplica por derivada de activación (ReLU)
                dZ = (self.weights[i+1].T) @ dZ * dReLU(forward[i][0])

            # Gradientes de pesos y sesgos para esta capa
            A_prev = forward[i-1][1] if i > 0 else X
            dW = (1.0 / m) * dZ @ A_prev.T
            db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
            diffe.append((dW, db))

        # Invertimos para que estén en orden directo (capa 1, capa 2, ...)
        diffe = diffe[::-1]
        return diffe

    def update_params(self, diffe):
        """
        Actualización de parámetros con Descenso de Gradiente Simple (SGD).
        W_new = W_old - alpha * dW
        b_new = b_old - alpha * db
        """
        for i in range(self.n_layers):
            self.weights[i] -= self.alpha * diffe[i][0]
            self.bias[i] -= self.alpha * diffe[i][1]

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Entrenamiento completo del modelo.
        Ejecuta iterativamente: forward -> backward -> update
        Muestra accuracy cada show_iter iteraciones si show=True
        """
        # Inicializar pesos si no existen
        if self._initial_params:
            self.weights, self.bias = self._initial_params[0], self._initial_params[1]
        else:
            self.weights, self.bias = self.init_params()

        accuracy_vector = []

        for i in range(self.iterations):
            # 1. Forward propagation
            forward = self.forward_propagation(X)

            # 2. Backpropagation
            diffe = self.back_propagation(forward, X, Y)

            # 3. Actualizar parámetros
            self.update_params(diffe)

            # 4. Mostrar progreso
            if self.show and (i % self.show_iter == 0):
                print(f"Iteración: {i}")
                predictions = self.get_predictions(forward[-1][1])
                acc = self.get_accuracy(predictions, Y)
                accuracy_vector.append(acc)
                print(f"Accuracy: {100 * acc:.2f}%\n")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicción de clases para datos de entrada.
        Hace forward pass y toma argmax de la capa de salida.
        """
        forward = self.forward_propagation(X)
        predictions = self.get_predictions(forward[-1][1])
        return predictions

    def get_predictions(self, A2: np.ndarray) -> np.ndarray:
        """
        Extrae la clase predicha: índice del máximo por columna (muestra).
        A2 tiene forma (10, n_muestras), argmax(axis=0) da (n_muestras,)
        """
        return np.argmax(A2, axis=0)

    def get_accuracy(self, predictions: np.ndarray, Y: np.ndarray) -> float:
        """
        Calcula accuracy: proporción de predicciones correctas.
        """
        return np.sum(predictions == Y) / Y.size

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna probabilidades de clase (output de softmax).
        Requerido para compatibilidad con scikit-learn (ej. ROC AUC).
        """
        forward = self.forward_propagation(X)
        return forward[-1][1]  # (10, n_muestras)

    # --- Métodos para compatibilidad con scikit-learn ---

    def get_params(self, deep=True):
        """Devuelve parámetros para clonación y GridSearch."""
        return {
            'layers': self.layers,
            'iterations': self.iterations,
            'alpha': self.alpha,
            'show': self.show,
            'show_iter': self.show_iter,
            'initial_params': self._initial_params
        }

    def set_params(self, **parameters):
        """Establece parámetros para GridSearch."""
        for param, value in parameters.items():
            if hasattr(self, param):
                setattr(self, param, value)
        if hasattr(self, 'layers'):
            self.n_layers = len(self.layers)
        return self

if __name__ == "__main__":
    # Prueba rápida del modelo con datos sintéticos
    np.random.seed(42)
    X_test = np.random.rand(100, 784)  # 100 muestras, 784 características
    Y_test = np.random.randint(0, 10, size=100)  # 100 etiquetas aleatorias (0-9)

    model = PMC(layers=[(50, 784), (20, 50), (10, 20)], iterations=10, alpha=0.1)
    model.fit(X_test, Y_test)
    predictions = model.predict(X_test)
    accuracy = model.get_accuracy(predictions, Y_test)
    print(f"Accuracy final en datos de prueba: {accuracy * 100:.2f}%")