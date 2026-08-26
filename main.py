import numpy as np

class Layer():
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.inputs = None
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        raise NotImplementedError

    def backward(self, output_grads: np.ndarray) -> np.ndarray:

        raise NotImplementedError

    def clear_grads(self):
        for key in self.grads:
            self.grads[key] = np.zeros_like(self.grads[key])


# создаём полносвязный класс слоя Dense который будет комбинировать признаки
class Dense(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # Используем метод инициализации весом Каиминга Хе
        he_scale = np.sqrt(2/input_dim)


        # Матрица весов W размерностью по входам (строки) и выходам (столбцы)
        self.params['W'] = np.random.randn(input_dim, output_dim) * he_scale

        # Смещение весов только в виде вектора по выходам
        self.params['B'] = np.zeros((1, output_dim))

        # Сразу готовим место под градиенты такого же размера
        self.grads['W'] = np.zeros_like(self.params['W'])
        self.grads['B'] = np.zeros_like(self.params['B'])

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        X = self.inputs @ self.params['W'] + self.params['B']
        return X

    def backward(self, output_grads: np.ndarray) -> np.ndarray:
        self.grads['W'] = self.inputs.T @ output_grads
        self.grads['B'] = np.sum(output_grads, axis=0, keepdims=True)
        input_grads = output_grads @ self.params['W'].T
        return input_grads


# Слой функции активации ReLU
class ReLU(Layer):
    def __init__(self):
        super().__init__()


    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs

        # ReLu функция
        return np.maximum(0, inputs)
        
    def backward(self, output_grads: np.ndarray) -> np.ndarray:
        return output_grads * (self.inputs > 0)


# Функция потерь на базе метрики по нахождению среднеквадратичной ошибки
class MSELoss:
    def forward(self, y_prediction: np.ndarray, y_true: np.ndarray) -> float:
        loss = np.mean(0.5 * (y_prediction - y_true)**2)
        return loss
    def backward(self, y_prediction: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        num_el = y_prediction.size
        loss_grad = (y_prediction - y_true) / num_el
        return loss_grad

# Функция оптимизатор
class SGD:
    def __init__(self, layers: list[Layer], learning_rate: float = 0.01):
        self.layers = layers
        self.lr = learning_rate

    def step(self):
        for l in self.layers:
            if hasattr(l, 'params') and l.params: 
                l.params['W'] -= self.lr * l.grads['W']
                l.params['B'] -= self.lr * l.grads['B']
            l.clear_grads()


# Тестовое обучение
x_train = np.random.randn(100, 4)
y_train = np.random.randn(100, 1)

layers = [
    Dense(input_dim=4, output_dim=8),
    ReLU(),
    Dense(input_dim=8, output_dim=1)
]
loss_fn = MSELoss()
optimizer = SGD(layers, learning_rate=0.1)

print("Начало обучения:")
for epoch in range(10001):
    out = x_train
    for layer in layers:
        out = layer.forward(out)
        
    loss = loss_fn.forward(out, y_train)
    
    grad = loss_fn.backward(out, y_train)
    for layer in reversed(layers):
        grad = layer.backward(grad)
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Эпоха {epoch:3d} | Ошибка: {loss:.6f}")