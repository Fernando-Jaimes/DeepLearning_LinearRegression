import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 1. Cargar el dataset desde el PDF
df = pd.read_csv('Diagnostico(datos)(1).csv', delimiter=' ')  # Convertir el PDF a CSV antes

# 2. Preprocesamiento: Ajustar a valores numéricos (sin normalización ni estandarización)
df['GENERO'] = df['GENERO'].map({'MASCULINO': 1, 'FEMENINO': 0})
df = df.replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# 3. Seleccionar características de entrada y etiqueta
X = df.drop(columns=['EVOLUCION']).values
y = df['EVOLUCION'].values.reshape(-1, 1)

# 4. División del dataset (80/20) sin aleatorización
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Convertir a tensores de PyTorch
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

# 5. Definir el modelo de regresión lineal
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(X_train.shape[1])

# 6. Hiperparámetros
learning_rate = 0.01
epochs = 100
batch_size = 16
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 7. Entrenamiento
def train_model(model, X_train, y_train, epochs, batch_size):
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

train_model(model, X_train, y_train, epochs, batch_size)

# 8. Evaluación
y_pred = model(X_test).detach().numpy()
loss = loss_function(torch.tensor(y_pred), y_test).item()
print(f'Final Test Loss: {loss:.4f}')

# Guardar el modelo
torch.save(model.state_dict(), 'linear_regression_model.pth')
