import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Carregar dados de exemplo
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/diamonds.csv"
data = pd.read_csv(url)

# Exibir as primeiras linhas dos dados
print(data.head())

# Selecionando colunas relevantes para a regressão
X = data[['carat', 'depth', 'table', 'x', 'y', 'z']]  # Variáveis independentes
y = data['price']  # Variável dependente

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criação do modelo de regressão linear
model = LinearRegression()

# Treinando o modelo
model.fit(X_train, y_train)

# Realizando previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calculando o erro médio quadrático
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Médio Quadrático (MSE): {mse}")

# Salvando o modelo treinado
joblib.dump(model, 'modelo_regressao_linear.pkl')

# Exibindo a previsão de exemplo
print(f"Primeiras previsões: {y_pred[:5]}")
