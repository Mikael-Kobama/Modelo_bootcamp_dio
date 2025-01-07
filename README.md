# Projeto de Previsão com Regressão Linear

Este projeto utiliza a regressão linear para prever o preço dos diamantes com base em suas características.

## Passo a Passo

1. **Carregamento dos Dados**: Carregamos o conjunto de dados de diamantes do [Data Science Dojo](https://raw.githubusercontent.com/datasciencedojo/datasets/master/diamonds.csv).

2. **Seleção de Variáveis**: Para a previsão, usamos as seguintes variáveis independentes:
   - `carat`: peso do diamante
   - `depth`: profundidade do diamante
   - `table`: proporção da mesa
   - `x`, `y`, `z`: dimensões do diamante

   E a variável dependente:
   - `price`: preço do diamante

3. **Divisão dos Dados**: Os dados são divididos em conjuntos de treinamento e teste usando a função `train_test_split` do `scikit-learn`.

4. **Treinamento do Modelo**: O modelo de regressão linear é treinado no conjunto de treinamento.

5. **Avaliação**: O modelo é avaliado utilizando o Erro Médio Quadrático (MSE) para determinar a qualidade da previsão.

6. **Salvando o Modelo**: O modelo treinado é salvo utilizando o `joblib` para ser reutilizado futuramente.

## Como Rodar o Projeto

1. Clone este repositório.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
