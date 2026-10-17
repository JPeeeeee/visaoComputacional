# Trabalho 5: Depuração, Experimentação e Análise de MLP from Scratch

Este repositório contém o código, o notebook interativo e o relatório do Trabalho 5 da disciplina de **Visão Computacional / Redes Neurais**, focado no estudo aprofundado, depuração matemática e otimização de uma **Rede Neural Multi-Layer Perceptron (MLP) construída do zero (from scratch)** utilizando apenas NumPy no [Iris Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/iris).

---

## 🎯 Objetivos e Diagnóstico dos Problemas

O código original disponibilizado apresentava falhas críticas que impediam o aprendizado da terceira classe (*Iris-Virginica*, que obtinha 0% de acurácia):
1. **`predict` sem função de ativação**: O método realizava combinações puramente lineares sem aplicar a não linearidade treinada.
2. **Atualização prematura de pesos na retropropagação**: A matriz de pesos de saída era alterada antes do cálculo do erro da camada oculta.
3. **Múltiplas atualizações de bias em loops aninhados**: O bias de saída era atualizado 5 vezes por amostra e o oculto 4 vezes por amostra.
4. **Particionamento desbalanceado**: O conjunto de teste continha apenas 1 amostra de Virginica.
5. **Saturação de sigmoide por falta de normalização**: Valores de entrada entre 0.1 e 7.9 cm causavam gradientes nulos (*vanishing gradient*).

### 💡 Solução Implementada
- **Backpropagation corrigido**: Gradientes exatos da regra da cadeia para pesos e bias.
- **Função Softmax + Cross-Entropy**: Formulação canônica para problemas multiclasse com gradientes estáveis.
- **Normalização Z-Score e Estratificação**: 80% treino e 20% teste com 10 amostras equilibradas por classe.
- **Acurácia resultante**: **96,67%** no teste (100% Setosa, 90% Versicolor, 100% Virginica).

---

## 🚀 Como Executar

```bash
cd T5
pip install -r requirements.txt

# Executar todos os experimentos e gerar as figuras:
python main.py
```

Para abrir o notebook interativo:
```bash
jupyter notebook NN_scratch_iris_resolvido.ipynb
```

---

## 📁 Estrutura de Diretórios

```text
T5/
├── NN_scratch_iris_original.ipynb    # Cópia do notebook original com os bugs
├── NN_scratch_iris_resolvido.ipynb   # Notebook completo depurado e comentado
├── output/                           # Gráficos em alta resolução e relatórios
│   ├── acuracia_por_classe.png
│   ├── analise_overfitting.png
│   ├── convergencia_perda_treino_val.png
│   ├── fronteiras_decisao_mlp.png
│   ├── matriz_confusao_3classes.png
│   └── relatorio_experimentos_mlp.txt
├── main.py                           # Script de reprodução e experimentação
├── relatorio.tex                     # Artigo científico sucinto em LaTeX
├── requirements.txt                  # Dependências do projeto
└── README.md                         # Documentação completa
```

---

## 🔗 Repositório Git

- [https://github.com/JPeeeeee/visaoComputacional](https://github.com/JPeeeeee/visaoComputacional)
