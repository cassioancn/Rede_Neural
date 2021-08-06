#-----------------------------------------------------
# Destao da Informacao Empresarial
# Redes Neurais Artificiais
# Problema: Doenca cardiaca 
# Prof. Leandro Gauss
#-----------------------------------------------------
# Instalar e carregar pacotes
#------------------------------------------------------

# Instalar pacotes
#install.packages("readxl")                # ler arquivos do excell
#install.packages("neuralnet")             # criar redes neurais artificiais (RNA)
#install.packages("caret")                 # criar matriz de confusao

# Carregar pacotes
library(readxl)                            # trabalha com arquivos em excell
library(neuralnet)                         # trabalha com RNAs
library(caret)                             # trabalha com matriz de confusao

#------------------------------------------------------
# Carregar dataset
#------------------------------------------------------

# Ler arquivos em excel
dados = read_xlsx(path = "doenca_cardiaca.xlsx")

#------------------------------------------------------
# Carregar base de treinamento (66,7%) e teste (33,3%)
#------------------------------------------------------

n = nrow(dados)                            # retorna o numero de linhas do dataset
n_treino = round(n * 2/3)                  # define o tamanho da base de teste
i_treino = sample(1:n, n_treino)           # cria uma mostra com o tamanho da base de treinamento
treino = dados[i_treino,]                  # define a base de treinamento
teste = dados[-i_treino,]                  # define a base de teste

#------------------------------------------------------
# Treinar a RNA
#------------------------------------------------------

set.seed(1250)                             # define a semente de geracao de numeros aleatorios

nn = neuralnet(heart_disease ~             # variavel dependente
                 .,                        # variaveis independentes (. = todas as demais)
               data = treino,              # data set para treinamento da RNA
               hidden = c(5),              # camada de neuronios ocultos
               algorithm = 'rprop+',       # algoritmo para determinacao dos pesos
               linear.output = F,          # tipo de saida, T = reg e F = class
               threshold = 0.01)           # definie limiar para parar o processo de otimizacao

plot(nn)                                   # gera graficamente a RNA

#------------------------------------------------------
# Testar a RNA
#------------------------------------------------------

nnp = predict(nn, teste)                   # faz previsao com base nos dados de teste

x = factor(ifelse(nnp >= 0.5, 1, 0))       # converte resultado da predicao em binario
y = factor(teste$heart_disease)            # seleciona a coluna de comparacao da base de teste
confusionMatrix(x,y,positive = "1")        # gera a matriz de confusao

#------------------------------------------------------
