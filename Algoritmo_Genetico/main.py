#importando libs

import numpy as np
import pandas as pd
import random as rd
from random import randint
import matplotlib.pyplot as plt

# numero de items - 10

n = 10
numero_items = np.arange(1, n+1)

#Gerar os pesos

pesos = [2.5, 1.8,0.7,2.1,1.5,2.2,0.9,1.6,0.5,1.1]

#Grerando valores de cada item

valores = [2000, 1450,3400,1900,1300,1000,600,1300,400,900]

nomes = ['Smartphone Samsung Galaxy S21',
         'Notebook Dell Inspiron 15',
         'Fone de Ouvido Bluetooth JBL',
         'Smartwatch Samsung Galaxy Watch 3',
         'Tablet Apple iPad 10.2',
         'Câmera Digital Canon EOS Rebel T7',
         'Mouse Gamer Logitech G Pro',
         'Teclado Mecânico Redragon Kumara',
         'Caixa de Som Bluetooth JBL GO',
         'Smartband Xiaomi Mi Band 6']

# definindo o peso máximo para mochila.
max_peso_mochila = 7

for i in range(numero_items.shape[0]):
    print('Item: {} \nPeso(Kg): {} \nValor(R$): {} \n'.format(nomes[i], pesos[i], valores[i]))


#Número de soluções ou indivíduos por população
solucao_por_populacao = 8
tamanho_populacao = (solucao_por_populacao, numero_items.shape[0])

print('Tamanho da população = {}'.format(tamanho_populacao))
print('Número de indivíduos (solução) = {}'.format(tamanho_populacao[0]))
print('Número itens (genes) = {}'.format(tamanho_populacao[1]))


#Definir numero de geracoes
n_geracoes = 10


#Criando a populacao onde somente um item sera levado por individuo
populacao_inical = np.eye(tamanho_populacao[0], tamanho_populacao[1], k=0)

#Converendo os tipos dos genes para inteiro
populacao_inical = populacao_inical.astype(int)

print('População Inicial: \n{}'.format(populacao_inical))

# Funcao calcular fitness de cada individuo
def cal_fitness(peso, valor, populacao, max_peso_mochila):
    fitness = np.zeros(populacao.shape[0])
    
    #Percorer cada individuo
    for i in range(populacao.shape[0]):

        #Multiplica os itens que o individuo esta levando pelo valor e soma
        S1 = np.sum(populacao[i] * valor)

        #Multiplica os itens que o individuo ta levando pleo peso e soma
        S2 = np.sum(populacao[i] * peso)

        #Verifica se o valor não passou da capacidade maxima da mochila
        if S2 <= max_peso_mochila:
            #Armazena o fitness do indiviuo
            fitness[i] = S1

        else:
            #Passou da capacidade maxima logo 0
            fitness[i] = 0

    return fitness.astype(float)

#Função para a seleção dos individuos
def selecao_roleta(fitness, numero_pais, populacao):

    #Soma todos os fitness
    max_fitness = sum(fitness)

    #Calcula a probabilidade de cada um
    probabilidade = fitness/max_fitness

    #Realizar a selecao com base nas probabilidades
    selecionados = populacao[np.random.choice(len(populacao),size=numero_pais, p=probabilidade)]

    return selecionados

def crossover(pais, numero_filhos):
    filhos = np.zeros((numero_filhos, pais.shape[1]))

    #O ponto em que o cruzamento ocorre entre dois pais.
    pronto_crossover = int(pais.shape[1]/2)
    for k in range(numero_filhos):
        #Indice do primeiro a ser fatiado
        pai_1_idx = k%pais.shape[0]
        #Indice do segundo a ser fatiado
        pai_2_idx = (k+1)%pais.shape[0]
        #A nova prole tera sua primeira metade de seus genes retirados do primeiro pai
        filhos[k, 0:pronto_crossover] = pais[pai_1_idx, 0:pronto_crossover]
        #A nova prole tera sua segunda metade de seus genes retirados do segundo pai
        filhos[k, pronto_crossover] = pais[pai_2_idx, pronto_crossover]

    return filhos

def mutacao(filhos):
    #Cria um vetor para armazenar os individuos mutados
    mutacao = filhos
    #Percorer todos os filhos
    for i in range(mutacao.shape[0]):
        #Pega aleatorio (posição) um gene do filho
        posicao_gene = randint(0, filhos.shape[1] - 1)
        #Se aquele gene é 0 muda para 1
        if mutacao[i, posicao_gene] == 0:
            mutacao [i, posicao_gene] = 1
        else:
            mutacao[i, posicao_gene] = 0

    return mutacao

def rodar_AG(pesos, valores, populacao, tamanho_populacao, n_geracoes, max_peso_mochila):
    #Criando variáveis para parametros, histórico de fitness
    historico_fitness, historico_populacao = [], []
    
    # Calculando o número de pais.
    numero_pais = int(tamanho_populacao[0] / 2)
    
    # Calculando o número de filhos
    numero_filhos = tamanho_populacao[0] - numero_pais
    
    fitness = []
    
    # Repetição até o número de gerações setado
    for i in range(n_geracoes):
        
        print('--- Começando a Geração {} ---'.format(i))
        
        # Calcula o fitness (aptidão) de cada individuo
        fitness = cal_fitness(pesos, valores, populacao, max_peso_mochila)
        
        # Armazena na variável de histórico
        historico_fitness.append(fitness.copy())
        historico_populacao.append(populacao.copy())
        
        # Pais selecionados
        pais = selecao_roleta(fitness, numero_pais, populacao)
        
        # Gerando os filhos
        filhos = crossover(pais, numero_filhos)
        
        # Mutando os filhos
        filhos_mutados = mutacao(filhos)
        
        print('População Antiga:')
        print(populacao)
        
        populacao[0:pais.shape[0], :] = pais
        populacao[pais.shape[0]:, :] = filhos_mutados
        
        print('População Nova:')
        print(populacao)
    
    return historico_populacao, historico_fitness


historico_populacao, historico_fitness = rodar_AG(pesos, valores, populacao_inical, tamanho_populacao, n_geracoes, max_peso_mochila)

#Criando o dataframe de historico(Gerações x Fitness Individuo)

dataFrame = pd.DataFrame(historico_fitness)

# Apresenta resultado

print(dataFrame)

# Encontra o melhor individuo
# Ou seja, dentro de todas as gerações qual foi o melhor individuo com maior fitness
#Encontra a linha e coluna

max_index = dataFrame.values.argmax()
linha, coluna = np.unravel_index(max_index, dataFrame.shape)

print("Valor do Fitness (Max): ", dataFrame.iloc[linha,coluna])
print("Linha maior Fitness (Geração): " , linha)
print("Coluna do maior Fitness (individuo): ", coluna)

#Armazena o melhor individuo
melhor_individuo = historico_populacao[linha][coluna]

#Criando um dataframe apenas com itens 'pegos'
itens_selecionados = numero_items * melhor_individuo
dataFrame_itens = pd.DataFrame(columns=['Item', 'Valor', 'Peso'])
#Percorre todos os itens
for i in itens_selecionados:
    if i != 0:
        posicao = i -1
        item = {'Item':  nomes[posicao], 'Valor': valores[posicao], 'Peso': pesos[posicao]}
        dataFrame_itens.loc[len(dataFrame_itens)] = item

print(dataFrame_itens)

fitness_medio = [np.mean(fitness) for fitness in historico_fitness]
fitness_max = [np.max(fitness) for fitness in historico_fitness]

plt.plot(list(range(n_geracoes)), fitness_medio, label = 'Fitness medio')
plt.plot(list(range(n_geracoes)), fitness_max, label = 'Fitness maximo')
plt.legend()
plt.title('Fitness ao decorer das gerações')
plt.xlabel('Geração')
plt.ylabel('Fitness')
plt.show()
print(np.asanyarray(historico_fitness).shape)
