import skfuzzy as fuzz
import skfuzzy.control as ctrl
import numpy as np

def __main__():
    # CRIANDO AS VARIAVEIS DE ENTRADA E SEUS UNIVERSOS
    peso = ctrl.Antecedent(np.arange(20,120,1), 'peso')
    altura = ctrl.Antecedent(np.arange(1.2,2,0.1), 'altura')


    # CRIANDO A VARIAVEL DE SAIDA E SEU UNIVERSO
    tamanho = ctrl.Consequent(np.arange(0,1,.1), 'tamanho')

    # CRIANDO OS CONJUNTOS FUZZY E SUAS FUNÇÕES DE PERTINENCIA PAA ENTRADA PESO

    peso['baixo'] = fuzz.trimf(peso.universe, [20, 45, 70])
    peso['medio'] = fuzz.trimf(peso.universe, [55,75,95])
    peso['alto'] = fuzz.trimf(peso.universe, [80, 100, 120])

    peso.view()

    test = input(" ")

    # CRIANDO OS CONJUNTOS FUZZY E SUAS FUNÇÕES DE PERTINENCIA PARA ENTRADA ALTURA

    altura['baixa'] = fuzz.trapmf(altura.universe,[1.2,1.2,1.4,1.5])
    altura['media'] = fuzz.trapmf(altura.universe, [1.4,1.5,1.6,1.7])
    altura['alta'] = fuzz.trapmf(altura.universe, [1.68,1.7,2, 2])

    altura.view()

    test2 = input(" ")

    # CRIANDO OS CONJUNTOS FUZZY E SUAS FUNÇÕES DE PERTINENCIA PARA SAIDA TAMANHO
    tamanho['pequeno'] = fuzz.trapmf(tamanho.universe, [0,0,0.4,0.5])
    tamanho['medio'] = fuzz.trapmf(tamanho.universe, [0.4,0.5,0.6,0.7])
    tamanho['grande'] = fuzz.trapmf(tamanho.universe, [0.65,0.7, 1,1])

    tamanho.view()

    test3 = input(" ")

    # REGRA - 1 SE PESO É BAIXO ENTÃO TAMANHO É PEQUENO

    regra1 = ctrl.Rule(peso['baixo'], tamanho['pequeno'])
    
    # REGRA - 2 SE ALTURA É MEDIANA OU PESO É MEDIO ENTÃO TAMANHO É MEDIO

    regra2 = ctrl.Rule(altura['media'] | peso['medio'], tamanho['medio'])

    # REGRA - 3 SE ALTURA É ALTA E PESO É ALTO ENTÃO TAMANHO É GRANDE

    regra3 = ctrl.Rule(altura['alta'] & peso['alto'], tamanho['grande'])

    # CRIANDO O CONTROLADOR FUZZY COM AS REGRAS 

    fuzzy_ctrl = ctrl.ControlSystem([regra1, regra2, regra3])

    # CRIANDO O MOTOR DE INFERENCIA

    engine = ctrl.ControlSystemSimulation(fuzzy_ctrl)

    # PREDIÇÕES DOS MODELOS
    InputPeso = int(input("Digite seu peso: "))
    engine.input['peso'] = InputPeso

    InputAltura = float(input("Digite sua altura: "))
    engine.input['altura'] = InputAltura

    # CALCULA A SAIDA DA LOGICA FUZZY

    engine.compute()

    # RETORNA O VALOR CRISP E O GRAFICO MOSTRANDO-O
    print(engine.output['tamanho'])
    tamanho.view(sim=engine)

    test4 = input(" ")

__main__()
    
