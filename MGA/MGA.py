
# Функция: f(x) = sin^6 (5*Pi(x^0.75 - 0.05)) на интервале [0,1].

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import math  as m
import random as r 

# Кол-во индивидов в популяции
POP_SIZE = 50
# Кол-во поколений 
MAX_GENETATION = 9
# Радиус вокруг особи для выбора партрнера 
R_c = 0.099
# Максимальное расстояние
R_max = 0.01
# Минимальное расстояние
R_min = 0.098
# Вероятность мутации
PROB_MUTATION = 0.001
# Вероятность секса
PROB_SEX = 0.8


# Исходная функция 
def equation(num):
    return (m.sin(5*m.pi*(num**0.75 - 0.05)))**6


# Преобразования чисел от 0 до 1 в 16-ричную (Точность до 5 знаков после запятой)
def float_to_bin(num):
    a = np.uint16(65535*num)
    b = list(format(a, 'b').zfill(16))
    return np.array(b)


 # Преобразовае обратно 
def bin_to_float(bin_array):
    num = ''.join([str(x) for x in bin_array.tolist()])
    return int(num,2) / 65535


# Генерирует случайную первую популяцию 
# можно написать в 1 строчку так то
def first_pop ():
    #first_pop = np.empty(POP_SIZE,dtype = np.ndarray)
    first_pop = np.empty(POP_SIZE)
    for i in range(POP_SIZE):
        #num = float_to_bin()
        first_pop[i] = round(r.random(),4)
    return first_pop


# Графическое представление данных 
def graph(x_array,number):
    fig = plt.figure()#figsize = (9,8))
    ax1 = fig.add_subplot()
    x = np.arange(0,1,0.001)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = equation(x[i])
    plt.title(f'Популяция {number} ')
    plt.xlabel('Ox')
    plt.ylabel('Oy')
    ax1.plot(x,y)
    #plt.plot(x, y, color = 'blue', linestyle = 'solid',label = 'F(x) = (x-1)cos(3x-15)')
    ax1.grid()
    for i in range (POP_SIZE):
        #xi = bin_to_float(x_array[i])
        yi = equation(x_array[i])
        plt.scatter(x_array[i], yi, color ='green', s = 60, marker = '*')
    #plt.show()
    fig.savefig(f'Популяция {number}.png')
    

# Оператор мутации 
def mutation(gen):
    shans = round(r.random(),1)
    gen = float_to_bin(gen)
    if shans <= PROB_MUTATION:
        d = r.randint(0,len(gen) -1)
        if gen[d] == 0:
            gen[d] = 1
        else:
            gen[d] = 0
    return bin_to_float(gen)


# Операция скрещивания  (ок)
def crossing(men,women):
    shans = round(r.random(),1)
    men = float_to_bin(men)
    women = float_to_bin(women)
    if shans <= PROB_SEX:
         d = r.randint(0,len(men) -1)
         men[d:],women[d:] = women[d:], men[d:]
    return bin_to_float(men),bin_to_float(women)

# Расстояние евклида
def eculid_old(x1,y1,x2,y2):
    return ((x1 - x2)**2 +(y1 - y2)**2)**1/2
# Расстояние евклида
def eculid(x1,x2):
    return ((x1 - x2)**2 )**1/2


# турнирный выбор
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = np.random.randint(len(pop))
	for ix in np.random.randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] > scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]


def main():
    # Генерация первой популяции 
    nw = first_pop()
    
    vfun = np.vectorize(equation)
    rr = np.vectorize(round)
    # цикл до поколения = MAX_GENETATION
    for q in range(MAX_GENETATION):
        fitnes_f = vfun(nw)
        jopa = []
        for i in range(POP_SIZE):
            jopa.append(selection(nw,fitnes_f,4))
        nw = np.array(jopa)



        #Опреатор кросингвера (скрещивание особей) 
        for j in range(0,POP_SIZE,2):
            nw[j],nw[j + 1] = crossing(nw[j],nw[j + 1])
        
        # Оператор мутации 
        for x in range(POP_SIZE):
            nw[x] = mutation(nw[x])
        print(q)
        graph(nw,q)





if __name__ == '__main__':
    main()

    print("end")

