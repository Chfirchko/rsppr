import random
import matplotlib.pyplot as plt
import numpy as np


def func(x, y):
    return x ** 2 - x * y + 3 * y ** 2 - x


f = open('data.txt', 'a')

all = []

children = []

e = -0.1
h = 0.1
qwerty = 0


result = 100

for zxc in range(0, 100):
    pop = []
    for i in range(0, 10):
        pop.append([float(random.randint(-10, 10)), float(random.randint(-10, 10))])
    for i in pop:
        i.append(func(i[0], i[1]))
    for i in pop:
        gen = random.randint(0, 1)
        antigen = 0
        if gen == 0:
            antigen = 1
        i[gen] += h * random.randint(1, 5)
        i[antigen] -= h * random.randint(1, 5)
    #print(pop)
    for i in range(0, len(pop)):
        if random.randint(0, 20) >= random.randint(0, 100):
            for j in range(i + 1, len(pop)):
                children.append([pop[i][random.randint(0, 1)], pop[j][random.randint(0, 1)]])
            for i in children:
                i.append(func(i[0], i[1]))
    pop += children
    pop.sort(key=lambda x: x[2])
    while len(pop) > 10:
        pop.pop()

    all.append(pop)
# for qwer in range(0, 10):
#     for i in range(0, len(pop)):
#         for j in range(i + 1, len(pop)):
#             children.append([[pop[i][0][random.randint(0, 1)], pop[j][0][random.randint(0, 1)]]])
#     for i in children:
#         i.append(func(i[0][0], i[0][1]))
#     children.sort(key=lambda x: x[1])
#     pop = pop + children
#     pop.sort(key=lambda x: x[1])
#     for i in pop:
#         if random.randint(0, 40) >= random.randint(0, 100):
#             for item in range(0, random.randint(1, 2)):
#                 gen = random.randint(0, 1)
#                 antigen = 0
#                 if gen == 0:
#                     antigen = 1
#                 i[0][gen] += h * random.randint(1, 5)
#                 i[0][antigen] -= h * random.randint(1, 5)
#     for i in pop:
#         i[1] = func(i[0][0], i[0][1])
#     while len(pop) > random.randint(5, 8):
#         pop.pop()
#     result = pop[0][1]
#     qwerty += 1
#     children.clear()
#     print(len(pop), pop)
#     for i in pop:
#         line1 = str(i[0][0])
#         line2 = str(i[0][1])
#         line3 = str(i[1])
#         line = line1 + ' ' + line2 + ' ' + line3
#         f.write(line)
#         f.write('\n')

# f1 = open('data.txt', 'r')
#
# x = np.linspace(-10, 10, 100)
# y = np.linspace(-10, 10, 100)
#
# X, Y = np.meshgrid(x, y)
# Z = func(X, Y)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 150, cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# table = []
# tab = []
# chislo = 0
# lines = f1.read().splitlines()
# peremennaya = ''
# print(lines)
# for i in lines:
#     for j in i:
#         if j != ' ':
#             peremennaya += j
#         elif j == '\\':
#             i = i[:1]
#         else:
#             chislo = float(peremennaya)
#             print(chislo)
#             peremennaya = ''
#             tab.append(chislo)
#     chislo = float(peremennaya)
#     tab.append(chislo)
#     table.append(tab)
#     tab = []
#     peremennaya = ''
# for i in table:
#     ax.scatter(i[0], i[1], i[2], s=10, color='r')

table1 = []
table2 = []
table3 = []
for i in all:
    for j in range(3):
        table1.append(i[j][0])
        table2.append(i[j][1])
        table3.append(i[j][2])


x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

X, Y = np.meshgrid(x, y)
Z = func(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 150, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.plot(table1, table2, table3, 'ro')
plt.show()
