import random


def func(x, y):
    return x ** 2 - x * y + 3 * y ** 2 - x


pop = [[[1.1, -2.6]], [[32, 0.4]], [[-5, 6]], [[7.7, 18]], [[-19, 0.1]]]
children = []
e = -0.00001
h = 0.1
qwerty = 0
for i in pop:
    i.append(func(i[0][0], i[0][1]))
result = 100
while result > e:
    for i in range(0, len(pop)):
        for j in range(i + 1, len(pop)):
            children.append([[pop[i][0][random.randint(0, 1)], pop[j][0][random.randint(0, 1)]]])
    for i in children:
        i.append(func(i[0][0], i[0][1]))
    children.sort(key=lambda x: x[1])
    pop = pop + children
    pop.sort(key=lambda x: x[1])
    for i in pop:
        if random.randint(0, 40) >= random.randint(0, 100):
            for item in range(0, random.randint(1, 2)):
                gen = random.randint(0, 1)
                antigen = 0
                if gen == 0:
                    antigen = 1
                i[0][gen] += h * random.randint(1, 5)
                i[0][antigen] -= h * random.randint(1, 5)
    for i in pop:
        i[1] = func(i[0][0], i[0][1])
    while len(pop) > random.randint(5, 8):
        pop.pop()
    result = pop[0][1]
    if round(result, 2) == -0.07:
        pop.pop(0)
    result = pop[0][1]
    qwerty += 1
    children.clear()
    print(len(pop), pop)

