import numpy as np
from scipy.spatial import distance

flat_list = []
o = [1, 2, 3, 4]
u=[1, 2, 3, 4]
new=[0]*10
u[1]/10

del (u[0])
u.append(6)
#print(u)

# Creamos una matrix vacia de NxN
dMatrix = np.zeros([len(u), len(u)])
# Solo vamos a calcular el triangulo derecho superior para no duplicar calculos
for i in range(0, len(u)):
    for j in range(i+1, len(u)):
        dMatrix[i, j] = u[i] +u[j]

#Insertar fila
new=np.insert(dMatrix,dMatrix.shape[0],np.zeros(len(dMatrix)),0)
#Insertar columna
new=np.insert(new,new.shape[1],np.zeros(len(new)),1)

new[0,len(new)-1]=2

print(dMatrix)
print()
print(new)
