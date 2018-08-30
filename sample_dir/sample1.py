from time import clock
from math import sqrt
num = 10**4

time_before = clock()
def_prime = [2]
for curr in range(3, num + 2, 2):
    sqrt_curr = sqrt(curr) + 1
    for prime in [i for i in def_prime if i < sqrt_curr]:
        if curr % prime == 0:
            break
    else:
        def_prime.append(curr)
time_after = clock()

for prime in def_prime:
    print(prime)

print('***********************')
print('Время вычисления: {}'.format(time_after - time_before))
