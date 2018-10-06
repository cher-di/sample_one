n = int(input())

a = b = int(n // 3)
c = n - a - b
if a % 3 == 0:
    a -= 1
    b += 1

if b % 3 == 0:
    b -= 1
    c += 1

if c % 3 == 0:
    if (b + 1) % 3 == 0:
        b += 2
        c -= 2
    else:
        b += 1
        c -= 1
print(a, b, c)