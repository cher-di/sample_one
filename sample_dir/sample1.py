def equal(weight, num, curr_sum, target):
    if curr_sum == target:
        return 'YES'
    if curr_sum > target:
        return 'NO'
    if num == -1:
        return 'NO'
    res1 = equal(weight, num - 1, curr_sum + weight[num], target)
    res2 = equal(weight, num - 1, curr_sum, target)
    if res1 == 'YES' or res2 == 'YES':
        return 'YES'
    return 'NO'

s = input().split()
weight = [int(i) for i in s]
if sum(weight) % 2:
    print('NO')
else:
    target =  sum(weight)//2;
    num = len(weight) - 1
    curr_sum = 0
    print(equal(weight, num, curr_sum, target))
