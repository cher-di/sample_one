class Stirling:
    def __init__(self):
        n_num = 1000
        self.stirling_cash = [[0 for i in range(k + 1)] for k in range(n_num)]
        self.stirling_cash[0][0] = 1
        self.stirling_cash[1][0] = 1
        self.stirling_cash[1][1] = 1
        for n in range(2, len(self.stirling_cash) + 1):
            self.stirling_cash[n - 1][0] = 1
            self.stirling_cash[n - 1][n - 1] = 1


    def count_Stirling(self, n, k):
        if self.stirling_cash[n - 1][k - 1]:
            return self.stirling_cash[n - 1][k - 1]

        self.stirling_cash[n - 1][k - 1] = self.count_Stirling(n - 1, k - 1) + k*self.count_Stirling(n - 1, k)
        return self.stirling_cash[n - 1][k - 1]
