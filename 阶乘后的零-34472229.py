class Solution:
    def trailingZeroes(self, n: int) -> int:
        factor_5 = []
        factor_10 = []
        
        #5,25,125,625,...
        while n > 4:
            n = n // 5
            factor_5.append(n)
        
        #10,100,1000,10000,....
        while n > 9:
            n = n // 10
            factor_10.append(n)
            
        
        for i in range(1,len(factor_5)):
            factor_5[i-1] = factor_5[i-1]-factor_5[i]

        sum_5 = 0
        for i in range(len(factor_5)):
            sum_5 += (i+1) * factor_5[i]
        
        for j in range(1, len(factor_10)):
            factor_10[j-1] = factor_10[j-1]-factor_10[j]
            
        sum_10 = 0
        for j in range(len(factor_10)):
            sum_10 += (j+1) * factor_10[j]
        
        return sum_10+sum_5-sum(factor_10)
        
            
        
        