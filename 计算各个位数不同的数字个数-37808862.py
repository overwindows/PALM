class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        if n == 0:
            return 1
        if n == 1:
            return 10

        dp = [0]*(n+1)

        dp[0] = 0
        dp[1] = dp[0]+10
        dp[2] = dp[1] + 9*9

        for i in range(3,n+1):
            factor = 9
            combo = 9
            for _ in range(1,i):
                 combo *= factor
                 factor -= 1
            dp[i] = dp[i-1] + combo
        
        return dp[n]
        
        