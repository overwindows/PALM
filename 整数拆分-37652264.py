class Solution:
    def integerBreak(self, n: int) -> int:
        if n == 0:
            return 0
        dp = [0]*(n+1)
        dp[1] = 1
        for i in range(2,n+1):
            for j in range(1,i):
                dp[i] = max(dp[i], j*max(i-j,dp[i-j]))
        #print(dp)
        return dp[n]


