class Solution(object):
    def numSquares(self, n):
        #py中通过,py3超时
        maxn = int(n ** 0.5)
        dp = [_ for _ in range(n + 1)]
        for i in range(1,maxn + 1):
            for j in range(i * i,n + 1):
                dp[j] = min(dp[j],dp[j - i * i] + 1)
        return dp[n]