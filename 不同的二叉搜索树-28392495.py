class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 1
        if n==1:
            return 1
        dp = [0]*(n+1)
        dp[1] = 1
        dp[2] = 2
        for i in range(3,n+1):
            dp[i] += dp[i-1] * 2
            for s in range(2,i):
                dp[i] += dp[s-1] * dp[i-s]
        #print(dp) 
        return dp[n]