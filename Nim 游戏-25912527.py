class Solution:
    def canWinNim(self, n: int) -> bool:
        
        return n % 4
        
        if n < 4:
            return True
        
        dp = [False] * (n+1)
        dp[1] = True
        dp[2] = True
        dp[3] = True
        
        for i in range(3,n+1):
            dp[i] = not (dp[i-1] & dp[i-2] & dp[i-3])
            
        return dp[n]
        