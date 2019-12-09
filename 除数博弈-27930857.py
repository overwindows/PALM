class Solution:
    def divisorGame(self, N: int) -> bool:
        dp = [False] * (N+1)
        for n in range(1,N+1):
            #print(dp[1:])
            for x in range(1,n):
                if n%x == 0 and not dp[n-x]:
                    dp[n] = True
                    break
        return dp[N]
                    
        