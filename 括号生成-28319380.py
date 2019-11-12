import itertools

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        dp = [[] for x in range(n+1)]
        if n == 0:
            return []
        dp[0] = []
        dp[1] = ['()']
        
        for i in range(2, n+1):
            for j in range(1, i):
                #print(j, i-j, i)
                #if (i-j) >= j:
                #print(dp[j],dp[i-j],dp[i])
                dp[i].extend([''.join(x) for x in list(itertools.product(dp[j],dp[i-j]))])
            dp[i].extend(['('+x+')' for x in dp[i-1]])
            dp[i] = list(set(dp[i]))
        '''
        s = self.generateParenthesis(n-1)
        res  = []
        for x in s:
            res.append('('+x+')')
            res.append('()'+x)
            res.append(x+'()')
        
        res = list(set(res))
        res.sort()
        '''
        dp[n].sort()
        return dp[n]