class Solution:
    def getMoneyAmount(self, n: int) -> int:
        if n==1:
            return 0
        if n==2:
            return 1
        if n==3:
            return 2

        dp = [[sys.maxsize]*(n) for _ in range(n)]
        
        for m in range(n):
            dp[m][m] = 0

        for i in range(1,n):
            for j in range(n-i):
                #print(j,j+i)
                for k in range(j,j+i+1):
                    if k == j:
                        dp[j][i+j] = min(k+dp[k+1][j+i]+1, dp[j][i+j])
                    elif k == j+i:
                        dp[j][i+j] = min(dp[j][k-1]+k+1, dp[j][i+j])
                    else:
                        dp[j][i+j] = min(max(dp[j][k-1],dp[k+1][j+i])+k+1, dp[j][i+j])
                #print(j+1,j+i+1, dp[j][j+i])
        return dp[0][n-1]
        '''
        cp = [0]*(n+1)
        dp = [sys.maxsize]*(n+1)
        dp[1] = 0
        cp[1] = 0
        dp[2] = 1
        cp[2] = 1
        dp[3] = 2
        cp[3] = 1

        for i in range(4,n+1):
            #print(1,i)
            for j in range(1,i+1):
                dp[i] = min(j+max(dp[j-1], cp[i-j]*j+dp[i-j]), dp[i])
                if dp[i] == j+dp[j-1]:
                    #assert cp[i] >= cp[j-1]+1
                    cp[i] = cp[j-1] + 1
                elif dp[i] == j+ cp[i-j]*j+dp[i-j]:
                    #assert cp[i] >= cp[i-j] + 1
                    cp[i] = cp[i-j] + 1

        return dp[n]
    '''
'''
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
100
'''