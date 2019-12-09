class Solution:
    def numberOfArithmeticSlices(self, A: List[int]) -> int:
        N = len(A)
        if N < 3:
            return 0
        cnt = 0
        dlt = [sys.maxsize] * N
        for i in range(N-1):
            dlt[i+1] = A[i+1]-A[i]
        dp = [0] * N
        for i in range(2,N):
            if dlt[i] == dlt[i-1]:
                dp[i] = 1 + dp[i-1]
        #print(dp)
        return sum(dp)
        '''
        dp = [[sys.maxsize]*N for _ in range(N)]
        for i in range(N-1):
            dp[i][i+1] = A[i+1]-A[i]
        #print(dp)
        for s in range(2,N):
            for i in range(N-s):
                dlt = A[i+s]-A[i+s-1]
                #print(i,s,dlt,i,i+s-1,dp[i][i+s-1])
                if dlt == dp[i][i+s-1]:
                    dp[i][i+s] = dlt
                    cnt += 1
        '''
        return cnt