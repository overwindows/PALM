class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        row_len = len(obstacleGrid)
        if row_len == 0:
            return 0
        col_len = len(obstacleGrid[0])
        if col_len == 0:
            return 0

        if row_len == 1 and col_len == 1:
            if obstacleGrid[0][0] == 0:
                return 1
            else:
                return 0
        
        if obstacleGrid[0][0] == 1:
            return 0
        
        dp = [[0] * col_len for _ in range(row_len)]
        dp[0][0] = 1
        
        for r in range(row_len):
            for c in range(col_len):
                #print(dp)
                if r == 0 and c > 0 and obstacleGrid[r][c] == 0:
                    dp[0][c] = dp[0][c-1]
                    #print(c,dp[0][c],dp[0][c-1])
                    continue
                
                if r >0 and c == 0 and obstacleGrid[r][c] == 0:
                    dp[r][0] = dp[r-1][0]
                    continue
                
                if r > 0 and c >0 and obstacleGrid[r][c] == 0:
                    dp[r][c] = dp[r][c-1] + dp[r-1][c]
                
        return dp[-1][-1]