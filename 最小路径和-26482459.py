class Solution(object):
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        rows = len(grid)
        cols = len(grid[0])
        dp = [[0]*cols for _ in range(rows)]
        #print(dp)
        for r in range(rows):
            for c in range(cols):
                if r == 0:
                    #print(r,c)
                    dp[r][c] = grid[r][c] + dp[r][c-1]
                    continue
                
                if c == 0:
                    dp[r][c] = grid[r][c] + dp[r-1][c]
                    continue
                
                dp[r][c] = grid[r][c] + min(dp[r-1][c], dp[r][c-1])
        
        #print(dp)
        return dp[rows-1][cols-1]
                
                
                