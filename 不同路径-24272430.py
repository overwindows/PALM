class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        mat = [[0]*n]*m
        mat[0][0]=1
        for i in range(m):
            for j in range(n):
                if i > 0 and j > 0:
                    mat[i][j] = mat[i-1][j] + mat[i][j-1] 
                if i == 0 and j > 0:
                    mat[i][j] = mat[i][j-1]
                if j == 0 and i > 0:
                    mat[i][j] = mat[i-1][j]
        
        return mat[m-1][n-1]
                    