class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        rows = len(matrix)
        if rows == 0:
            return 0
        cols = len(matrix[0])
        maxSquare = 0
        
        mat = [list(map(int,x)) for x in matrix]
        
        for r in range(rows):
            for c in range(cols):
                if mat[r][c] > 0 and r > 0 and c > 0:
                        mat[r][c] += min(mat[r-1][c],mat[r][c-1],mat[r-1][c-1])
                maxSquare = max(maxSquare,mat[r][c])
        
        return maxSquare**2