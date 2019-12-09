class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        flat = []
        l = len(matrix)
        if l == 0:
            return 0
        w = len(matrix[0])
        if w == 0:
            return 0
        
        mat = [[0]*w for _ in range(l)]
        
        for i in range(l):
            for j in range(w):
                flat.append((matrix[i][j],i,j))
                
        flat.sort()
        # print(flat,mat)
        longest = 0
        while flat:
            _,i,j = flat.pop()
            
            if i>0 and matrix[i][j] > matrix[i-1][j]:
                mat[i-1][j] = max(mat[i-1][j], mat[i][j]+1)
                longest = max(longest, mat[i-1][j])
            
            if i+1 < l and matrix[i][j] > matrix[i+1][j]:
                mat[i+1][j] = max(mat[i+1][j], mat[i][j]+1)
                longest = max(longest, mat[i+1][j])
                
            if j+1 < w and matrix[i][j] > matrix[i][j+1]:
                mat[i][j+1] = max(mat[i][j+1], mat[i][j]+1)
                longest = max(longest, mat[i][j+1])
            
            if j>0 and matrix[i][j] > matrix[i][j-1]:
                mat[i][j-1] = max(mat[i][j-1], mat[i][j]+1)
                longest = max(longest, mat[i][j-1])
            
            
        return longest+1
        