class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        for i in range(n-1): #loops
            for j in range(i,n-1-i):
                #print(matrix[i][j],matrix[j][n-1-i],matrix[n-1-i][n-1-j],matrix[n-1-j][i],i,j)
                matrix[j][n-1-i],matrix[n-1-i][n-1-j],matrix[n-1-j][i],matrix[i][j] = matrix[i][j],matrix[j][n-1-i],matrix[n-1-i][n-1-j],matrix[n-1-j][i]
                #matrix[n-1-j][i], matrix[i][j], matrix[n-1-i][n-1-j], matrix[j][n-1-i] = matrix[i][j],matrix[j][n-1-i],matrix[n-1-i][n-1-j],matrix[n-1-j][i]
        