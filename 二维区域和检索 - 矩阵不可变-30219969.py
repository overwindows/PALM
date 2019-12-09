class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.rows = len(matrix)
        if self.rows > 0:
            self.cols = len(matrix[0])
        else:
            self.cols = 0
        
        self.dp = matrix
        for i in range(self.rows):
            for j in range(self.cols):
                if j > 0 and i > 0:
                    self.dp[i][j] += (self.dp[i-1][j] + self.dp[i][j-1] - self.dp[i-1][j-1])
                else:
                    if j == 0 and i==0:
                        continue
                    elif j == 0:
                        self.dp[i][j] += self.dp[i-1][j]
                    elif i == 0:
                        self.dp[i][j] += (self.dp[i][j-1])
                
        

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        val = 0
        if row1 > -1 and col1 > -1 and row2 < self.rows and col2 < self.cols:
            if row1 > 0 and col1 > 0:
                val = self.dp[row2][col2] + self.dp[row1-1][col1-1] - self.dp[row1-1][col2] - self.dp[row2][col1-1]
            else:
                if row1 == 0 and col1 == 0:
                    val = self.dp[row2][col2]
                elif row1 == 0:
                    val = self.dp[row2][col2] - self.dp[row2][col1-1]
                elif col1 == 0:
                    val = self.dp[row2][col2] - self.dp[row1-1][col2]
        
        return val
            


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)