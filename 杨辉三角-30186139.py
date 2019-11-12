class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = []
        for i in range(numRows):
            row = []
            for j in range(i+1):
                val = 1
                if i > 0  and j > 0 and j < i:
                    val = res[i-1][j-1] + res[i-1][j]
                row.append(val)
            res.append(row)
        
        return res
            
                    