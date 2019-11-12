class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m = len(matrix)
        if m > 0:
            n = len(matrix[0])
        else:
            return None
        T = min(m,n) // 2 + min(m,n) % 2
        res = []
        i = 0
        j = 0
        for t in range(T):
            i = t
            for j in range(t,n-t):
                res.append(matrix[i][j])
                
            j = n-1-t
            if t+1 == m-t:
                break
            for i in range(t+1, m-t):
                res.append(matrix[i][j])
            if n-t-2 == t-1:
                break
            i = m-1-t
            for j in range(n-t-2, t-1, -1):
                res.append(matrix[i][j])
            
            j = t
            for i in range(m-t-2, t, -1):
                res.append(matrix[i][j])
        return res
            