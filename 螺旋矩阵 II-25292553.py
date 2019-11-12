class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0] * n for _ in range(n)]
        lst = [(i+1) for i in range(n**2)]
        T = n // 2 + n % 2
        m = n
        #print(m,n,lst)
        x = 0
        for t in range(T):
            i = t
            for j in range(t,n-t):
                matrix[i][j] = lst[x]
                #print(lst[x],matrix)
                x += 1
            j = n-1-t
            if t+1 == m-t:
                break
            for i in range(t+1, m-t):
                matrix[i][j] = lst[x]
                #print(lst[x],matrix)
                x += 1
            if n-t-2 == t-1:
                break
            i = m-1-t
            for j in range(n-t-2, t-1, -1):
                matrix[i][j] = lst[x]
                #print(lst[x],matrix)
                x += 1
            j = t
            for i in range(m-t-2, t, -1):
                matrix[i][j] = lst[x]
                #print(lst[x],matrix)
                x += 1
        return matrix 