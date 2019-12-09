import heapq
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        if n == 0:
            return 0
        if k == 1:
            return matrix[0][0]
        if k == n*n:
            return matrix[n-1][n-1]
        
        #print(n)
        #'Split Matrix?'
        
        heap = []
        for i in range(n):
            heapq.heappush(heap, (matrix[i][0],i,0))
        
        for _ in range(k):
            s,i,j = heapq.heappop(heap)
            if j+1 < n:
                heapq.heappush(heap, (matrix[i][j+1],i,j+1))
        
        smallest = s
        '''
        for _ in range(k):
            smallest = matrix[0][0]
            matrix[0][0] = matrix[n-1][n-1]
            
            x = 0
            y = 0
            
            #cnt = 0
            while (x+1<n and matrix[x][y] > matrix[x+1][y]) or (y+1<n and matrix[x][y] > matrix[x][y+1]):
                #print(matrix[x][y])
                #cnt += 1
                
                if x+1<n and y+1<n:
                    if matrix[x+1][y] >= matrix[x][y+1]:
                        matrix[x][y+1], matrix[x][y] = matrix[x][y], matrix[x][y+1]
                        #print(matrix[x][y])
                        y += 1
                    else:
                        matrix[x+1][y], matrix[x][y] = matrix[x][y], matrix[x+1][y]
                        #print(matrix[x][y])
                        x += 1
                else:
                    if y+1 < n:
                        matrix[x][y+1], matrix[x][y] = matrix[x][y], matrix[x][y+1]
                        y += 1
                    else: # x+1 < n
                        matrix[x+1][y], matrix[x][y] = matrix[x][y], matrix[x+1][y]
                        x += 1
        '''
                
        return smallest
                