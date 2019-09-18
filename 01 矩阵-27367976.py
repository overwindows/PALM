class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        X = len(matrix) - 1
        Y = len(matrix[0]) - 1
        
        q = []
        visited = set()

        for i in range(X+1):
            for j in range(Y+1):
                if matrix[i][j] == 0:
                    q.append((i,j))
                    visited.add((i,j))
                else:
                    matrix[i][j] = -1
        
        while q:
            x,y = q.pop(0)
            
            if x > 0 and (x-1,y) not in visited:
                matrix[x-1][y] = matrix[x][y] + 1
                q.append((x-1,y))
                visited.add((x-1,y))
            if y > 0 and (x, y-1) not in visited:
                matrix[x][y-1] = matrix[x][y] + 1
                q.append((x,y-1))
                visited.add((x,y-1))
            if x < X and  (x+1,y) not in visited:
                matrix[x+1][y] = matrix[x][y] + 1
                q.append((x+1,y))
                visited.add((x+1,y))
            if y < Y and (x,y+1) not in visited:
                matrix[x][y+1] = matrix[x][y] + 1
                q.append((x,y+1))
                visited.add((x,y+1))
        
        return matrix
                    