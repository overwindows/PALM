from queue import Queue

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows = len(grid)
        if rows == 0:
            return 0
        
        cols = len(grid[0])
        if cols == 0:
            return 0
        
        nums = 0
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '0':
                    continue
                nums += 1
                q = Queue()
                q.put((i,j))    
                
                while not q.empty():
                    _x,_y = q.get()
                    
                    if not int(grid[_x][_y]):
                        continue
                    else:
                        grid[_x][_y] = '0'
                                        
                    if _x+1 < rows and grid[_x+1][_y] == '1':
                        q.put((_x+1, _y))
                    if _x > 0 and grid[_x-1][_y] == '1':
                        q.put((_x-1, _y))
                    if _y+1 < cols and grid[_x][_y+1] == '1':
                        q.put((_x, _y+1))
                    if _y > 0 and grid[_x][_y-1] == '1':
                        q.put((_x, _y-1))
        
        return nums     