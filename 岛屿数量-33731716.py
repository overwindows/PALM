from queue import Queue

class Solution(object):
    def numIslands(self, grid):
        try:
            r = 0; m = len(grid); n = len(grid[0])
            around = ((0, 1), (1, 0), (0, -1), (-1, 0))
        except:
            return 0
        
        for i in range(m):
            for j in range(n):
                if int(grid[i][j]):
                    r += 1
                    
                    #---------------------------BFS 开始-----------------------------
                    # 1.把根节点投入队列
                    q = Queue()
                    q.put((i, j))

                    # 开始循环
                    while not q.empty():
                        # 取出还未沉没的陆地节点并沉没陆地（防止下次遍历到的时候再算一遍）
                        x, y = q.get()
                        
                        if int(grid[x][y]):
                            grid[x][y] = '0'

                            # 放入周围的陆地节点
                            for a, b in around:
                                a += x; b += y;
                                if 0 <= a < m and 0 <= b < n and int(grid[a][b]):
                                    q.put((a, b))
                    #----------------------------------------------------------------
        return r
'''

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
                    
                    if grid[_x][_y] == '0':
                        continue
                    else:
                        grid[_x][_y] == '0'
                    
                    if _x+1 < rows and grid[_x+1][_y] == '1':
                        q.put((_x+1, _y))
                    if _x > 0 and grid[_x-1][_y] == '1':
                        q.put((_x-1, _y))
                    if _y+1 < cols and grid[_x][_y+1] == '1':
                        q.put((_x, _y+1))
                    if _y > 0 and grid[_x][_y-1] == '1':
                        q.put((_x, _y-1))
        
        return nums     
'''