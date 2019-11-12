class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        if len(grid) == 0:
            return 0
        
        if len(grid[0]) == 0:
            return 0
            
        status = [[False]*len(grid[0]) for _ in range(len(grid))]
        
        start_x = 0
        start_y = 0
        
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if grid[x][y] == 1:
                    start_x = x
                    start_y = y
                if grid[x][y] == -1:
                    status[x][y] = True
        
        status[start_x][start_y] = True
        
        def _uniqPaths(x,y, grid:List[List[int]], status: List[List[bool]]):
            assert x < len(grid) and y < len(grid[0])
            if grid[x][y] == 2:
                #print(status)
                for i in range(len(grid)):
                    for j in range(len(grid[0])):
                        if not status[i][j]:
                            #print(x, y)
                            return 0
                return 1
            if grid[x][y] == -1:
                assert status[x][y]
                return 0
            
            #print(x, y)
            paths = 0
            if y+1 < len(grid[0]) and not status[x][y+1]:
                status[x][y+1] = True
                paths += _uniqPaths(x,y+1, grid, status)
                status[x][y+1] = False
            
            if y>0 and not status[x][y-1]:
                status[x][y-1] = True
                paths += _uniqPaths(x, y-1, grid, status)
                status[x][y-1] = False
            
            if x>0 and not status[x-1][y]:
                status[x-1][y] = True
                paths += _uniqPaths(x-1, y, grid, status)
                status[x-1][y] = False
            
            if x+1 < len(grid) and not status[x+1][y]:
                status[x+1][y] = True
                paths += _uniqPaths(x+1, y, grid, status)
                status[x+1][y] = False
            
            return paths
        
        all_paths = _uniqPaths(start_x, start_y, grid, status)
        return all_paths