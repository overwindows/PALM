class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        if n == 0:
            return 0

        for i in range(n):
            m = len(triangle[i])
            for j in range(m):
                left_adj = None
                right_adj = None
                if i > 0 and j > 0:
                    left_adj = triangle[i-1][j-1]
                if i > 0 and j < len(triangle[i-1]):
                    right_adj = triangle[i-1][j]
                
                if left_adj == None and right_adj == None:
                    continue
                
                if left_adj == None:
                    triangle[i][j] += right_adj
                    continue
                if right_adj == None:
                    triangle[i][j] += left_adj
                    continue
                triangle[i][j] += min(left_adj,right_adj)
        
        #print(triangle)
        return min(triangle[-1])
                
            