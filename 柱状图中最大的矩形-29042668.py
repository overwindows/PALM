class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        len_h = len(heights)
        
        if len_h == 0:
            return 0
        
        if len_h == 1:
            return heights[0]
        
        if sorted(heights) == heights:
            max_rect = 0
            for i in range(len_h):
                max_rect = max(max_rect, (len_h-i)*heights[i])
            
            return max_rect
        
        min_hix = 0
        for ix in range(1,len_h):
            if heights[ix] < heights[min_hix]:
                min_hix = ix
        
        cur_max = heights[min_hix] * len_h
        left_max = 0
        right_max = 0
        
        if min_hix > 0:
            left_max = self.largestRectangleArea(heights[:min_hix])
        if min_hix < len_h:
            right_max = self.largestRectangleArea(heights[min_hix+1:])
        
        return max(cur_max, left_max, right_max)