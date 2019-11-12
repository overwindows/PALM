class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        st = 0
        ed = len(height)-1
        most = 0
        
        while st < ed:
            _most = (ed-st)*min(height[st],height[ed])
            most = max(most,_most)
            
            if height[st] < height[ed]:
                st += 1
            else:
                ed -= 1
                
        return most
           
            
