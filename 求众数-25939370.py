class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        size = len(nums)
        stack = []
        
        for x in nums:
            if len(stack) == 0:
                stack.append(x)
            else:
                if stack[-1] == x:
                    stack.append(x)
                else:
                    stack.pop()
        return stack.pop()
        
        
            
                
            