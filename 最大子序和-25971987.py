class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """            
    
        max_sum = nums[0]
        cum_sum = nums[0]
        
        for i in range(1, len(nums)):
            cum_sum = max(0, cum_sum+nums[i], nums[i])
            if cum_sum == 0:
                max_sum = max(max_sum, nums[i])
            else:
                max_sum = max(max_sum, nums[i], cum_sum)
        
        return max_sum
            