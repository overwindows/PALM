class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        maxProd = nums[0]
        
        posProd = [0] * len(nums)
        negProd = [0] * len(nums)
        
        posProd[0] = nums[0]
        negProd[0] = nums[0]
        
        for i in range(1, len(nums)):
            posProd[i] = max(posProd[i-1]*nums[i], negProd[i-1]*nums[i], nums[i])
            negProd[i] = min(posProd[i-1]*nums[i], negProd[i-1]*nums[i], nums[i])
            
            maxProd = max(maxProd, posProd[i], negProd[i])
        
        return maxProd