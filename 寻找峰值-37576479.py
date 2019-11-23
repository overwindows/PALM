class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        lens = len(nums)
        if lens == 1:
            return 0
        if lens == 2:
            if nums[0] > nums[1]:
                return 0
            else:
                return 1
        if nums[0] > nums[1]:
            return 0
        if nums[lens-1] > nums[lens-2]:
            return lens-1
        for i in range(1,lens-1):
            if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                return i
        
        return -1