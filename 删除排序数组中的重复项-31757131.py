class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        lens = len(nums)
        new_lens = lens
        if lens == 0:
            return 0
        if lens == 1:
            return 1
        p0 = 0
        p1 = 1
        
        while p1 < lens:
            if nums[p0] == nums[p1]:
                p1 += 1
            else:
                p0 += 1
                nums[p0] = nums[p1]
                p1 += 1
        
        return p0+1
            