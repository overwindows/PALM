class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        _ix = 1
        while True:
            ix = nums[_ix-1]
            if _ix == ix:
                _ix = (_ix+1) % (len(nums)-1)
            else:    
                if nums[_ix-1] == nums[ix-1]:
                    return nums[ix-1]
                else:
                    nums[_ix-1] = nums[ix-1]
                    nums[ix-1] = ix