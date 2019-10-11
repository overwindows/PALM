class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        lens = len(nums)
        ix = 0
        
        if lens == 0:
            return 0
        
        if lens == 1 and nums[0] != 1:
            return nums[0]+1
        
        while ix < lens:
            _ix = nums[ix]
            
            if _ix == ix:
                ix += 1
            else:
                if _ix == lens:
                    ix += 1
                else: 
                    t = nums[_ix]
                    nums[_ix] = _ix
                    nums[ix] = t
        
        for i in range(lens):
            if nums[i] != i:
                return i
        return nums[-1] + 1