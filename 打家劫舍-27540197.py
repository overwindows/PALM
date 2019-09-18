class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        
        dp0 = [0]*len(nums)
        dp1 = [0]*len(nums)
        
        dp0[0] = 0
        dp1[0] = nums[0] 
        if len(nums) > 1:
            dp0[1] = nums[0]
            dp1[1] = nums[1]
        else:
            return nums[0]
        
        for i in range(2, len(nums)):
            dp0[i] = max(dp0[i-1], dp1[i-1])
            dp1[i] = max(dp0[i-1],dp1[i-2]) + nums[i]
        
        return max(max(dp0),max(dp1))