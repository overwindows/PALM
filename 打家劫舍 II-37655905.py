class Solution:
    def rob(self, nums: List[int]) -> int:
        lens = len(nums)

        if lens == 0:
            return 0
        
        if lens == 1:
            return nums[0]

        dp0 = [0] * lens
        dp1 = [0] * lens

        dp0[0] = 0
        dp0[1] = nums[1]
        for j in range(2, lens):
            dp0[j] = max(dp0[j-1], nums[j]+dp0[j-2])
        
        max0 = dp0[lens-1]

        
        dp1[0] = nums[0]
        dp1[1] = max(nums[0],nums[1])
        for i in range(2,lens):
            dp1[i] = max(dp1[i-1], nums[i]+dp1[i-2])
        
        max1 = dp1[lens-2]

        #print(max0,max1)

        return max(max0, max1)