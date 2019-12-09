class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        if len(nums) < 3:
            return False
        
        while len(nums) > 3 and nums[0] > nums[1]:
            nums.pop(0)
        
        if len(nums) < 3:
            return False


        
        dp = [0]*len(nums)

        for i in range(1,len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[j]+1,dp[i])
        
        for l in dp:
            if l > 1:
                return True
        
        return False
        
        