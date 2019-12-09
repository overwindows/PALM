class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        nums.sort()
        if target == 0:
            return 1
        dp = [0] * (1+target)
        N = len(nums)
        for t in range(1,target+1):
            for i in range(N):
                if nums[i] == t:
                    dp[t] += 1
                if nums[i] < t:
                    dp[t] += dp[t-nums[i]]
        return dp[target]

