class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        if len(nums) == 0:
            return nums
        nums.sort()
        dp = [[nums[i]] for i in range(len(nums))]
        max_size = 0
        max_idx = 0
    
        for i in range(len(nums)):
            for j in range(1,i+1):
                if nums[i] % nums[i-j] == 0:
                    if len(dp[i]) < len(dp[i-j])+1:
                        dp[i] = dp[i-j][:]
                        dp[i].append(nums[i])
                    
                        if max_size < len(dp[i]):
                            max_idx = i
                            max_size = len(dp[i])

        #print(dp)
        return dp[max_idx]

'''
[2,3,8,9,27]
[3,4,16,8]
[4,8,10,240]
[]
[1]
[2,3]
[1,2,3]
[1,2,3,4,8]
[1,2,4,8]
'''