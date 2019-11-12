class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        sum_nums = sum(nums)
        if sum_nums % 2:
            return False
        target = sum_nums//2
        #print(target)
        dp = [False] * (100*101)
        dp[0] = True
        #nums.sort()
        for num in nums:
            update = []
            for i in range(target+1):
                if dp[i]:
                    update.append(i+num)
                    #print(i,num,i+num)
            for x in update:
                dp[x] = True
            if dp[target]:
                break
        
        
        return dp[target]