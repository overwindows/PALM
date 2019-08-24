class Solution:
    def findTargetSumWays(self, nums: List[int], S: int) -> int:
        dp = [0] * 1001
        sum_nums = sum(nums)
        s = (S+sum_nums)//2
        
        if (S+sum_nums) % 2 != 0 or (sum_nums < S):
            return 0
        
        dp[0] = 1
        for n in nums:
            for i in range(s,n-1,-1):
                dp[i] += dp[i-n]
        
        '''
        dup = [0] * 1001
        for x in nums:
            dup[x] += 1
        
        SEP = 1001
        
        BFS = [0,1001]
        for x in nums:
            if x == 0:
                continue
            num = BFS.pop(0)
            while num != SEP:
                BFS.append(num+x)
                BFS.append(num-x)
                num = BFS.pop(0)
            BFS.append(SEP)
        cnt = 0
        for x in BFS:
            if x == S:
                cnt += 1
        cnt *= (2**dup[0])
        '''
        return dp[s]
            
            
        