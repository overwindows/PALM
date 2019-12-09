class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        N = len(nums)
        dp_pos = [1]*N 
        dp_neg = [1]*N 

        if N == 0:
            return 0
        if N == 1:
            return 1
        #if N == 2:
        #    return 2

        for i in range(1,N):
            delta = nums[i] - nums[i-1]
            if delta > 0:
                dp_neg[i] = dp_pos[i-1]+1
                dp_pos[i] = dp_pos[i-1]
            elif delta < 0:
                dp_pos[i] = dp_neg[i-1]+1
                dp_neg[i] = dp_neg[i-1]
            else:
                dp_pos[i] = dp_pos[i-1]
                dp_neg[i] = dp_neg[i-1]
        #print(dp_pos,dp_neg)
        return max(max(dp_pos),max(dp_neg))