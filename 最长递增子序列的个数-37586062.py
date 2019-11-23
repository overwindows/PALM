class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        lens = len(nums)
        if lens == 0:
            return 0
        dp = [1] * lens
        cnt = [1] * lens

        for i in range(lens):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[i] < dp[j]+1:
                        dp[i] = dp[j]+1
                        cnt[i] = cnt[j]
                    elif dp[i] == dp[j]+1:
                        cnt[i] += cnt[j]
                    #dp[i] = max(dp[i], dp[j]+1)
        _max = max(dp)
        #print(_max, dp, cnt)
        _max_cnt = 0
        for i in range(lens):
            if dp[i] == _max:
                _max_cnt += cnt[i]

        return _max_cnt