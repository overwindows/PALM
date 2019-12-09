class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        set_num = set()
        for num in nums:
            set_num.add(num)
        
        max_cnt = 0
        cnt = 0
        q = []
        while set_num:
            x = set_num.pop()
            q.append(x)
            cnt = 0
            while q:
                y = q.pop()
                cnt += 1
                if (y+1) in set_num:
                    set_num.remove(y+1)
                    q.append(y+1)
                    
                if (y-1) in set_num:
                    set_num.remove(y-1)
                    q.append(y-1)
            max_cnt = max(max_cnt, cnt)
        
        return max_cnt