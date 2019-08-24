class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        if len(nums) == 0 or nums[-1] < 0:
            return []
        res_set = set()
        for ix in range(len(nums)-2):
            if nums[ix] > 0: 
                return res
            if nums[ix] == 0 and nums[ix+1] == 0 and nums[ix+2] == 0:
                res.append([0,0,0])
                return res
            
            
            target = -nums[ix]
            
            st = ix+1
            end = len(nums)-1
            #print(st,end)
            
            while st < end:
                if nums[st] + nums[end] == target:
                    _res = [nums[ix], nums[st], nums[end]]
                    _key = (nums[ix], nums[st], nums[end])
                    if _key not in res_set:
                        res.append(_res)
                        res_set.add(_key)
                    st += 1
                    end -= 1
                if nums[st] + nums[end] > target:
                    end -= 1
                if nums[st] + nums[end] < target:
                    st += 1
                    
        return res
        