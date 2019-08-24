class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        ret = []
        #print(nums)
        for i in range(len(nums)-3):
            if (nums[i] > 0 and nums[i] > target):
                return ret
            for j in range(i+1,len(nums)-2):
                if (nums[j] > 0 and nums[i]+nums[j] > target):
                    # print(nums[i],nums[j])
                    continue
                
                t = target - (nums[j]+nums[i])
                s = j+1
                e = len(nums)-1
                
                while s < e:
                    if nums[s] + nums[e] == t:
                        res = [nums[i],nums[j],nums[s],nums[e]]
                        if res not in ret:
                            ret.append(res)
                        s += 1
                        e -= 1
                    if nums[s] + nums[e] > t:
                        e -= 1
                    if nums[s] + nums[e] < t:
                        s += 1
                '''
                for k in range(j+1,len(nums)):
                    if nums[k] > 0 and nums[i]+nums[j]+nums[k] > target:
                       continue
                    for l in range(k+1,len(nums)):
                        # print(nums[i],nums[j],nums[k],nums[l])
                        if nums[i]+nums[j]+nums[k]+nums[l] == target:
                            res = [nums[i],nums[j],nums[k],nums[l]]
                            if res not in ret:
                                ret.append(res)
                '''
        return ret