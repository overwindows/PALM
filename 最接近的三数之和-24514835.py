class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        closest_sum_list = []
        for x in nums:
            _nums = nums.copy()
            _nums.remove(x)
            _target = target-x
            
            st = 0
            ed = len(_nums)-1
            
            # init
            closest_sum = _nums[st] + _nums[ed]            
            
            while st < ed:
                _sum = _nums[st] + _nums[ed]
                if abs(closest_sum - _target) > abs(_sum - _target):
                        closest_sum = _sum
                if _sum == _target:
                    closest_sum = _target
                    break
                if _sum > _target:
                    if abs(closest_sum-_target) > abs(_nums[st+1]+_nums[ed]-_target):
                        closest_sum = _nums[st+1]+_nums[ed]      
                    ed -= 1
                if _sum < _target:
                    if abs(closest_sum-_target) > abs(_nums[st]+_nums[ed-1]-_target):
                        closest_sum = _nums[st]+_nums[ed-1]
                    st += 1
                #print(x,closest_sum, _target)
            closest_sum_list.append(closest_sum + x)
        _min = abs(closest_sum_list[0] - target)
        #print(closest_sum_list, _min)
        _m = closest_sum_list[0] 
        for m in closest_sum_list:
            if abs(m - target) < _min:
                _min = abs(m - target)
                _m =m
        return _m
                    
            