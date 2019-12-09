class Solution:
    def search(self, nums: List[int], target: int) -> int:
        start = 0
        end = len(nums) -1
        if len(nums) < 4:
            if target in nums:
                return nums.index(target)
            else:
                return -1
        
        while start <= end:
            mid = (end+start)//2
            if nums[mid] == target:
                return mid
            
            if target == nums[start]:
                return start
            if target == nums[end]:
                return end
            
            if target > nums[start] and target < nums[end]:
                #print(nums[start:end+1])
                if target > nums[mid]:
                    start = mid + 1
                else:
                    end = mid - 1
                continue

            if target < nums[start] and target < nums[end]:
                if target < nums[mid] and nums[mid] < nums[end]:
                    end = mid - 1
                else:
                    start = mid + 1                    
                continue
            if target > nums[start] and target > nums[end]:
                if target > nums[mid] and nums[mid] > nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1
                continue
            
            return -1
            
            '''
            if nums[mid] < target:
                if nums[mid+1] > nums[mid-1]:
                #if nums[end] > nums[mid]:
                    # ordered
                    start = mid + 1
                else:
                    # disordered
                    if target <= nums[end]:
                        if target == nums[end]:
                            return end
                        start = mid
                    elif target >= nums[start]:
                        if target == nums[start]:
                            return start
                        end = mid-1
                    else:
                        return -1
            if target < nums[mid]:
                if nums[mid+1] > nums[mid-1]:
                #if nums[mid] > nums[start]:
                    end = mid - 1
                else:
                    if target >= nums[start]:
                        if target == nums[start]:
                            return start
                        end = mid-1
                    elif target <= nums[end]:
                        if target == nums[end]:
                            return end
                        start = mid + 1
                    else:
                        return -1  
            '''
        return -1