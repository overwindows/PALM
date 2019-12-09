class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        if len(nums) < 2:
            return nums
        
        def partition(nums: List[int], mid: int, st: int, ed: int):
            pivot = nums[mid]
            while st < ed:
                while st < ed and nums[st] < pivot:
                    st += 1
                #assert nums[st] > pivot
                if st < ed:
                    while st < ed and nums[ed] > pivot:
                        ed -= 1
                    #assert nums[ed] <= pivot
                    if st < ed:
                        pass
                    else:
                        break
                else:
                    break
                if nums[st] == nums[ed]:
                    st += 1
                else:
                    nums[st],nums[ed] = nums[ed],nums[st]
            
            #if nums[st] == pivot:
            #    return st
            #else:
            return st
        
        def qsort(nums: List[int],lbound,ubound):
            mid = (lbound+ubound)//2
            #print(nums,nums[mid])
            mid = partition(nums,mid,lbound,ubound)
            #print(nums[mid])
            if mid-1 > lbound:
                qsort(nums,lbound,mid-1)
            if mid+1 < ubound:
                qsort(nums,mid+1,ubound)
        
        qsort(nums,0,len(nums)-1)
        return nums
                