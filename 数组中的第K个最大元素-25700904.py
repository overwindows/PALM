class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        st = 0
        ed = len(nums)-1
        _k = len(nums)-k
        while True:
            pivot = nums[st]
            #print(nums[st:ed+1])
            while True:
                while pivot <= nums[ed] and st < ed:
                    ed -= 1

                if st < ed:
                    nums[st] = nums[ed]
                    st += 1
                else:
                    break

                while pivot > nums[st] and st < ed:
                    st += 1

                if st < ed:
                    nums[ed] = nums[st]
                    ed -= 1
                else:
                    break
            
            nums[st] = pivot
            #print(nums,pivot,st,_k)
            
            if st < _k:
                st = st + 1
                ed = len(nums)-1
            elif st > _k:
                ed = st -1
                st = 0
            else:
                return pivot
            continue 