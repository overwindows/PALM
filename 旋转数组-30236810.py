class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums) == 1:
            return nums
        
        k = k % len(nums)
        
        def reverse(nums: List[int], start, end) -> None:
            if end-start < 2:
                pass
            else:
                n = (end-start)//2
                for i in range(n):
                    #print(start,end,i)
                    nums[start+i],nums[end-i-1] = nums[end-i-1],nums[start+i]
            
        reverse(nums,0,len(nums))
        reverse(nums,0,k)
        reverse(nums,k,len(nums))
        
        
        
        '''    
        for _ in range(k%len(nums)):
            end = nums[-1]
            lens = len(nums)
            for i in range(1,lens):
                nums[lens-i] = nums[lens-i-1]
            nums[0] = end
        '''
        return nums