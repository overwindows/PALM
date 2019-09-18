class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums) == 0:
            return True
        path = [False] * len(nums)
        path[0] = True
        
        node = []
        node.append(0)
        while node:
            i = node.pop(0)
            steps = nums[i]
            print(i,steps,i+steps,len(nums)-1)
            if (i+steps) >= (len(nums)-1):
                return True
            if steps == 0:
                return False
            next_step = 0
            max_step = 0
            for step in range(1,steps+1):
                if step + nums[i+step] > max_step:
                    max_step = step+nums[i+step]
                    next_step = step
            #print(i+next_step) 
            node.append(i+next_step)

    
        '''
        for i in range(len(nums)):
            #print(path)
            steps = nums[i]
            if path[i]:
                for step in range(1,steps+1):
                    if i+step < len(nums):
                        path[i+step] = True
                    else:
                        break
        '''
        return path[-1]