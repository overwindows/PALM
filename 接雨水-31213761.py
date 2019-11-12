class Solution:
    def trap(self, height: List[int]) -> int:
        if len(height) < 3:
            return 0
            
        h0 = [0] * len(height)
        h1 = [0] * len(height)
        
        init0 = 1
        for i in range(len(height)):
            if height[i] >= init0:
                init0 = height[i]
                h0[i] = init0
        
        t0 = 0
        M = 0
        
        while h0[-1] == 0:
            h0.pop()
        
        for i in range(len(h0)):
            
            if h0[i] > 0:
                t0 = h0[i]
            else:
                if t0 == 0:
                    pass
                else:
                    M += (t0-height[i])
        #print(h0,M)       
        
        init1 = 0
        height.reverse()
        for i in range(len(height)):
            if height[i] > init1:
                init1 = height[i]
                h1[i] = init1
          
        t1 = 0
        N = 0
        
        while h1 and h1[-1] == 0:
            h1.pop()
        
        for i in range(len(h1)):
            
            if h1[i] > 0:
                t1 = h1[i]
            else:
                if t1 == 0:
                    pass
                else:
                    N += (t1-height[i])
        
        #print(height,h1,N)
        
        return M+N
                
                
            