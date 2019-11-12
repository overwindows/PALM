class Solution:

    def __init__(self, w: List[int]):
        self.w = {}
        self.w_aux = []
        self.ww = []
        self.len = 0
        for i in range(len(w)):
            st = self.len
            self.len += w[i]
            ed = self.len-1
            self.w[st] = i
            self.w_aux.append(st)
            self.ww.append((i,st,ed))
        
    def pickIndex(self) -> int:
        val = random.randint(0,self.len-1)
        
        s = 0
        e = len(self.w_aux)-1
        
        while s < e:
            m = (s+e)//2
            if self.w_aux[m] > val:
                e = m - 1
            elif self.w_aux[m] < val:
                s = m + 1
            else:
                s = m
                break
        
        if self.w_aux[s] > val:
            return self.w[self.w_aux[s-1]]
        elif self.w_aux[s] < val:
            return self.w[self.w_aux[s]]
        else:
            return self.w[self.w_aux[s]]
    
        '''
        for i,s,e in self.ww:
            if ix>=s and ix<=e:
                return i
        return 0
        '''
        


# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()