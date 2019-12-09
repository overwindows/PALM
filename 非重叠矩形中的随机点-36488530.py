class Solution:

    def __init__(self, rects: List[List[int]]):
        if len(rects) == 0:
            return
        self._rects = []
        self.rects = rects
        sum = 0
        for x1,y1,x2,y2 in rects:
            sum = (y2-y1+1)*(x2-x1+1)
            self._rects.append(sum)
            #print(sum)
            #if sum > 0:
            #self._rects.append((sum,[x1,y1,x2,y2]))
        #_rects.sort()
        '''
        self.rects = []
        self.rects.append(_rects[0][1])
        for i in range(1,len(_rects)):
            dup = _rects[i][0] // _rects[0][0]
            for _ in range(dup+1):
                self.rects.append(_rects[i][1])
        self.len = len(self.rects)
        #print(len(rects),self.len)
        '''
    def pick(self) -> List[int]:
        #if self.len == 0:
        #    return None
        x0,y0,x1,y1 = random.choices(self.rects, self._rects)[0]
        #ix = random.randint(0,self.len-1)
        #x0,y0,x1,y1 = self.rects[ix]
        
        x = random.randint(x0,x1)
        y = random.randint(y0,y1)
        
        return [x,y]


# Your Solution object will be instantiated and called as such:
# obj = Solution(rects)
# param_1 = obj.pick()