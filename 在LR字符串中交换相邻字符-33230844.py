class Solution:
    def canTransform(self, start: str, end: str) -> bool:
        if len(start) != len(end):
            return False
        
        _start = []
        _end = []
        
        
        for c_i in range(len(start)):
            if start[c_i] != 'X':
                _start.append((start[c_i],c_i))
        
        for c_i in range(len(end)):
            if end[c_i] !='X':
                _end.append((end[c_i],c_i))
        
        if len(_start) != len(_end):
            return False
        
        print(_start,_end)
        for i in range(len(_start)):
            e, e_i = _end[i]
            s, s_i = _start[i]
            
            if e == s:
                if s == 'R' and s_i <= e_i:
                    pass
                elif s == 'L' and s_i >= e_i:
                    pass
                else:
                    return False
            else:
                return False
        
        return True