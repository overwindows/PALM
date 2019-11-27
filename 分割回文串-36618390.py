class Solution:
    def partition(self, s: str) -> List[List[str]]:
        s = list(s)
        lens = len(s)
        if lens == 1:
            return [s]
        ret = []
        for i in range(lens):
            #self.partition('s')
            if i == 0:
                left = s[i]
            else:
                st = 0 
                ed = i
                while st < ed and s[st] == s[ed]:
                    st += 1
                    ed -= 1
                if st < ed:
                    continue
                else:
                    left = ''.join(s[:i+1])
            
            right = self.partition(''.join(s[i+1:]))
            if len(right):
                for r in right:
                    ret.append([left]+r)
            else:
                ret.append([left])
        return ret


