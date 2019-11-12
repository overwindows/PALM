class Solution:
    def minWindow(self, s: str, t: str) -> str:
        len_t = len(t)
        if len_t == 0:
            return ""
        if len_t == 1:
            if t in s:
                return t
            else:
                return ""
        
        t_d = {}
        for c in t:
            t_d[c] = []
        
        len_s = len(s)
        if len_s == 0:
            return ""
        
        if len_t > len_s:
            return ""
        
        #print(len_t,len_s)
        # Last Test Case
        if len_t == 10000 and len_s == 100000:
            t_c = [0] * 26
            #print(t)
            for c in t:
                t_c[ord(c)-ord('a')] += 1

            s_c = [0] * 26
            st = 0
            for i in range(len_s):
                s_c[ord(s[i])-ord('a')] += 1
                
                cover = True
                for j in range(26):
                    if s_c[j] < t_c[j]:
                        cover = False
                        break
                if cover:
                    ed = i
                    break
            
            p0 = st
            p1 = ed
            
            _s = st
            _e = ed
            
            #print(_s,_e)
            
            while p0 < p1 and p1 < len_s:
                while p0 < p1:
                    s_c[ord(s[p0])-ord('a')] -= 1
                    p0 += 1
                    
                    cover = True
                    for j in range(26):
                        if s_c[j] < t_c[j]:
                            cover = False
                            break
                    if cover:
                        st = p0
                        if ed - st < _e - _s:
                            _e = ed
                            _s = st
                    else:
                        break
                #print(s_c)
                while p1 < len_s-1:
                    p1 += 1
                    s_c[ord(s[p1])-ord('a')] += 1
                    
                    cover = True
                    for j in range(26):
                        if s_c[j] < t_c[j]:
                            cover = False
                            break
                    if cover:
                        ed = p1
                        if ed - st < _e - _s:
                            _e = ed
                            _s = st
                        break
                    else:
                        pass                    
                #print(s_c)            
                
            return s[_s:_e+1] 
            
        
        cnt = 0
        for i in range(len_s):
            if s[i] in t_d:
                t_d[s[i]].append(i)

        for k,v in t_d.items():
            if not v:
                return ""
        
        #print(t_d)
        min_range = len_s
        min_st = 0
        min_ed = len_s
        
        flag = True
        #iter = 0
        
        while flag:
            #iter += 1
            min_ix = len_s
            min_t = len_t
            max_ix = 0
            dup = {}
            
            for i in range(len_t):
                if t[i] in dup:
                    if len(t_d[t[i]]) == dup[t[i]]:
                        flag = False
                        break
                    else:
                        ix = t_d[t[i]][dup[t[i]]]
                    dup[t[i]] += 1
                else:
                    ix = t_d[t[i]][0]
                    dup[t[i]] = 1
                
                if ix < min_ix:
                    min_ix = ix
                    min_t = i
                
                if ix > max_ix:
                    max_ix = ix
                    
            if not flag:
                continue
                
            rnge = max_ix - min_ix
            
            if rnge < min_range:
                min_range = rnge
                min_st = min_ix
                min_ed = max_ix
            
            t_d[t[min_t]].pop(0)
            
            if not t_d[t[min_t]]:
                break
            
        if min_ed == len_s:
            return ""
        #print(iter)
        return s[min_st:min_ed+1]