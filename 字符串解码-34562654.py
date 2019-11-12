class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack_n = []
        #stack_t = []
        stack_s = []
        i = 0
        flag = False
        while i < len(s):
            n = ''
            while s[i] >= '0' and s[i] <= '9':
                n += s[i]
                i+=1
                
            if n:
                stack_n.append(int(n))
                flag = True
                continue
                
            if s[i] == '[':
                stack_s.append(s[i])
                i+=1
                continue
                
            if s[i] == ']':
                flag = False
                i+=1
                #stack_s.pop()
                n = stack_n.pop()
                _s = stack_s.pop()
                _s = _s*n
                
                c = stack_s.pop()
                assert c == '['
                
                if stack_s and stack_s[-1] != '[':
                    stack_s[-1] = stack_s[-1] + _s
                else:
                    stack_s.append(_s)
                #print(stack_t)
                continue
                
            _s = ''
            #print(s, i)
            while (s[i] >='a' and s[i]<='z') or (s[i]>='A' and s[i]<='Z'):
                _s += s[i]
                i+=1
                if not i<len(s):
                    break
            
            if _s:
                if flag:
                    #print(stack_t,stack_s,stack_n,_s)
                    stack_s.append(_s)
                    flag = False
                else:
                    #print(stack_t,stack_s,stack_n,_s)
                    if stack_s:
                        stack_s[-1] = stack_s[-1] + _s
                    else:
                        stack_s.append(_s)
                #print(stack_s)
        #print(stack_s)
        return ''.join(stack_s)