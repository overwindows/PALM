class Solution:
    def romanToInt(self, s: str) -> int:
        d = {}
        d['I'] = 1
        d['V'] = 5
        d['X'] = 10
        d['L'] = 50
        d['C'] = 100
        d['D'] = 500
        d['M'] = 1000

        Int = 0
        i = 0
        while i < len(s):
            if s[i] == 'I':
                if i+1 < len(s):
                    if s[i+1] == 'V' or s[i+1] == 'X':
                        Int = Int - d[s[i]] + d[s[i+1]]
                        i += 1
                        
                    else:
                        Int += d[s[i]]
                else:
                    Int += d[s[i]]
                
            elif s[i] == 'X':
                if i+1 < len(s):
                    if s[i+1] == 'L' or s[i+1] == 'C':
                        Int = Int - d[s[i]] + d[s[i+1]]
                        i += 1
                    else:
                        Int += d[s[i]]
                else:
                    Int += d[s[i]]
            elif s[i] == 'C':
                if i+1 < len(s):
                    if s[i+1] == 'D' or s[i+1] == 'M':
                        Int = Int - d[s[i]] + d[s[i+1]]
                        i += 1
                        
                    else:
                        Int += d[s[i]]
                else:
                    Int += d[s[i]]
            else:
                Int += d[s[i]]
            i += 1

        return Int