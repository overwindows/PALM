class Solution:
    def countAndSay(self, n: int) -> str:
        ret = [1]
        for _ in range(1,n):
            size = len(ret)
            cnt = 0
            _v = 0
            while size > 0:
                v = ret.pop(0)
                size -= 1
                if _v == v:
                    cnt += 1
                else:
                    if cnt > 0:
                        ret.append(cnt)
                        ret.append(_v)
                    _v = v
                    cnt = 1
                
            if cnt > 0:
                ret.append(cnt)
                ret.append(_v)
                
        #print(ret,list(map(int,ret)))
        return ''.join(list(map(str,ret)))