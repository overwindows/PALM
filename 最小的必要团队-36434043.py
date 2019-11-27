class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        req_skills_num = len(req_skills)
        req_skills_combo_size = 2**req_skills_num # will return req_skills_combo_size-1. Binary Code for skills.
        people_num = len(people)
        
        skills = [people_num] * req_skills_combo_size # DP
        skills_dict = {} # skill --> hash
        
        #init
        for i in range(req_skills_num):
            skills_dict[req_skills[i]] = 2**i
        
        people_dict = {} # people --> hash --> ix
        people_code = []
        people_skills = {}
        
        ix = 0
        for p in people:
            code = 0
            for x in p: # iterate skills for each person
                code |= skills_dict[x]
            #assert code not in people_dict
            if code not in people_dict:
                people_code.append(code)
                people_dict[code] = len(people_code)-1
            else: # duplicate skills
                people_code.append(0)
            assert ix == len(people_code)-1
            ix += 1
        
        for c in range(1,req_skills_combo_size):        
            for i in range(people_num):
                code = people_code[i]
                if code == 0: # skip no skills person
                    continue     
                if code&c == c: # 'code' contain 'c'
                    '''
                    if c in people_skills and skills[c] == 1:
                        people_skills[c].append([people_dict[code]])
                    else:
                        people_skills[c] = [[people_dict[code]]]
                    assert len(people_skills[c][0]) == 1
                    '''
                    people_skills[c] = [people_dict[code]]
                    skills[c] = 1
                # 'code' not contain 'c', check if 'code' could help.
                else:
                    if c > c^code and skills[c] > 1+skills[c^code]:
                        #if i == 17 and c > 1000:
                        #    print(c, bin(code), bin(skills[c^code], skills[])
                        skills[c] = 1+skills[c^code]
                        people_skills[c] = copy.deepcopy(people_skills[c^code])
                        people_skills[c].append(people_dict[code])
                        #print(c,people_skills)
                        '''
                        for x in people_skills[c]:
                            #assert people_dict[code] == i
                            x.append(people_dict[code])
                        '''
                        '''
                        for i in range(1,c):
                            if i & c == i and skills[i] > skills[c]:
                                people_skills[i] = copy.deepcopy(people_skills[c])
                                skills[i] = skills[c]
                        '''
            #if req_skills_combo_size-1 in people_skills:
            #    print(people_skills[req_skills_combo_size-1])

        #print(skills[req_skills_combo_size-1])
        #print(people_skills[req_skills_combo_size-1])
        #for x in  people_skills[req_skills_combo_size-1][0]:
        #    print("{0:b}".format(people_code[x]).zfill(10))
        #print(people_skills[req_skills_combo_size-1])
        return people_skills[req_skills_combo_size-1]
'''
["hfkbcrslcdjq","jmhobexvmmlyyzk","fjubadocdwaygs","peaqbonzgl","brgjopmm","x","mf","pcfpppaxsxtpixd","ccwfthnjt","xtadkauiqwravo","zezdb","a","rahimgtlopffbwdg","ulqocaijhezwfr","zshbwqdhx","hyxnrujrqykzhizm"]
[["peaqbonzgl","xtadkauiqwravo"],["peaqbonzgl","pcfpppaxsxtpixd","zshbwqdhx"],["x","a"],["a"],["jmhobexvmmlyyzk","fjubadocdwaygs","xtadkauiqwravo","zshbwqdhx"],["fjubadocdwaygs","x","zshbwqdhx"],["x","xtadkauiqwravo"],["x","hyxnrujrqykzhizm"],["peaqbonzgl","x","pcfpppaxsxtpixd","a"],["peaqbonzgl","pcfpppaxsxtpixd"],["a"],["hyxnrujrqykzhizm"],["jmhobexvmmlyyzk"],["hfkbcrslcdjq","xtadkauiqwravo","a","zshbwqdhx"],["peaqbonzgl","mf","a","rahimgtlopffbwdg","zshbwqdhx"],["xtadkauiqwravo"],["fjubadocdwaygs"],["x","a","ulqocaijhezwfr","zshbwqdhx"],["peaqbonzgl"],["pcfpppaxsxtpixd","ulqocaijhezwfr","hyxnrujrqykzhizm"],["a","ulqocaijhezwfr","hyxnrujrqykzhizm"],["a","rahimgtlopffbwdg"],["zshbwqdhx"],["fjubadocdwaygs","peaqbonzgl","brgjopmm","x"],["hyxnrujrqykzhizm"],["jmhobexvmmlyyzk","a","ulqocaijhezwfr"],["peaqbonzgl","x","a","ulqocaijhezwfr","zshbwqdhx"],["mf","pcfpppaxsxtpixd"],["fjubadocdwaygs","ulqocaijhezwfr"],["fjubadocdwaygs","x","a"],["zezdb","hyxnrujrqykzhizm"],["ccwfthnjt","a"],["fjubadocdwaygs","zezdb","a"],[],["peaqbonzgl","ccwfthnjt","hyxnrujrqykzhizm"],["xtadkauiqwravo","hyxnrujrqykzhizm"],["peaqbonzgl","a"],["x","a","hyxnrujrqykzhizm"],["zshbwqdhx"],[],["fjubadocdwaygs","mf","pcfpppaxsxtpixd","zshbwqdhx"],["pcfpppaxsxtpixd","a","zshbwqdhx"],["peaqbonzgl"],["peaqbonzgl","x","ulqocaijhezwfr"],["ulqocaijhezwfr"],["x"],["fjubadocdwaygs","peaqbonzgl"],["fjubadocdwaygs","xtadkauiqwravo"],["pcfpppaxsxtpixd","zshbwqdhx"],["peaqbonzgl","brgjopmm","pcfpppaxsxtpixd","a"],["fjubadocdwaygs","x","mf","ulqocaijhezwfr"],["jmhobexvmmlyyzk","brgjopmm","rahimgtlopffbwdg","hyxnrujrqykzhizm"],["x","ccwfthnjt","hyxnrujrqykzhizm"],["hyxnrujrqykzhizm"],["peaqbonzgl","x","xtadkauiqwravo","ulqocaijhezwfr","hyxnrujrqykzhizm"],["brgjopmm","ulqocaijhezwfr","zshbwqdhx"],["peaqbonzgl","pcfpppaxsxtpixd"],["fjubadocdwaygs","x","a","zshbwqdhx"],["fjubadocdwaygs","peaqbonzgl","x"],["ccwfthnjt"]]
["zp","jpphhnhwpw","pscleb","arn","acrsxqvus","fseqih","fpqbjbbxglivyonn","cjozlkyodt","mvtwffgkhjrtibto","kumdvfwsvrht","i","s","ucr","oo","yqkqkhhhwngyjrg","odiwidzqw"]
[["acrsxqvus"],["zp"],[],["fpqbjbbxglivyonn"],[],[],["kumdvfwsvrht"],[],["oo"],[],["fseqih"],[],["arn"],[],[],["yqkqkhhhwngyjrg"],[],[],[],["kumdvfwsvrht"],["s"],[],[],["zp","ucr"],[],["pscleb"],[],[],[],[],[],[],[],["jpphhnhwpw"],[],[],[],["oo"],[],["i"],["pscleb"],[],[],[],[],[],[],["i"],[],["mvtwffgkhjrtibto","odiwidzqw"],[],["cjozlkyodt","odiwidzqw"],["arn"],[],[],["acrsxqvus"],[],[],[],["ucr"]]
'''