from math import *
import statistics

def phi(z):
    #'Cumulative distribution function for the standard normal distribution'
    ##pass the z value this will return cumulative 1 tailed probability to Z
    return (1.0 + erf(z / sqrt(2.0))) / 2.0


def dist_intersect(dist1_mu, dist1_sig, dist2_mu, dist2_sig):
    #returns the intersection probability of two distributions
    step_sig = max(dist1_sig, dist2_sig)
    start_mu = min(dist1_mu, dist2_mu)
    end_mu = max(dist1_mu, dist2_mu)
    step = 6 * step_sig / 10

    #print("mu diff in sigma is ", abs(dist1_mu  - dist2_mu)/min(dist1_sig, dist2_sig))
    #print(dist1_mu , dist2_mu,  dist1_sig, dist2_sig)

    #if abs(dist1_mu  - dist2_mu) > 6 * min(dist1_sig, dist2_sig):
    #    return 0.01  ##return a small positive value if the two distributions are too far apart

    startx = start_mu - 6 * step_sig
    endx = end_mu + 6 * step_sig
    currx = startx
    int_prob = 0  ##intersection probability
    while(currx < endx):
        refz1 = (currx - dist1_mu) / dist1_sig
        refz2 = ((currx+step) - dist1_mu) / dist1_sig
        p1 = phi(refz1)
        p2 = phi(refz2)
        prob1 = abs(p2-p1)

        refz1 = (currx - dist2_mu) / dist2_sig
        refz2 = ((currx+step) - dist2_mu) / dist2_sig
        p1 = phi(refz1)
        p2 = phi(refz2)
        prob2 = abs(p2-p1)

        int_prob += min(prob1,prob2)
        currx += step

    return int_prob

def est_sig(rng, prob, map_probs):
    ##estimates the sigma by determining the standard normal two
    prob = round(prob,2)
    if prob in map_probs.keys():
        #print(" range is ", rng, " prob is ", prob, " sigma is ", map_probs[prob] * rng)
        return map_probs[prob] * rng  ##this is the estimate of sigma
    else:
        if prob <= 0 or prob > 1:
            return None
    

def gen_dict():
    map_probs = {}
    for x in range(1,350):
        prob = round(phi(x/100)-phi(-x/100),2)
        #print("x:", x/100, ", prob:", round(phi(x/100)-phi(-x/100),2) , ",")
        if prob not in map_probs.keys():
            map_probs[prob] = x/100

    return map_probs

def intersect_or(means1, means2, range1, range2, prob, map_probs):
    ##if collecting middle 3 out of 7 values, prob is 3/7
    ##means1 is the 12 means of the 3 points above, range1 is the +/- range of the 3 points from above
    ##means2 is the 12 means of the 3 points above, range1 is the +/- range of the 3 points from above
    mult = 1
    probs = []
    cnt = 0
    nan_cnt = 0
    tot = 0

    mean_count1 = 0
    mean_count2 = 0
    for x in range(0,len(means1)):
        sig1 = est_sig(range1[x], prob, map_probs)
        sig2 = est_sig(range2[x], prob, map_probs)
        if sig1 == 0 or sig2 ==0:
            probs.append("NAN")
            nan_cnt += 1
            continue;
        
        int_prob = dist_intersect(means1[x], sig1, means2[x], sig2)
        if int_prob == 0:
            probs.append("NAN")
            nan_cnt += 1
            continue;
        #print(sig1, sig2, "int_prob", int_prob)
        mult += int_prob
        tot += int_prob
        cnt += 1
        probs.append(int_prob)
        mean_count1 += means1[x]
        mean_count2 += means2[x]

    mult = mult / cnt

    return mult, probs



def intersect(means1, means2, range1, range2, prob, map_probs):
    ##if collecting middle 3 out of 7 values, prob is 3/7
    ##means1 is the 12 means of the 3 points above, range1 is the +/- range of the 3 points from above
    ##means2 is the 12 means of the 3 points above, range1 is the +/- range of the 3 points from above
    mult = 1
    probs = []
    cnt = 0
    nan_cnt = 0
    tot = 0
    for x in range(0,len(means1)):
        sig1 = est_sig(range1[x], prob, map_probs)
        sig2 = est_sig(range2[x], prob, map_probs)
        if sig1 == 0 or sig2 ==0:
            probs.append("NAN")
            nan_cnt += 1
            continue;
        
        int_prob = dist_intersect(means1[x], sig1, means2[x], sig2)
        if int_prob == 0:
            probs.append("NAN")
            nan_cnt += 1
            continue;
        #print(sig1, sig2, "int_prob", int_prob)
        mult *= int_prob
        tot += int_prob
        cnt += 1
        probs.append(int_prob)

    # avg_prob = tot / cnt
    avg_prob = tot / cnt
    if nan_cnt > 0:
        mult *= 10**(-nan_cnt)
    # if nan_cnt > 0:
    #     mult *= avg_prob**nan_cnt

    

    # print("mult is ", mult, probs)
    # print("mult is ", mult)
    # return mult, probs
    return mult




def merge_lists(list1, list2):
    # Ensure both lists are of the same length
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length")

    # Merge lists index-wise
    merged_list = [list1[i] + list2[i] for i in range(len(list1))]
    return merged_list


class Frame:
    def __init__(self, mean_local, range_local, parts_local):
        self.mean_local = mean_local
        self.range_local = range_local
        self.parts_local = parts_local

    def __eq__(self, other):
        if not isinstance(other, Frame):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return (self.mean_local == other.mean_local and 
                self.range_local == other.range_local and 
                self.parts_local == other.parts_local)


def merge(merge_parts):
    result_part=[]
    ranges=[]
    means=[]
    for i in merge_parts:
        temp=sorted(i)
        n=len(temp)
        # print(n)
        low_bound=round(0.4*n)
        up_bound=round(0.6*n)
        # print('temp is: ',temp, low_bound,up_bound)
        if n>5:
            range1=temp[up_bound]-temp[low_bound]
            # mean1=statistics.median(temp[low_bound:up_bound])
            mean1=statistics.median(temp)
        elif n==5:
            range1=(temp[3]-temp[1])/2
            mean1=temp[2]
        elif n==4:
            range1=(temp[2]-temp[1])/2
            mean1=(temp[2]+temp[1])/2
        elif n==3:
            range1=0.5
            mean1=temp[1]
        elif n==2:
            range1=0.5
            mean1=(temp[1]+temp[0])/2
        elif n==1:
            range1=0.5
            mean1=temp[0]
        else:
            range1=0.5
            mean1=0
            
        if range1<0.5:
            range1=0.5
        
        if range1>1.5:
            range1=1.5

        result_part.append(temp)
        ranges.append(range1)
        means.append(mean1)

    return Frame(means, ranges, result_part)



def get_group_frame(person_list):
    
    person=[]
    
    for i in range(12):
        part=[]
        for j in range(len(person_list)):
            if person_list[j][i]!=0:
                part.append(person_list[j][i])
        person.append(sorted(part))
    print(person)
    frame=merge(person)
    return frame

ret_val = dist_intersect(1, 1, 6, 1)
# print(ret_val)
map_probs  = gen_dict()  ##get the map of two tailed probabilities

distri=0.5


person1_frame7_mean=[33.07597961425781, 22.79368896484375, 30.333639526367186, 26.813742065429686, 24.153294372558594, 33.56095581054687, 27.01553955078125, 35.01539306640625, 30.152960205078124, 46.26310729980469, 45.985833740234376, 20.969447326660156]
person1_frame7_range=[1.990182495117189, 4, 4, 4, 2.144461059570311, 2.6675384521484418, 1.1403137207031264, 3.1383728027343736, 4, 1.7947753906249986, 4, 4]

person1_frame8_mean=[30.0996337890625, 22.673858642578125, 19.77113037109375, 26.7338623046875, 25.13885040283203, 33.13398132324219, 23.651487731933592, 33.08494567871094, 31.24925842285156, 45.44921875, 47.093316650390626, 18.1104248046875]
person1_frame8_range=[3.0073516845703097, 2.043894958496093, 3.050367736816405, 3.319830322265627, 1.7269577026367209, 3.783831787109378, 2.394825744628907, 3.062835693359375, 1.1656036376953125, 4, 4, 1.333770751953125]

person1_frame12_mean=[33.440451049804686, 26.57593688964844, 25.587922668457033, 28.248629760742187, 23.277873229980468, 33.186712646484374, 27.463194274902342, 35.390444946289065, 581.4769287109375, 47.31827087402344, 48.23824462890625, 20.03370361328125]
person1_frame12_range=[1.7937591552734347, 3.8080261230468757, 0.846546936035157, 2.004220581054689, 4, 1.79833984375, 1.0926544189453118, 0.8001708984375, 4, 4, 1.6116302490234347, 1.7763793945312507]

person1_frame13_mean=[33.63212585449219, 25.98887939453125, 27.256683349609375, 25.959649658203126, 24.95164337158203, 36.07652282714844, 32.767947387695315, 35.6219482421875, 22.55382080078125, 45.61631469726562, 47.78746032714844, 20.611395263671874]
person1_frame13_range= [1.4294952392578182, 1.4193481445312521, 2.1906860351562507, 1.9044235229492195, 1.5150115966796847, 1.411752319335939, 0.5, 4, 0.5, 1.8265747070312486, 1.5807952880859375, 1.7734924316406264]

person1_frame14_mean=[32.6603515625, 27.958526611328125, 24.579209899902345, 27.17713623046875, 26.004534912109374, 34.843743896484376, 0, 34.254513549804685, 27.936495971679687, 47.9951171875, 47.748016357421875, 19.981289672851563]
person1_frame14_range= [1.653005981445311, 1.6278839111328125, 2.661070251464846, 1.0728698730468729, 0.7879623413085923, 2.1956329345703125, 0, 1.3620056152343807, 2.1728576660156236, 1.9563934326171903, 2.0782836914062486, 2.236749267578123]

#----------------------------------

person2_frame57_mean=[32.010064697265626, 23.03443298339844, 20.51402587890625, 25.781439208984374, 26.52181396484375, 35.34126586914063, 32.69452819824219, 30.556661987304686, 31.639016723632814, 48.12599182128906, 50.084649658203126, 21.551971435546875]
person2_frame57_range= [4, 3.2913589477539062, 4, 2.011468505859373, 4, 1.928445434570314, 4, 1.9162353515624986, 2.826904296875, 3.736001586914064, 2.358743286132814, 1.9900711059570284]

person2_frame58_mean=[32.758847045898435, 25.043742370605468, 20.28502197265625, 24.366871643066407, 18.418362426757813, 36.71141357421875, 404.8851318359375, 35.10899963378906, 53.05157470703125, 48.098873901367185, 50.92255859375, 20.739682006835938]
person2_frame58_range= [3.0647918701171903, 3.073466491699218, 3.8229660034179673, 3.8931106567382834, 0.6971832275390639, 0.8140533447265597, 1.8278076171874886, 2.016409301757818, 4, 3.1405059814453082, 2.7649688720703125, 2.0890426635742188]

person2_frame59_mean=[31.91904296875, 25.532432556152344, 18.476585388183594, 24.334027099609376, 19.015509033203124, 37.2513671875, 36.16116790771484, 34.40577697753906, 32.71627502441406, 48.05036315917969, 49.34334716796875, 21.867486572265626]
person2_frame59_range=[3.6225524902343764, 4, 4, 1.8315277099609375, 1.8910583496093736, 1.8942626953125057, 4, 3.4407226562499957, 2.0698516845703097, 0.9601837158203139, 3.8687072753906193, 2.885830688476563]

person2_frame60_mean=[35.057681274414065, 23.134722900390628, 19.459364318847655, 22.701979064941405, 22.660989379882814, 37.5898193359375, 185.41871337890626, 35.890997314453124, 158.13209228515626, 50.746377563476564, 49.793243408203125, 22.217291259765624]
person2_frame60_range=[4, 3.525881958007812, 1.273640441894532, 3.900045776367186, 3.954165649414062, 3.0661590576171847, 3.7811401367187614, 1.8244201660156207, 0.5, 2.5428527832031236, 2.791912841796872, 2.560017395019532]



print('compare person1frame7 to person2allframes: ')
print(intersect(person1_frame7_mean, person2_frame57_mean, person1_frame7_range, person2_frame57_range, distri, map_probs))
print(intersect(person1_frame7_mean, person2_frame58_mean, person1_frame7_range, person2_frame58_range, distri, map_probs))
print(intersect(person1_frame7_mean, person2_frame59_mean, person1_frame7_range, person2_frame59_range, distri, map_probs))
print(intersect(person1_frame7_mean, person2_frame60_mean, person1_frame7_range, person2_frame60_range, distri, map_probs))

print('----------------------')

print('compare person1frame8 to person2allframes: ')
print(intersect(person1_frame8_mean, person2_frame57_mean, person1_frame8_range, person2_frame57_range, distri, map_probs))
print(intersect(person1_frame8_mean, person2_frame58_mean, person1_frame8_range, person2_frame58_range, distri, map_probs))
print(intersect(person1_frame8_mean, person2_frame59_mean, person1_frame8_range, person2_frame59_range, distri, map_probs))
print(intersect(person1_frame8_mean, person2_frame60_mean, person1_frame8_range, person2_frame60_range, distri, map_probs))

print('----------------------')

print('compare person1frame12 to person2allframes: ')
print(intersect(person1_frame12_mean, person2_frame57_mean, person1_frame12_range, person2_frame57_range, distri, map_probs))
print(intersect(person1_frame12_mean, person2_frame58_mean, person1_frame12_range, person2_frame58_range, distri, map_probs))
print(intersect(person1_frame12_mean, person2_frame59_mean, person1_frame12_range, person2_frame59_range, distri, map_probs))
print(intersect(person1_frame12_mean, person2_frame60_mean, person1_frame12_range, person2_frame60_range, distri, map_probs))

print('----------------------')

print('compare person1frame13 to person2allframes: ')
print(intersect(person1_frame13_mean, person2_frame57_mean, person1_frame13_range, person2_frame57_range, distri, map_probs))
print(intersect(person1_frame13_mean, person2_frame58_mean, person1_frame13_range, person2_frame58_range, distri, map_probs))
print(intersect(person1_frame13_mean, person2_frame59_mean, person1_frame13_range, person2_frame59_range, distri, map_probs))
print(intersect(person1_frame13_mean, person2_frame60_mean, person1_frame13_range, person2_frame60_range, distri, map_probs))

print('----------------------')

print('compare person1frame14 to person2allframes: ')
print(intersect(person1_frame14_mean, person2_frame57_mean, person1_frame14_range, person2_frame57_range, distri, map_probs))
print(intersect(person1_frame14_mean, person2_frame58_mean, person1_frame14_range, person2_frame58_range, distri, map_probs))
print(intersect(person1_frame14_mean, person2_frame59_mean, person1_frame14_range, person2_frame59_range, distri, map_probs))
print(intersect(person1_frame14_mean, person2_frame60_mean, person1_frame14_range, person2_frame60_range, distri, map_probs))

