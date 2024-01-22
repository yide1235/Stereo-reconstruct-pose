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


mean1=
range2=

mean2=
range2=

print('compare person1frame7 to person2allframes: ')
print(intersect(mean1, mean2, range1, range2, distri, map_probs))
