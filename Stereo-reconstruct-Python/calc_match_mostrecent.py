from math import *

def phi(z):
    #'Cumulative distribution function for the standard normal distribution'
    ##pass the z value this will return cumulative 1 tailed probability to Z
    return (1.0 + erf(z / sqrt(2.0))) / 2.0

def point_match(dist1_mu, dist1_sig, val):
    ##takes a two tailed match of how close the value is to the mean.
    ## eg. 1 - P value
    z = (val - dist1_mu) / dist1_sig
    z = abs(z)  ## flip it around so always more than 0.5
    cum_prob = phi(z)
    return 1 - cum_prob
    
    


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
        return 0.5 * rng / map_probs[prob]  ##this is the estimate of sigma
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

def intersect(means1, means2, range1, range2, prob, map_probs):
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
        mult *= int_prob
        tot += int_prob
        cnt += 1
        probs.append(int_prob)
        mean_count1 += means1[x]
        mean_count2 += means2[x]

    avg_prob = tot / cnt
    if nan_cnt > 0:
        mult *= 10**(-1*nan_cnt)

    ##mult = ((mean_count2 - mean_count1)**2/(mean_count1+mean_count2)) * mult 

    # print("mult is ", mult, probs)
    return mult, probs


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


#above intersection function


#above intersection function



ret_val = dist_intersect(1, 1, 6, 1)
print(ret_val)
map_probs  = gen_dict()  ##get the map of two tailed probabilities

distri=0.677

mean1=[32.51783905029297, 22.714483642578124, 18.473861694335938, 23.965794372558594, 20.531011962890624, 36.12490539550781, 35.75204772949219, 34.6367431640625, 32.66388854980469, 48.96383972167969, 50.294253540039065, 20.608946228027342]
range1=[1.5, 0.7120620727539055, 1.5, 0.5262313842773452, 0.833995056152343, 1.004507446289061, 0.8336090087890611, 1.1875183105468778, 1.5, 1.3794799804687514, 1.5, 0.5]

mean2=[31.859530639648437, 24.62757110595703, 20.4177490234375, 22.398268127441405, 22.508877563476563, 36.436947631835935, 36.81101684570312, 37.32706909179687, 35.57789001464844, 45.0595718383789, 47.786416625976564, 19.50404052734375]
range2=[0.5, 0.8721343994140653, 1.0946456909179716, 1.4546157836914055, 1.5, 0.5, 0.9813110351562528, 0.9918975830078125, 1.5, 1.5, 0.7932556152343722, 0.7743011474609389]


print('11111111111111: ')
print(intersect(mean1, mean2, range1, range2, distri, map_probs))
