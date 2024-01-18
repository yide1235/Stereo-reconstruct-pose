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

    print("mult is ", mult, probs)
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


ret_val = dist_intersect(1, 1, 6, 1)
print(ret_val)
map_probs  = gen_dict()  ##get the map of two tailed probabilities

print(point_match(0, 1, -2))
'''
means1 = [32.13058166503906, 25.295718383789062, 25.491850280761717, 27.271368408203124, 24.351949310302736, 33.56095581054687, 26.9886962890625, 34.9342041015625, 30.268148803710936, 46.349002075195315, 47.36042785644531, 20.035533142089843]
range1=[0.8239501953125021, 0.8646194458007805, 1.4896102905273452, 0.5, 1.2689804077148423, 1.5, 1.5, 0.6897949218749986, 1.2020507812500014, 0.8130249023437486, 1.0358673095703068, 0.6020858764648445]

means2=[32.110101318359376, 23.1695556640625, 18.78723907470703, 25.118328857421876, 20.03465118408203, 35.939239501953125, 402.0792724609375, 33.623876953125, 32.35946807861328, 48.73312072753906, 49.9539794921875, 20.72131805419922]
range2=[0.5469451904296854, 1.1086364746093764, 1.4272705078125014, 0.8860549926757812, 1.3944244384765625, 1.0052642822265625, 1.5, 1.108938598632811, 1.5, 0.9799560546874986, 1.5, 0.9533462524414062]
'''

means1 =[31.909490966796874, 22.8201171875, 18.382177734375, 24.38310546875, 20.373094177246095, 37.12849578857421, 77.48353271484375, 34.765628051757815, 32.74283142089844, 49.60035247802735, 50.11253051757812, 20.81147918701172]
range1=[1.5, 1.5, 1.5, 0.5, 0.9336868286132791, 0.7793426513671875, 1.5, 0.9670745849609403, 1.5, 1.0522918701171875, 0.880035400390625, 0.9688461303710909]

means2=[33.086376953125, 23.339700317382814, 18.382177734375, 23.93510284423828, 20.14325866699219, 37.10711669921875, 37.04456634521485, 35.167840576171876, 34.94752197265625, 49.72235107421875, 50.65043334960937, 20.623877716064456]
range2=[1.1939666748046918, 1.3844650268554695, 1.1762756347656236, 1.1706527709960923, 1.5, 1.1026611328125, 0.8463775634765653, 0.5, 0.8116455078125, 0.9663482666015639, 1.051675415039064, 0.9584396362304695]

#means1=[35.0259765625, 22.900640869140624, 18.343902587890625, 24.651348876953126, 21.368550109863282, 38.3728759765625, 188.24010009765624, 35.280111694335936, 148.6227233886719, 49.72235107421875, 50.851376342773435, 22.010946655273436] 
#range1=[0.8630798339843722, 1.5033996582031257, 0.5, 2.18022003173828, 3.6639572143554666, 1.5482940673828125, 4, 1.7222747802734375, 4, 1.8236877441406278, 1.0513122558593722, 2.8668243408203153]

#means2=[33.06582946777344, 26.3018310546875, 20.88753662109375, 22.050022888183594, 22.499449157714842, 35.957525634765624, 36.878640747070314, 35.718194580078126, 34.99725341796875, 50.669430541992185, 52.75661010742188, 19.808670043945312]
#range2=[1.2495605468749957, 4, 2.615153503417968, 2.431315612792968, 4, 3.9769927978515582, 2.689178466796875, 2.197161865234378, 0.5, 2.1281402587890597, 2.023480224609372, 3.055245971679689]

'''
means1 = [33.0146484375, 24.357485961914062, 19.47406997680664, 26.2373291015625, 19.318236541748046, 35.09274597167969, 35.8320327758789, 34.09553527832031, 33.218989562988284, 49.71528930664063, 51.746057128906244, 21.437864685058592]
range1=[1.0702453613281264, 0.6469314575195284, 2.1997589111328146, 1.5575027465820312, 2.6679275512695284, 0.5, 4, 0.5, 3.2592712402343764, 0.5, 0.89154052734375, 0.8846771240234368]

means2=[29.61957550048828, 27.673895263671874, 23.57106170654297, 40.10067291259766, 29.844113159179688, 32.21791381835938, 32.27177429199219, 35.0269775390625, 39.954141235351564, 47.38838500976563, 44.33758697509766, 19.854644775390625]
range2=[1.440377807617189, 0.8752044677734361, 1.4419021606445312, 1.230477905273439, 4, 0.8713562011718707, 1.1716629028320291, 1.4515441894531236, 0.5, 1.4452636718749972, 2.575204467773439, 1.0365676879882812]
'''
intersect(means1, means2, range1, range2, 0.67, map_probs)

'''



##Same_cloth_all_	
means1 = [31.594603474934896, 26.55020497639974, 24.33733367919922, 28.57050018310547, 24.512224578857424, 35.803386840820316, 26.801624043782553, 36.40996551513672, 37.008740234375, 48.244301350911456, 49.76127370198568, 20.024667612711593]
#####
range1 = [4.016165161132813, 1.2052154541015625, 0.700540924072266, 1.9542800903320305, 1.8369369506835938, 2.046347045898436, 4.277120208740234, 1.9503295898437507, 7.261322021484375, 1.882130432128907, 0.8288909912109368, 1.3831985473632802]

##different_cloth_front_
means2 =   [35.228719075520836, 29.979376729329427, 24.213468933105464, 33.00527445475261, 25.484789784749353, 34.15295562744141, 29.99436798095703, 34.73474934895833, 31.42778930664063, 54.072411092122394, 57.4769276936849, 19.95940373738607]
##------
range2 =  [2.11322021484375, 0.9510543823242195, 0.5031196594238274, 3.124990844726561, 1.9442024230957031, 0.710560607910157, 1.0747726440429677, 0.9111038208007791, 1.969349670410157, 1.94390869140625, 2.2960723876953146, 1.4543411254882823]


##Person 2 Same_cloth_frontback:  ##negative test
means3 = [31.67782470703125, 27.130612640380853, 21.9646053314209, 27.427981058756515, 23.558985392252605, 37.99293721516927, 51.224253082275396, 37.52398223876953, 42.83992665608724, 48.722646484375005, 49.61410217285156, 19.266798570421006]
##------
range3 = [1.7366119384765621, 2.7200584411621094, 1.3248321533203118, 2.1945167541503903, 1.3218681335449212, 2.724659729003907, 12.862861633300781, 1.8451629638671854, 9.037312316894528, 1.3030532836914084, 1.5007202148437493, 0.8748825073242195]


#Same_cloth_front_
means4 =  [33.5146484375, 27.64031982421875, 23.979890441894533, 27.931968688964844, 25.236145782470704, 34.63337860107422, 21.694468688964843, 35.04783477783204, 30.425020853678387, 48.520300292968756, 48.95369110107422, 20.304075622558592]
##------
range4 =   [0.3203338623046861, 0.6231689453125, 0.15695495605468857, 0.43279571533203054, 0.014476776123046875, 0.26034698486328267, 0.0, 0.571125793457032, 4.029199981689452, 0.3450500488281243, 0.26798858642577983, 0.037844848632813566]


#Same_cloth_back_
means5 =  [32.36956075032552, 32.0501708984375, 24.641452026367187, 28.217807006835937, 21.917877705891925, 35.62004699707031, 30.701316833496094, 37.450042724609375, 32.25698394775391, 48.129249064127606, 47.026023356119794, 21.775068664550776]
##------
range5 =   [1.3723754882812518, 1.2871154785156236, 0, 0.5234054565429691, 1.6067039489746087, 1.318228149414061, 0.03202972412109517, 1.5013763427734403, 0.5375717163085945, 2.62660369873047, 1.7905136108398452, 0.34953536987304545]


##person2 same clothes front
means6 =   [32.81448567708333, 24.493742370605464, 20.194433593750002, 23.901813252766928, 21.339313761393228, 38.7869618733724, 57.29260406494141, 37.73876647949219, 32.16432495117188, 49.73700866699219, 49.08387044270833, 20.60067596435547]
##------
range6 = [1.5851699829101573, 0.2223236083984368, 1.081776428222657, 0.34437637329101456, 0.4781982421874993, 0.13546142578124787, 21.45186004638672, 1.9110488891601562, 0.0, 0.8372222900390653, 0.9484939575195312, 0.51995849609375]



intersect(means1, means2, range1, range2, 0.5, map_probs)
intersect(means1, means4, range1, range4, 0.5, map_probs)
intersect(means1, means5, range1, range5, 0.5, map_probs)
intersect(means3, means6, range3, range6, 0.5, map_probs)
print("false positives start....")
#intersect(means1, means3, range1, range3, 0.5, map_probs)
#intersect(means2, means3, range2, range3, 0.5, map_probs)
#intersect(means3, means4, range3, range4, 0.5, map_probs)
#intersect(means3, means5, range3, range5, 0.5, map_probs)

intersect(means1, means6, range1, range6, 0.5, map_probs)
intersect(means2, means6, range2, range6, 0.5, map_probs)
intersect(means6, means4, range6, range4, 0.5, map_probs)
intersect(means6, means5, range6, range5, 0.5, map_probs)
'''
