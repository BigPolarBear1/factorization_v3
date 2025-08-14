###Author: Essbee Vanhoutte
###WORK IN PROGRESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###Factorization_v3 minor_version: 055

##To do: Insert sources as reference from which I have modified or copied code (Note to self: Important, do not forget!!!!!!)

###notes: use with python3 QSv3_055_2d_sieving_WIP.py -base 100 -keysize 100

##To do: Find most optimal parameters for randomized modulus selection
##To do: Dont just sieve with exact cmod match, we should calculate some threshold here... will see big gains from doing this.
##To do: Right now it has a function called construct_sieve_interval_1d .. but ideally, we want to fill out the interval in 2d dimension and minimize calculating linear congruences.

import random
import sympy
from itertools import chain
import itertools
import sys
import argparse
import multiprocessing
import time
import copy
from timeit import default_timer
import math
import gc

key=0                 #Define a custom modulus to factor
keysize=100            #Generate a random modulus of specified bit length
workers=8 #max amount of parallel processes to use
quad_co_per_worker=1 #Amount of quadratic coefficients to check. Keep as small as possible.
base=1000
lin_sieve_size=100000
quad_sieve_size=1000
g_debug=0 #0 = No debug, 1 = Debug, 2 = A lot of debug
g_lift_lim=0.5
thresvar=40  ##Log value base 2 for when to check smooths with trial factorization. Eventually when we fix all the bugs we should be able to furhter lower this.
matrix_mul=2 ##1.0 = square.. increase to overshoot min smooths, for this version, gathering double the amount seems best, since we don't concentrate all our smooths at a few quadratic coefficients as other versions do.
lp_multiplier=2
min_prime=40
g_enable_custom_factors=0
g_p=107
g_q=41

##Key gen function##
def power(x, y, p):
    res = 1;
    x = x % p;
    while (y > 0):
        if (y & 1):
            res = (res * x) % p;
        y = y>>1; # y = y/2
        x = (x * x) % p;
    return res;

def miillerTest(d, n):
    a = 2 + random.randint(1, n - 4);
    x = power(a, d, n);
    if (x == 1 or x == n - 1):
        return True;
    while (d != n - 1):
        x = (x * x) % n;
        d *= 2;
        if (x == 1):
            return False;
        if (x == n - 1):
            return True;
    # Return composite
    return False;

def isPrime( n, k):
    if (n <= 1 or n == 4):
        return False;
    if (n <= 3):
        return True;
    d = n - 1;
    while (d % 2 == 0):
        d //= 2;
    for i in range(k):
        if (miillerTest(d, n) == False):
            return False;
    return True;

def generateLargePrime(keysize = 1024):
    while True:
        num = random.randrange(2**(keysize-1), 2**(keysize))
        if isPrime(num,4):
            return num

def findModInverse(a, m):
    if gcd(a, m) != 1:
        return None
    u1, u2, u3 = 1, 0, a
    v1, v2, v3 = 0, 1, m
    while v3 != 0:
        q = u3 // v3
        v1, v2, v3, u1, u2, u3 = (u1 - q * v1), (u2 - q * v2), (u3 - q * v3), v1, v2, v3
    return u1 % m

def generateKey(keySize):
    while True:
        p = generateLargePrime(keySize)
        print("[i]Prime p: "+str(p))
        q=p
        while q==p:
            q = generateLargePrime(keySize)
        print("[i]Prime q: "+str(q))
        n = p * q
        print("[i]Modulus (p*q): "+str(n))
        count=65537
        e =count
        if gcd(e, (p - 1) * (q - 1)) == 1:
            break

    phi=(p - 1) * (q - 1)
    d = findModInverse(e, (p - 1) * (q - 1))
    publicKey = (n, e)
    privateKey = (n, d)
    print('[i]Public key - modulus: '+str(publicKey[0])+' public exponent: '+str(publicKey[1]))
    print('[i]Private key - modulus: '+str(privateKey[0])+' private exponent: '+str(privateKey[1]))
    return (publicKey, privateKey,phi,p,q)
##END KEY GEN##

def bitlen(int_type):
    length=0
    while(int_type):
        int_type>>=1
        length+=1
    return length   

def gcd(a,b): # Euclid's algorithm ##To do: Use a version without recursion?
    if b == 0:
        return a
    elif a >= b:
        return gcd(b,a % b)
    else:
        return gcd(b,a)

def formal_deriv(y,x,z):
    result=(z*2*x)-(y)
    return result

def find_r(mod,total):
    mo,i=mod,0
    while (total%mod)==0:
        mod=mod*mo
        i+=1
    return i
        
def create_partial_results(sols):
    new=[]
    i=0
    while i < len(sols):
        j=0
        new.append(sols[i])
        new.append([])
        while j < len(sols[i+1]):
            k=0
            temp=sols[i+1][j]
            tot=sols[i]
            while k < len(sols):
                if sols[k] != sols[i]:
                    inv=inverse(sols[k],sols[i])
                    temp=temp*inv*sols[k]
                    tot*=sols[k]
                k+=2
            new[-1].append(temp%tot)    
            j+=1
        i+=2    
    return new,tot    

def QS(n,factor_list,sm,xlist,flist):
    factor_list.insert(0,2) ##To do: remove when we fix lifting for powers of 2
    if len(sm) < base:
        print("[i]Not enough smooth numbers found")
        if len(sm)==0:
            return
    g_max_smooths=round(base*matrix_mul)
    if len(sm) > g_max_smooths: 
        print('[*]trimming smooth relations...')
        del sm[g_max_smooths:]
        del xlist[g_max_smooths:]
        del flist[g_max_smooths:]  
    print("[i]Building matrix and performing gaussian elimination over gf(2)")
    M2 = build_matrix(factor_list, sm, flist)
    null_space=solve_bits(M2)
    print("[i]Checking nullspace for factors")
    f1,f2=extract_factors(n, sm, xlist, null_space)
    if f1 != 0:
        print("[SUCCESS]Factors are: "+str(f1)+" and "+str(f2))
        return f1,f2   
    print("[FAILURE]No factors found")
    return 0

def extract_factors(N, relations, roots, null_space):
    n = len(relations)
    for vector in null_space:
        prod_left = 1
        prod_right = 1
        for idx in range(len(relations)):
            bit = vector & 1
            vector = vector >> 1
            if bit == 1:
                prod_left *= roots[idx]
                prod_right *= relations[idx]
            idx += 1
        sqrt_right = math.isqrt(prod_right)
        ###Debug shit, remove for final version
        sqr1=prod_left**2%N 
        sqr2=prod_right%N
        if sqrt_right**2 != prod_right:
            print("something fucked up")
        if sqr1 != sqr2:
            print("ERROR ERROR")
        ###End debug shit#########
        prod_left = prod_left % N
        sqrt_right = sqrt_right % N
        factor_candidate = gcd(N, abs(sqrt_right-prod_left))
        if factor_candidate not in (1, N):
            other_factor = N // factor_candidate
            return factor_candidate, other_factor

    return 0, 0

def solve_bits(matrix):
    n=round(base*matrix_mul)
    lsmap = {lsb: 1 << lsb for lsb in range(n)}
    m = len(matrix)
    marks = []
    cur = -1
    mark_mask = 0
    for row in matrix:
        if cur % 100 == 0:
            print("", end=f"{cur, m}\r")
        cur += 1
        lsb = (row & -row).bit_length() - 1
        if lsb == -1:
            continue
        marks.append(n - lsb - 1)
        mark_mask |= 1 << lsb
        for i in range(m):
            if matrix[i] & lsmap[lsb] and i != cur:
                matrix[i] ^= row
    marks.sort()
    # NULL SPACE EXTRACTION
    nulls = []
    free_cols = [col for col in range(n) if col not in marks]
    k = 0
    for col in free_cols:
        shift = n - col - 1
        val = 1 << shift
        fin = val
        for v in matrix:
            if v & val:
                fin |= v & mark_mask
        nulls.append(fin)
        k += 1
        if k == 100000: ### TO DO: ??
            break
    return nulls

def build_matrix(factor_base, smooth_nums, factors):
    fb_len = len(factor_base)
    fb_map = {val: i for i, val in enumerate(factor_base)}
    ind=1
    #factor_base.insert(0, -1)
    M2=[0]*(round(base*matrix_mul)+1)
    for i in range(len(smooth_nums)):
        for fac in factors[i]:
            idx = fb_map[fac]
            M2[idx] |= ind
        ind = ind + ind
    return M2

def launch(n,primeslist1):
    gc.enable
    manager=multiprocessing.Manager()
    return_dict=manager.dict()
    jobs=[]
    start= default_timer()
    print("[i]Creating iN datastructure... this can take a while...")
    primeslist1c=copy.deepcopy(primeslist1)
    plists=[]
    i=0
    while i < workers:
        plists.append([])
        i+=1
    i=0
    while i < len(primeslist1):
        plists[i%workers].append(primeslist1c[0])
        primeslist1c.pop(0)
        i+=1
    z=0
    while z < workers:
        p=multiprocessing.Process(target=create_hashmap, args=(n,z,return_dict,plists[z]))
        jobs.append(p)
        p.start()
        z+=1 
    for proc in jobs:
        proc.join(timeout=0)  
    complete_hmap=[]
    while 1:
        time.sleep(1)
        check123=return_dict.values()
        if len(check123)==workers:
            check123.sort()
            copy1=[]
            i=0
            while i < len(check123):
                copy1.append(check123[i][1])
                i+=1
            while 1:
                found=0
                m=0
                while m < len(copy1):
                    if len(copy1[m])>0:
                        complete_hmap.append(copy1[m][0])
                        copy1[m].pop(0)
                        found=1
                    m+=1
                if found ==0:
                    break
            break
    del check123
    del return_dict
    return_dict2=manager.dict()
    duration = default_timer() - start
    print("[i]Creating iN datastructure in total took: "+str(duration))
    z=0
    print("[*]Launching attack with "+str(workers)+" workers\n")

    part=quad_sieve_size
    rstart=1
    rstop=rstart+part
    if rstart == rstop:
        rstop+=1
    while z < workers:
        p=multiprocessing.Process(target=find_comb, args=(n,z,return_dict2,rstart,rstop,complete_hmap,primeslist1))
        rstart+=part  
        rstop+=part  
        jobs.append(p)
        p.start()
        z+=1            
    for proc in jobs:
        proc.join(timeout=0)        
    lastlen=0
    start=default_timer()
    fsm=[]
    fxlist=[]
    flist=[]
    seen=[]
    tick=0
    while 1:
        time.sleep(1)
        z=0
        balive=0
        while z < len(jobs):
            if jobs[z].is_alive():
                balive+=1
            z+=1
        check=return_dict2.values()
        tlen=0

        for item in check:
            a=0
            while a < len(item[1]):
                if item[0][a]%n not in seen:
                    seen.append(item[0][a]%n)
                    fsm.append(item[0][a])
                    fxlist.append(item[1][a])
                    flist.append(item[2][a])
                a+=1
        tlen=len(fsm)
        if tlen > lastlen:
            print("[i]Smooths found: "+str(tlen)+"/"+str(round(base*matrix_mul)))
            lastlen=tlen
        if balive == 0 or tlen > round(base*matrix_mul):
            for proc in jobs:
                proc.terminate()
            duration = default_timer() - start
            print("[i]Smooth finding took: "+str(duration)+" (seconds)")     
            QS(n,primeslist1,fsm,fxlist,flist)
            print("[i]All procs exited")
            return 0     
        tick+=1
        if tick == 30:
            print("[i]Alive workers: "+str(balive))
            tick=0
    return 


def equation(y,x,n,mod,z,z2):
    rem=z*(x**2)-y*x+n*z2
    rem2=rem%mod
    return rem2,rem 

def get_root(n,p,b,a):
    ##Just factor the quadratic I guess.
    ##Need tonelli shanks only if it is irreducable?
    a_inv=inverse(a,p)
    if a_inv == None:
        return []
    ba=(b*a_inv)%p 
    c=(-n)%p
    ca=(c*a_inv)%p
    bdiv = (ba*inverse(2,p))%p
    return [(-bdiv)%p]


def squareRootExists(n,p,b,a):
    b=b%p
    c=n%p
    bdiv = (b*inverse(2,p))%p
    alpha = (pow_mod(bdiv,2,p)-c)%p
    if alpha == 0:
        return 1
    
    if jacobi(alpha,p)==1:
        return 1
    return 0

def inverse(a, m):
    if gcd(a, m) != 1:
        return None
    u1,u2,u3 = 1,0,a
    v1,v2,v3 = 0,1,m
    while v3 != 0:
        q = u3//v3
        v1,v2,v3,u1,u2,u3=(u1-q*v1),(u2-q*v2),(u3-q*v3),v1,v2,v3
    return u1%m

def pow_mod(base, exponent, modulus):
    return pow(base,exponent,modulus)  

def solve_lin_con(a,b,m):
    ##ax=b mod m
  #  g=gcd(a,m)
    #a,b,m = a//g,b//g,m//g


    ###Question: Can we speed this up by precomputing inverses?
    return pow(a,-1,m)*b%m  

def tonelli(n, p):  # tonelli-shanks to solve modular square root: x^2 = n (mod p)
    assert jacobi(n, p) == 1, "not a square (mod p)"
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    if s == 1:
        r = pow(n, (p + 1) // 4, p)
        return r, p - r
    for z in range(2, p):
        if -1 == jacobi(z, p):
            break
    c = pow(z, q, p)
    r = pow(n, (q + 1) // 2, p)
    t = pow(n, q, p)
    m = s
    t2 = 0
    while (t - 1) % p != 0:
        t2 = (t * t) % p
        for i in range(1, m):
            if (t2 - 1) % p == 0:
                break
            t2 = (t2 * t2) % p
        b = pow(c, 1 << (m - i - 1), p)
        r = (r * b) % p
        c = (b * b) % p
        t = (t * c) % p
        m = i

    return (r, p - r)


def solve_roots(prime,n):
    iN=0      
    ret_hmaps=[]
    while iN < prime:
        new_square=(iN*4*n)%prime
        test=jacobi(new_square,prime)
        if test ==1:
            roots=tonelli(new_square,prime)
            root=roots[0]
            s=solve_lin_con(4*n,root**2,prime)
           # if prime-s < quad_co_per_worker*workers+1:
            ret_hmaps=lift_a(prime,n,root,prime-s,ret_hmaps)
        if test == 0:
            if len(ret_hmaps)>0:
                temp_hmap=ret_hmaps[0]
            else:
                temp_hmap=[]
                ret_hmaps.append(temp_hmap)
            temp_hmap.append([0,0])
        iN+=1   
    return ret_hmaps

def create_hashmap(n,procnum,return_dict,primeslist):
    i=0
    hmap=[]
    while i < len(primeslist):
        hmap_p=solve_roots(primeslist[i],n)
        hmap.append(hmap_p)
        i+=1
    return_dict[procnum]=[procnum,hmap]
    return 

def jacobi(a, n):
    t=1
    while a !=0:
        while a%2==0:
            a /=2
            r=n%8
            if r == 3 or r == 5:
                t = -t
        a, n = n, a
        if a % 4 == n % 4 == 3:
            t = -t
        a %= n
    if n == 1:
        return t
    else:
        return 0    

def equation2(y,x,n,mod,z,z2):
    rem=z*(x**2)+y*x-n*z2
    rem2=rem%mod
    return rem2,rem

def lift(exp,sol,n,z,z2,prime):
    i=0
    offset=0
    ret=[]
    while 1:
        root=sol[1]+offset
        if root > prime**exp:
            break
        rem,rem2=equation(0,root,n,prime**exp,z,z2)
        if rem ==0:
            co2=(formal_deriv(0,root,z))%(prime**exp)
            rem,rem2=equation2(prime**exp-co2,root,n,prime**exp,z,z2) 
            if rem == 0:
                ret.append([prime**exp-co2,root])
        offset+=prime**(exp-1)
    return ret

def lift_a(prime,n,root,z,ret_hmaps):
    if len(ret_hmaps)>0:
        temp_hmap=ret_hmaps[0]
    else:
        temp_hmap=[]
        ret_hmaps.append(temp_hmap)
    if root > prime // 2:
        root=(prime-root)%prime
    temp_hmap.append([z,root])
    return ret_hmaps   ##Code below is for lifting.. which we need to fix first.

    max_lim=round((4*n*quad_co_per_worker*workers)**g_lift_lim)
    z2=1
    k=0
    ret=[]
    new=[]
    r=get_root(n,prime,root,z) 
    if len(r)==0:
        return []
    rt=[]
    i=0
    while i < len(r):
        rem,rem2=equation(0,r[i],n,prime,z,z2)
        if rem == 0:
            rt.append([z,[root,r[i]]])
        i+=1
    
    exp=2
    while exp < 2:
        iN=z
        if len(ret_hmaps)>exp-1:
            temp_hmap=ret_hmaps[exp-1]
        else:
            temp_hmap={}
            ret_hmaps.append(temp_hmap)
        all_ret=[]
        i=0
           
        while i < len(rt):  ##To do: Can skip the second one.
            iN=rt[i][0]
            while iN < prime**exp and iN < quad_co_per_worker*workers:
                ret=lift(exp,rt[i][1],n,iN,z2,prime)
                if len(ret)>0:
                    if ret[0][0]<max_lim  and ((prime**(exp))-ret[0][0])<max_lim:
                        ret.insert(0,iN)
                        all_ret.append(ret)
                        try:
                            test=temp_hmap[str(ret[0])]
                            test.extend([ret[1][0],(prime**(exp))-ret[1][0]])
                        except Exception as e:
                            temp_hmap[str(ret[0])]=[ret[1][0],(prime**(exp))-ret[1][0]]

                iN+=prime**(exp-1)
            i+=1
        rt=all_ret
        if len(rt)==0:
            break
        exp+=1
    return ret_hmaps

def factorise_fast(value, factor_base):
    factors = set()
    if value < 0:
        factors ^= {-1}
        value = -value
    while value % 2 == 0:
        factors ^= {2}
        value //= 2
    for factor in factor_base:
        while value % factor == 0:
            factors ^= {factor}
            value //= factor
    return factors, value

def construct_interval_1d(quad_co,lin_co,cmod,primeslist,hmap):
    ###TO DO IMPORTANT!!!!!!!!!!!!!!!!!!!!! We should precompute which primes happen at which quad_co
    interval=[0]*lin_sieve_size

    i=0
    while i < len(primeslist):
        if primeslist[i] < min_prime or cmod%primeslist[i]==0:
            i+=1
            continue

        j=0
        while j < len(hmap[i][0]):
            if hmap[i][0][j][0]==quad_co%primeslist[i]:
                co2=hmap[i][0][j][1]
                dist1=solve_lin_con(cmod,co2-lin_co,primeslist[i])
                co2=(primeslist[i]-co2)%primeslist[i]
                dist2=solve_lin_con(cmod,co2-lin_co,primeslist[i])
                if dist1 < len(interval):
                    x=dist1
                    while x <len(interval):
                        interval[x]+=round(math.log2(primeslist[i]))
                        x+=primeslist[i]

                if dist2 < len(interval):
                    x=dist2
                    while x <len(interval):
                        interval[x]+=round(math.log2(primeslist[i]))
                        x+=primeslist[i]

            j+=1
        i+=1
    return interval

def construct_interval(n,cmod,cfact,qli,primeslist,hmap,gathered_quad_interval,gathered_ql_interval,rstart):
    all_interval=[]
    quad_co_list=[]
    index_list=[]
    lin_co_list=[]
    i=0
    while i < len(gathered_quad_interval):
        if gathered_quad_interval[i]==cmod:
            quad=rstart+i 
            lin_list=gathered_ql_interval[i]
            lin_list,total=create_partial_results(lin_list)
            lin=0
            j=1
            while j < len(lin_list):
                lin+=lin_list[j][0]
                j+=2
            lin%=cmod
            #print("Linear co: "+str(lin)+" quad co: "+str(quad))
            if (lin**2+4*n*quad)%cmod !=0: ##To do: Delete for final version
                print("WTF ERROR")
                time.sleep(10000)
            quad_co_list.append(quad)
            index_list.append(i)
            lin_co_list.append(lin)
            interval=construct_interval_1d(quad,lin,cmod,primeslist,hmap)
            all_interval.append(interval)
        i+=1
    return all_interval,quad_co_list,lin_co_list,index_list

def process_interval(procnum,return_dict,interval,cmod,n,quad_co_list,lin_co_list,primeslist,ret_array,cfact,partials,large_prime_bound,quad_interval,index_list):
    i=0
    while i < len(interval):
        j=0
        threshold = int(math.log2(len(interval[i]) * math.sqrt(n*4*quad_co_list[i])) - thresvar)
        while j < len(interval[i]):
            if interval[i][j] > threshold:
                co=lin_co_list[i]+cmod*j
                poly_val=co**2+n*4*(quad_co_list[i])
                if poly_val % cmod != 0:##To do: delete for final version
                    print("blah bug")
                local_factors, value = factorise_fast(poly_val,quad_interval[index_list[i]])
                if value != 1:
                    if value < large_prime_bound:
                        if value in partials:
                            rel, lf, pv = partials[value]
                            co *= rel
                            local_factors ^= lf
                            poly_val *= pv
                        else:
                            partials[value] = (co, local_factors, poly_val)
                            j+=1
                            continue
                    else:
                        j+=1
                        continue
                ret_array[0].append(poly_val)
                ret_array[1].append(co)
                ret_array[2].append(local_factors)
                return_dict[procnum]=ret_array 
            j+=1
        i+=1
    return

def generate_modulus(procnum,return_dict,n,primeslist,seen,max_bits,tnum,max_bit_inc,start_bits):
    ###PARAM###############
    close_range = 5
    too_close = 10
    LOWER_BOUND_SIQS=1000
    UPPER_BOUND_SIQS=4000
    total_poly=0
    log_maps=[]

    small_B = len(primeslist)-1
    lower_polypool_index = 2
    upper_polypool_index = small_B - 1

    min_ratio = LOWER_BOUND_SIQS
    poly_low_found = False
    ###Leave this here so we can mess with those bounds to see what works better
    for i in range(small_B):
        if primeslist[i] > LOWER_BOUND_SIQS and not poly_low_found:
            lower_polypool_index = i
            poly_low_found = True
        if primeslist[i] > UPPER_BOUND_SIQS:
            upper_polypool_index = i - 1
            break
    while max_bits < start_bits+max_bit_inc:
        cmod = 1
        cfact = []
        qli = []
        while True:
            found_a_factor = False
            while(found_a_factor == False):
                randindex = random.randint(lower_polypool_index, upper_polypool_index)
                potential_a_factor = primeslist[randindex]
                found_a_factor = True
                if potential_a_factor in cfact:
                    found_a_factor = False

            cmod = cmod * potential_a_factor
            cfact.append(potential_a_factor)
            qli.append(randindex)
            j = tnum.bit_length() - cmod.bit_length()
            if j < too_close:
                cmod = 1
                s = 0
                cfact = []
                qli = []
                continue
            elif j < (too_close + close_range):
                break
        a1 = tnum // cmod
        if a1 < min_ratio: ##Ergh, do we need this?
            continue

        mindiff = 100000000000000000
        randindex = 0

        for i in range(small_B):
            if abs(a1 - primeslist[i]) < mindiff:
                mindiff = abs(a1 - primeslist[i])
                randindex = i

        found_a_factor = False
        while not found_a_factor:
            potential_a_factor = primeslist[randindex]
            found_a_factor = True
            if potential_a_factor in cfact:
                found_a_factor = False
            if not found_a_factor:
                randindex += 1

        if randindex > small_B:
            continue

        cmod = cmod * primeslist[randindex]
        cfact.append(primeslist[randindex])
        qli.append(randindex)

        diff_bits = (tnum - cmod).bit_length()
        if diff_bits < max_bits:
            if cmod in seen:
                max_bits += 1
                if g_debug > 0:
                    print("["+str(procnum)+"][DEBUG1]Increasing max_bits to"+str(max_bits)+" at iN "+str(totali))
                continue
            else:
                seen.append(cmod)
                if g_debug > 1:
                    print("["+str(procnum)+"][DEBUG2]Seen: ",seen)
                break


    return cmod,cfact,qli,max_bits

def construct_quad_interval(hmap,primeslist1,rstart,rstop):
    quad_interval_size=rstop
    quad_interval=[]
    i=rstart
    while i < rstop:
        quad_interval.append([])
        i+=1
    i=0
    while i < len(hmap):
        j=0
        while j < len(hmap[i][0]):
            x=hmap[i][0][j][0]
            start=rstart%primeslist1[i]
            x-=start
            if x < 0:
                x+=primeslist1[i]
            while x < len(quad_interval):
                quad_interval[x].append(primeslist1[i])
                x+=primeslist1[i]

            j+=1
        i+=1
    return quad_interval

def process_quad_interval(quad_interval,cfact,hmap,primeslist1,rstart,rstop):
    #Gather all quadratic coefficient which we want to sieve and in addition have a seperate list for their modulus/factors
    gathered_quad_interval=[1]*len(quad_interval)
    gathered_ql_interval=[]
    i=0
    while i < len(quad_interval):
        gathered_ql_interval.append([])
        i+=1
  #  indexes=[]
    i=0
    while i < len(hmap):
        if primeslist1[i] in cfact:
            j=0
            while j < len(hmap[i][0]):
                if hmap[i][0][j][0] < len(quad_interval):
                    x=hmap[i][0][j][0]
                    start=rstart%primeslist1[i]
                    x-=start
                    if x < 0:
                        x+=primeslist1[i]
                    while x < len(quad_interval):
                        gathered_quad_interval[x]*=primeslist1[i]#.append(quad_interval[x])
                        gathered_ql_interval[x].append(primeslist1[i])
                        gathered_ql_interval[x].append([hmap[i][0][j][1]])
                      #  indexes.append(x)
                        x+=primeslist1[i]
                j+=1
        i+=1

    i=0
    return    gathered_quad_interval,gathered_ql_interval

def find_comb(n,procnum,return_dict,rstart,rstop,hmap,primeslist1):
    ret_array=[[],[],[]]
    seen=[]
    start=default_timer()
    quad_interval=construct_quad_interval(hmap,primeslist1,rstart,rstop)
    if g_debug > 0:
        duration = default_timer() - start
        print("["+str(procnum)+"]Constructing quad interval took: "+str(duration))
    partials={}
    gc.enable()
    test=[[],[],[]]
    sm=[]
    xl=[]
    fac=[]

    large_prime_bound = primeslist1[-1] ** lp_multiplier
    max_bits_inc=50
    tmul =0.9
    tnum = int(((n*4*rstop)**0.50) / (lin_sieve_size))

    start_bits=int(tnum.bit_length() * tmul)
    max_bits =start_bits
    while max_bits < start_bits+max_bits_inc:
        start=default_timer()
        cmod,cfact,qli,max_bits=generate_modulus(procnum,return_dict,n,primeslist1,seen,max_bits,tnum,max_bits_inc,start_bits)
        if g_debug > 0:
            duration = default_timer() - start
            print("["+str(procnum)+"]Generating modulus took: "+str(duration))
        start=default_timer()
        gathered_quad_interval,gathered_ql_interval=process_quad_interval(quad_interval,cfact,hmap,primeslist1,rstart,rstop)
        if g_debug > 0:
            duration = default_timer() - start
            print("["+str(procnum)+"]Processing quad interval took: "+str(duration))
        start=default_timer()
        interval,quad_co_list,lin_co_list,index_list=construct_interval(n,cmod,cfact,qli,primeslist1,hmap,gathered_quad_interval,gathered_ql_interval,rstart)
        if g_debug > 0:
            duration = default_timer() - start
            print("["+str(procnum)+"]Constructing 2d interval took: "+str(duration))
        start=default_timer()
        process_interval(procnum,return_dict,interval,cmod,n,quad_co_list,lin_co_list,primeslist1,ret_array,cfact,partials,large_prime_bound,quad_interval,index_list)
        if g_debug > 0:
            duration = default_timer() - start
            print("["+str(procnum)+"]Checking 2d interval took: "+str(duration))
    return 0

def get_primes(start,stop):
    return list(sympy.sieve.primerange(start,stop))

def main():
    global key
    start = default_timer() 
    if g_p !=0 and g_q !=0 and g_enable_custom_factors == 1:
        p=g_p
        q=g_q
        key=p*q
    if key == 0:
        print("\n[*]Generating rsa key with a modulus of +/- size "+str(keysize)+" bits")
        publicKey, privateKey,phi,p,q = generateKey(keysize//2)
       # p=41
      #  q=107
        n=p*q
        key=n
    else:
        print("[*]Attempting to break modulus: "+str(key))
        n=key

    sys.set_int_max_str_digits(1000000)
    sys.setrecursionlimit(1000000)
    bits=bitlen(n)
    primeslist=[]
    primeslist1=[]
    print("[i]Modulus length: ",bitlen(n))
    count = 0
    num=n
    while num !=0:
        num//=10
        count+=1
    print("[i]Number of digits: ",count)
    print("[i]Gathering prime numbers..")


    primeslist.extend(get_primes(3,1000000))

    i=0
    while i < base:
        primeslist1.append(primeslist[0])
        i+=1
        primeslist.pop(0)   
    launch(n,primeslist1)     
    duration = default_timer() - start
    print("\nFactorization in total took: "+str(duration))


def print_banner():
    print("Polar Bear was here       ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀                       ")
    print("⠀         ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⣀⣀⣀⣤⣤⠶⠾⠟⠛⠛⠛⠛⠷⢶⣤⣄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀   ")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣴⠶⠾⠛⠛⠛⠛⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⠛⢻⣿⣟ ⠀⠀⠀⠀      ")
    print("⠀⠀⠀⠀⠀⠀⠀⢀⣤⣤⣶⠶⠶⠛⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠳⣦⣄⠀⠀⠀⠀⠀   ")
    print("⠀⠀⠀⠀⠀⣠⡾⠟⠉⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠹⣿⡆⠀⠀⠀   ")
    print("⠀⠀⠀⣠⣾⠟⠀⠀⠀⠈⢉⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢿⡀⠀⠀   ")
    print("⢀⣠⡾⠋⠀⢾⣧⡀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣄⠈⣷⠀⠀   ")
    print("⢿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠀⢹⡆⣿⡆⠀   ")
    print("⠈⢿⣿⣛⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣆⣸⠇⣿⡇⠀   ")
    print("⠀⠀⠉⠉⠙⠛⠛⠓⠶⠶⠿⠿⠿⣯⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⠟⠀⣿⡇⠀   ")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠠⣦⢠⡄⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡞⠁⠀⠀⣿⡇⠀   ")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣶⠄⠀⠀⠀⠀⠀⠀⢸⣿⡇⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⠇⣼⠋⠀⠀⠀⠀⣿⡇⠀   ")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡿⣿⣦⠀⠀⠀⠀⠀⠀⠀⣿⣧⣤⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⣿⣾⠃⠀⠀⠀⠀⠀⣿⠛⠀   ")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠀⠘⢿⣦⣀⠀⠀⠀⠀⠀⠸⣇⠀⠉⢻⡄⠀⠀⠀⠀⠀⠀⡘⣿⢿⣄⣠⠀⠀⠀⠀⠸⣧⡀   ")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⠀⠀⠀⠙⣿⣿⡄⠀⠀⠀⠀⠹⣆⠀⠀⣿⡀⠀⠀⠀⠀⠀⣿⣿⠀⠙⢿⣇⠀⠀⠀⠀⠘⣷   ")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⡏⠀⠀⢀⣿⡿⠻⢿⣷⣦⠀⠀⠀⠹⠷⣤⣾⡇⠀⠀⠀⠀⣤⣸⡏⠀⠀⠈⢻⣿⠀⠀⠀⠘⢿   ")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠿⠁⠀⠀⢸⡿⠁⠀⠀⠙⢿⣧⠀⠀⠀⠀⠠⣿⠇⠀⠀⠀⠀⣸⣿⠁⠀⠀⢀⣾⠇⠀⠀⠀⠀⣼   ")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⡁⠀⠀⠀⠀⣸⡇⠀⠀⠀⠀⠈⠿⣷⣤⣴⡶⠛⡋⠀⠀⠀⠀⢀⣿⡟⠀⠀⣴⠟⠁⠀⣀⣀⣀⣠⡿   ")
    print("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣤⣾⣧⣤⡿⠁⠀⠀⠀⠀⠀⠀⠀⠈⣿⣀⣾⣁⣴⣏⣠⣴⠟⠉⠀⠀⠀⠻⠶⠛⠛⠛⠛⠋⠉⠀   ")
    return

def parse_args():

    ###To do: Fix this.... still lame relics from factorization_v1
    global keysize,key,workers,g_debug,base
    parser = argparse.ArgumentParser(description='Factor stuff')
    parser.add_argument('-key',type=int,help='Provide a key instead of generating one') 
    parser.add_argument('-keysize',type=int,help='Generate a key of input size')    
    parser.add_argument('-workers',type=int,help='# of cpu cores to use')
    parser.add_argument('-debug',type=int,help='1 to enable more verbose output')
    parser.add_argument('-base',type=int,help='Size of the factor base')
    args = parser.parse_args()
    if args.keysize != None:    
        keysize = args.keysize
    if args.key != None:    
        key=args.key
    if args.workers != None:  
        workers=args.workers
    if args.debug != None:
        g_debug=args.debug  
    if args.base != None:
        base=args.base  
  
    return

if __name__ == "__main__":
    parse_args()
    print_banner()
    main()


