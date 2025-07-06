###Author: Essbee Vanhoutte
###WORK IN PROGRESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###Factorization_v3_minor_version_020
###notes: use with python3 QSv3.py -keysize 40

###To do: Bunch of places where I really need to use tonelli or optimize things.
###To do: We probably shouldn't waste time on p-adic lifting results that arn't above the ceiling yet (arn't garantueed to be a solution in higher exponents)
###To do: Shouldn't waste time on small moduli because a solutions went above the ceiling early
###To do: Fix lifting for powers of 2.. since that will be a big performance gainer.
###To do: Explore different strategies, maybe we should just check if there is a small common solution in the liftedn quadratic coefficient solutions rather then doing cartesian product.



import random
import sympy
import itertools
import sys
import argparse
import multiprocessing
import time
import copy
from timeit import default_timer
import math

key=0                 #Define a custom modulus to factor
keysize=30            #Generate a random modulus of specified bit length
workers=8    #max amount of parallel processes to use
g_limit=0.1
g_limit_mult=1 #lower bound for total modulus of partial resutls, is exponential multiplier..
sieve_interval=1000000
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

def gcd(a,b): # Euclid's algorithm
    if b == 0:
        return a
    elif a >= b:
        return gcd(b,a % b)
    else:
        return gcd(b,a)

def formal_deriv(y,x,z):
    result=(z*2*x)+(y)
    return result

def find_r(mod,total):
    mo,i=mod,0
    while (total%mod)==0:
        mod=mod*mo
        i+=1
    return i

def find_all_solp(n,start,limit):
    ##This code is shit, if lifting takes too long, blame this function.
    rlist=[]    
    if start == 2:
        rlist=[[0,1]]
    else:
        i=0
        while i<start:
            if squareRootExists(n,start,i):
                temp=find_solution_x(n,start,i)
                rlist.append(temp[0])
            i+=1
    newlist=[]
    mod=start**2
    g=0
    while g<limit-1:
        rlist2=[]
        for i in rlist:
            if i[1]== -1:
                rlist2.append([i[0],-1,i[2]])
                continue
            j=0
            while j<len(i)-1:
                j+=1
                x=i[j]
                y=i[0]
                while 1:
                    xo=x    
                    while 1:
                        test,test2=equation(y,x,n,mod)
                        if test == 0:
                            b=0
                            while b<len(rlist2):
                                if rlist2[b][0] == y and rlist2[b][1] != -1:
                                    rlist2[b].append(x)
                                    b=-1
                                    break
                                b+=1    
                            if b!=-1:       
                                rlist2.append([y,x])
                        x+=mod//start
                        if x>mod-1:
                            break
                    x=xo    
                    y+=mod//start  
                    if y>mod-1:
                        break
            b=0
            while b<len(rlist2):
                if rlist2[b][1] != -1:
                    x=rlist2[b][1]
                    y=rlist2[b][0]
                    re=formal_deriv(y,x,1)
                    r=find_r(start,re)
                    ceiling=(start*r)+1
                    ceiling=start**ceiling
                    if mod < ceiling:
                        b+=1
                        continue    
                    rlist2[b]=[]
                    rlist2[b].append(y)
                    rlist2[b].append(-1)
                    rlist2[b].append(ceiling)
                b+=1    
        rlist=rlist2.copy() 
        mod*=start
        g+=1
    fe=[]
    
    for i in rlist2:
        cmod=mod//start
        if i[0] not in fe:
            if i[0] != cmod-i[0]:
                fe.append(i[0])
            else:
                fe.append(i[0])
            if i[1]==-1:
                y=i[0]
                while 1:
                    y+=i[2]
                  
                    if y<cmod:
                        if y != cmod-y:
                            fe.append(y)
                        else:
                            fe.append(y)
                    else:
                        break   
    newlist.append(mod//start)
    fe.sort()

    newlist.append(fe)  
    return newlist
        
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

def create_partial_results_small(sols,y,mod):
    k=0
    temp=y
    tot=mod
    while k < len(sols):
        if sols[k] != mod:
            inv=inverse(sols[k],mod)
            temp=temp*inv*sols[k]
            tot*=sols[k]
        k+=2 
    return temp%tot  

def lift2(rt,prime,exp,co,n,z,z2):
    found=[]
    i=0
    all_ceil=0
    while i < len(rt):
        rem=-1
        offset=0
        while 1:
            root=rt[i][1]+offset
            if root > prime**exp:
                break
            rem,rem2a=equation2(co,root,n,prime**exp,z,z2)
            offset+=prime**(exp-1)
            if rem == 0:
                co2=formal_deriv(co,root,z)
                r=find_r(prime,co2)
                co2%=prime**exp
                ceiling=(prime*r)+1
                ceiling=prime**ceiling
                all_ceil=ceiling
                if ceiling < (prime**(exp))+1:
                    all_ceil=1
                rem,rem2b=equation(co2,root,n,prime**exp,z,z2)
                if rem == 0:
                    if [co2,root] not in found:
                      #  print(str([co2,root])+" ceiling: "+str(ceiling)+" mod: "+str(prime**exp)+" rem2a: "+str(rem2a%n)+" rem2b: "+str(rem2b%n))
                        found.append([co2,root])

        i+=1
    return found, all_ceil



def check_lift_p(prime,co,n,collected):
    z2=1
    k=0
    ret=[]
    while k < len(collected):
        debug=[]
        ret_p=[]
        z=collected[k]%prime
        r=find_solution_x_r(n,prime,co,z,z2)
        if len(r)==0:
            ##To do: do I need to do anything else here?
            #print("none found: "+str(prime)+" co: "+str(co))
            k+=1
            continue
        rt=[]
        co2=[]
        i=0
        while i < len(r):
            co2.append(formal_deriv(co,r[i],z)%prime)

            i+=1
        j=0
        while j < len(co2):
            i=0
            while i < len(r):
                rem,rem2=equation(co2[j],r[i],n,prime,z,z2)
                if rem == 0:
                    rt.append([co2[j],r[i]])
                i+=1    
            j+=1       
        xp=2
        all_res=[rt]
        all_z=[z]
        stop=0
        while stop==0 and xp<5:
            debug.append(prime**xp)
            debug.append([])
            new_res=[]
            new_z=[]
            c=0
            ret_p.append(prime**xp)
            ret_p.append([])
            while c < len(all_res):
                l=0
                while l < prime:
                    z_temp=all_z[c]+l*prime**(xp-1)
                    res,ceil=lift2(all_res[c],prime,xp,co,n,z_temp,z2)
                    if len(res) > 0:
                            new_res.append(res)
                            new_z.append(z_temp)
                            #else:
                            debug[-1].append(z_temp)
                            ret_p[-1].append([z_temp,ceil])
                            if ceil ==1:
                                stop=1
                    l+=1
                c+=1
            #if len(new_res)==0:
                #print("yay**********************************************************************************")
               # break
            debug[-1].sort()
            all_res=new_res      
            all_z=new_z  
            xp+=1        
        ##To do: remove this loop later... but the debug info about ceil may be useful for now
        ####ONE TEST WE NEED TO DO IS CHECK IF RESULTS THAT HIT THE CEIL ARE BETTER THEN THOSE THAT HAVNT YET
        h=0
        new_ret=[]
        new_ret.append(ret_p[-2])
        new_ret.append([])
        while h < len(ret_p[-1]):
            new_ret[-1].append(ret_p[-1][h][0])
            h+=1
        k+=1
        ret.append(new_ret)    
    return ret

def check_lift(co,lists,n,collected):
    all_ret=[]
    z=0
    while z< len(lists):
        ret=check_lift_p(lists[z],co,n,collected[z+1])
        all_ret.append(ret)
        z+=2
    return all_ret

def search_small_co(collected,lists,co,n,rstart,rstop,return_dict,procnum,mod,primefield_primes):
    sieve_len=50
    enum=[]
    i=0
    lifted=check_lift(co,lists,n,collected)
    lifted_fin=[]
    h=0
    while h < len(lifted):
        h2=0
        largest=[0,[]]
        while h2 < len(lifted[h]):
            #print("\n",lifted[h][h2])
            if lifted[h][h2][0]> largest[0]:
                largest=lifted[h][h2]
            h2+=1
        lifted_fin.extend(largest)
            
        h+=1
    lifted_fin,total=create_partial_results(lifted_fin)
    enum=[]
    h=0
    while h < len(lifted_fin):
        enum.append(lifted_fin[h+1])
        h+=2
    for a in itertools.product(*enum):
        tot=0
        for l in a:
            tot+=l
        tot%=total
        x=0
       # print(tot)
        while x < 10:
            tot2=tot+total*x
            square=co**2+n*4*tot2
               # print("tot: ",tot2) 
            test=isqrt(square)
            if test**2 == square:
                gcd_result=gcd(abs(co+test),n)
                if gcd_result != 1 and gcd_result != n:
                    print("Found at linear co: "+str(co)+" linear co2: "+str(test)+" quadratic co: "+str(tot2)+" modulus: "+str(total))

                    results=[]
                    results.append(gcd_result)
                    return_dict[procnum]=results
                    return 0
            x+=1      
               
       
    return 0

def launch(lists,n,primeslist1,primefield_primes):
    lists=lists[0]
    manager=multiprocessing.Manager()
    return_dict=manager.dict()
    jobs=[]
    procnum=0
    start= default_timer()
    print("[i]Creating iN datastructure... this can take a while...")



    hmap=create_hashmap(lists,n)
    duration = default_timer() - start
    print("[i]Creating iN datastructure in total took: "+str(duration))
    ####Move this into a worker functio

    z=0
    print("[*]Launching attack with "+str(workers)+" workers\n")

    part=(sieve_interval+1)//workers
    rstart=2#round(n**0.5)
    rstop=part#+round(n**0.5)
    if rstart == rstop:
        rstop+=1
    while z < workers:
        p=multiprocessing.Process(target=find_comb, args=(lists,n,procnum,return_dict,rstart,rstop,hmap,primefield_primes))
        rstart+=part  
        rstop+=part  
        jobs.append(p)
        p.start()
        procnum+=1
        z+=1            
    
    for proc in jobs:
        proc.join(timeout=0)        

    start=default_timer()

    while 1:
        time.sleep(1)
        z=0
        balive=0
        while z < len(jobs):
            if jobs[z].is_alive():
                balive=1
            z+=1
        check=return_dict.values()
        for item in check:
            if len(item)>0:
                factor1=item[0]
                factor2=n//item[0]
                if factor1*factor2 != n:
                    print("some error happened")
                print("\n[i]Factors of " +str(n)+" are: "+str(factor1)+" and "+str(factor2))
                for proc in jobs:
                    proc.terminate()
                return 0
        if balive == 0:
            print("[i]All procs exited")
            return 0    
    return 

def equation(y,x,n,mod,z,z2):
    rem=z*(x**2)+y*-x+n*z2
    rem2=rem%mod
    return rem2,rem  

def legendre(a, p):
    return pow_mod(a,(p-1)//2,p) 

def squareRootExists(n,p,b):
    b=b%p
    c=n%p
    bdiv = (b*inverse(2,p))%p
    alpha = (pow_mod(bdiv,2,p)-c)%p
    if alpha == 0:
        return 1
    
    if legendre(alpha,p)==1:
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

def find_sol_for_p(n,p):
    rlist=[]
    xlist=[p,[]]
    y=0
 
    while y<(p//2)+1:
            if squareRootExists(n,p,y):
                if (p-y)%p == y:
                    rlist.append([y])
                else:
                    rlist.append([y,p-y])
            y+=1
  #  print("xlist: ",xlist)          
    return rlist

def find_solution_x(n,mod,y,z):
    ##to do: can use tonelli if this ends up taking too long
    rlist=[]
    x=0
    while x<mod:
        test,test2=equation(y,x,n,mod,z)
        if test == 0:
            rlist.append([y,x])     
        x+=1
    return rlist


def find_solution_x2(n,mod,y):
    ##to do: can use tonelli if this ends up taking too long
    rlist=[]
    x=0
    while x<mod:
       # test,test2=equation(y,x,n,mod,z,z2)
        if x**2%mod == y%mod:
        #if test == 0:
            rlist.append(x)     
        x+=1  
    return rlist

def equation2(y,x,n,mod,z,z2):
    rem=z*(x**2)+y*x-n*z2
    rem2=rem%mod
    return rem2,rem  

def find_solution_x_r(n,mod,y,z,z2):
    ##to do: can use tonelli if this ends up taking too long
    rlist=[]
    x=0
    while x<mod:
        test,test2=equation2(y,x,n,mod,z,z2)
       # print("test: ",test)
        if test == 0:
            rlist.append(x)     
        x+=1
    return rlist        

def normalize_sols(n,sum1):  
    sum1,total=create_partial_results(sum1)
    return sum1,total    

def build_sols_list(prime1,n,test1,mod1,limit):
    found1=0
    mult1=[]
    mult1=[]
    mult1.append(prime1)
   # bmap_all=[]
    if prime1==2:
        mult1=[2,[[1]]]
    else:   
        ylist=find_sol_for_p(n,mult1[0])
        mult1.append(ylist)

    lift=2
    liftlim=1
    if prime1==2:
        liftlim=9
    elif prime1==3:
        liftlim=5
    elif prime1 < 20:
        liftlim=2
    if prime1 < 2:
        while 1:
            oldmult1=copy.deepcopy(mult1)
            mult1=find_all_solp(n,prime1,lift)
            if(len(mult1[1])-len(oldmult1[1])>prime1-1):
                if lift > liftlim:
                    mult1=oldmult1
                    fmult1=[]
                    pr=mult1[0]
                    fmult1.append(pr)
                    fmult1.append([])
                    z=0
                    while z < len(mult1[1]):
                        #To do: bug here I need to fix
                        if (pr-mult1[1][z]) in mult1[1]:
                            if mult1[1][z] < (pr//2)+1:
                                fmult1[1].append([mult1[1][z],pr-mult1[1][z]])
                        else:
                            fmult1[1].append([mult1[1][z]])
                        z+=1
                    mult1=fmult1
                    break
            if lift > liftlim:
                mult1=oldmult1
                fmult1=[]
                pr=mult1[0]
                fmult1.append(pr)
                fmult1.append([])
                z=0
                while z < len(mult1[1]):
                    if (pr-mult1[1][z]) in mult1[1]:
                        if mult1[1][z] < (pr//2)+1:
                            fmult1[1].append([mult1[1][z],pr-mult1[1][z]])
                    else:
                        fmult1[1].append([mult1[1][z]])
                    z+=1
                   # print("fmult1: ",fmult1)
                mult1=fmult1
                break       
            lift+=1 

    test1.append(mult1[0])
    test1.append(mult1[1])
    mod1*=mult1[0]
    if mod1>limit:
        found1=1
    return test1,found1,mod1

def solve_lin_con(a,b,m):
    ##ax=b mod m
    g=gcd(a,m)
    a,b,m = a//g,b//g,m//g
    return pow(a,-1,m)*b%m  

def solve_roots2(prime,co,n,hmap_p):
    iN=0
    while iN < prime:
        square=co**2+n*4*iN
        if jacobi(square,prime)!=-1:
            roots=find_solution_x2(n,prime,square)
            if len(roots) > 0:
                try:
                    c=hmap_p[str(co)]
                    c.append(iN)
                except Exception as e:
                    c=hmap_p[str(co)]=[iN]
        iN+=1

    return 

def solve_roots(prime,co_list,n):
    hmap_p={}
    iN=0
    while iN < prime:
        new_square=(co_list[0][0]**2-iN*n)%prime
        roots=find_solution_x2(n,prime,new_square)
        j=0
        if len(roots)>0:
            for root in roots:
                solve_roots2(prime,root,n,hmap_p)
        iN+=1     
    return hmap_p


def create_hashmap(lists,n):
   # print("lists: ",lists)
    i=0
    hmap=[]
    while i < len(lists):
        hmap_p=solve_roots(lists[i],lists[i+1],n)
        hmap.append(hmap_p)
        i+=2
    print(hmap)
    return hmap


def isqrt(n): # Newton's method, returns exact int for large squares
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2

    return x

def jacobi(a, n):
    #assert(n > a > 0 and n%2 == 1)
    t=1
    while a !=0:
        while a%2==0:
            a /=2
            r=n%8
            if r == 3 or r == 5:
                t = -t
                #return -1
        a, n = n, a
        if a % 4 == n % 4 == 3:
            t = -t
        #   return -1
        a %= n
    if n == 1:
        return t
    else:
        return 0    


def tonelli(n, p):  # tonelli-shanks to solve modular square root: x^2 = n (mod p)
    assert legendre(n, p) == 1, "not a square (mod p)"
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    if s == 1:
        r = pow(n, (p + 1) // 4, p)
        return r, p - r
    for z in range(2, p):
        if p - 1 == legendre(z, p):
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

def find_comb(lists,n,procnum,return_dict,rstart,rstop,hmap,primefield_primes):
    co=rstart
    co_max=rstop
    while co < co_max:
        collected=[]   
        i=0
        skip=0
        while i < len(hmap):
            try:
                collected.append(lists[i*2])
                collected.append(hmap[i][str(co%lists[i*2])])
            except Exception as e:
                skip=1
                break

            i+=1
        if skip ==0:
            collected,mod=create_partial_results(collected)
            res=search_small_co(collected,lists,co,n,rstart,rstop,return_dict,procnum,mod,primefield_primes)
 
        co+=2
    print("done")
    return 0
     
def init(n,primeslist1,primefield_primes):    
    global workers
    lists=[]
    mods=[]
    xlist=[]
    limit=round(n**g_limit)**g_limit_mult
    found=[]

    i=0
    while i < 1:
        lists.append([])
        xlist.append([])
        found.append(0)
        mods.append(1)
        i+=1

    while 1:
        i=0
        hit=0
        while i < len(lists):
            if found[i]==0:
                prime1=primeslist1[0]
                primeslist1.pop(0)
                lists[i],found[i],mods[i]=build_sols_list(prime1,n,lists[i],mods[i],limit)
                hit=1
            i+=1         
        if hit ==0:
            break 
    print("lists: ",lists)    
    launch(lists,n,primeslist1,primefield_primes)
    return 

def get_primes(start,stop):
    return list(sympy.sieve.primerange(start,stop))

def main():
    global key
    global base
    global workers
    start = default_timer() 
    if g_p !=0 and g_q !=0 and g_enable_custom_factors == 1:
        p=g_p
        q=g_q
        key=p*q
    if key == 0:
        print("\n[*]Generating rsa key with a modulus of +/- size "+str(keysize)+" bits")
        publicKey, privateKey,phi,p,q = generateKey(keysize//2)
        #p=41
        #q=107
        n=p*q
        key=n
    else:
        print("[*]Attempting to break modulus: "+str(key))
        n=key

    sys.set_int_max_str_digits(1000000)
    sys.setrecursionlimit(1000000)
    bits=bitlen(n)
    primeslist=[]
    print("[i]Modulus length: ",bitlen(n))
    print("[i]Gathering prime numbers..")
    primeslist.extend(get_primes(3,1000000))

    primefield_primes=[]
    counter=0
    nm=n*3
    while len(primefield_primes)< 20:
        if isPrime(nm+counter,10):
            primefield_primes.append(nm+counter)
        counter+=1
    print("primefield_primes: ",primefield_primes)
    
    init(n,primeslist,primefield_primes)
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
    global keysize,key,workers,debug,show,printcols
    parser = argparse.ArgumentParser(description='Factor stuff')
    parser.add_argument('-key',type=int,help='Provide a key instead of generating one') 
    parser.add_argument('-keysize',type=int,help='Generate a key of input size')    
    parser.add_argument('-workers',type=int,help='# of cpu cores to use')
    parser.add_argument('-debug',type=int,help='1 to enable more verbose output')
    parser.add_argument('-show',type=int,help='1 to render input matrix. 2 to render input+ouput matrix. -1 to render input matrix truncated by --printcols. -2 to render input+output matrix truncated by --printcols')
    parser.add_argument('--printcols',type=int,help='Truncate matrix output if enabled')

    args = parser.parse_args()
    if args.keysize != None:    
        keysize = args.keysize
    if args.key != None:    
        key=args.key
    if args.workers != None:  
        workers=args.workers
    if args.debug != None:
        debug=args.debug    
    if args.show != None:
        show=args.show
        if show < 0 and args.printcols  != None:
            printcols=args.printcols    
    return

if __name__ == "__main__":
    parse_args()
    print_banner()
    main()


