###Author: Essbee Vanhoutte
###WORK IN PROGRESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###Factorization_v3_minor_version_002
###notes: Only use with python3 QSv3.py -keysize 20  ... the parameters are set for 20 bit. 
###Next version will rework everything and get rid of a bunch of loop and iterations.


import random
import sympy
import itertools
import sys
import argparse
import multiprocessing
import time
import copy
from timeit import default_timer

key=0                 #Define a custom modulus to factor
keysize=12            #Generate a random modulus of specified bit length
workers=8      #max amount of parallel processes to use
g_limit_mult=4  #lower bound for total modulus of partial resutls, is exponential multiplier..
sieve_interval=5000
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

def formal_deriv(y,x):
    result=(2*x)+(y)
    return result

def find_r(mod,total):
    mo,i=mod,0
    while (total%mod)==0:
        mod=mod*mo
        i+=1
    return i

def find_all_solp(n,start,limit):
    ##This code is shit, if lifting takes too long, blame this function.
    #print("Checking: ",start)
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
        #print("rlist: ",rlist)
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
                 #   print("Y: "+str(y)+" mod: "+str(mod)) 
                    if y>mod-1:
                        break
            b=0
            while b<len(rlist2):
                if rlist2[b][1] != -1:
                    x=rlist2[b][1]
                    y=rlist2[b][0]
                    re=formal_deriv(y,x)
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
          #  print("i[0]",i[0])
            if i[1]==-1:
                y=i[0]
                while 1:
                    y+=i[2]
                  
                    if y<cmod:
                       # print("fe: ",fe)
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
    #new=[]
   # i=0
  #  while i < len(sols):
    #    j=0
    #    new.append(sols[i])
     #   new.append([])
     #   while j < len(sols[i+1]):
    k=0
    temp=y#sols[i+1][j]
    tot=mod
    while k < len(sols):
        if sols[k] != mod:
            inv=inverse(sols[k],mod)
            temp=temp*inv*sols[k]
            tot*=sols[k]
        k+=2
            #new[-1].append(temp%tot)    
           # j+=1
      #  i+=2    
    return temp%tot   


def collect(bmap,lists,xN):
    ###Note may need to return index list so I can index into xN counterparts
    i=0
    same=[]
    total=1
    while i < len(bmap):
        ind=xN%lists[i*2]
        if bmap[i][ind] != None:
            same.extend([lists[i*2],bmap[i][ind]])
            total*=lists[i*2]
        else:
            same.extend([None,None])
        i+=1

    return same,total

def collect_counterpart(same,hmap,lists,xN):
    ###Note may need to return index list so I can index into xN counterparts
    i=0
    ret=[]
    total=1
    while i < len(same):
       # print("same[i]: ",same[i])
       # print("hmap[i]":,hmap[i])
        if same[i] != None:
           # print("hmap[i]: ",hmap[i//2][xN%same[i]][same[i+1]])
            if len(hmap[i//2][xN%same[i]][same[i+1]])!=0:
                ret.extend([same[i],hmap[i//2][xN%same[i]][same[i+1]]])
                total*=same[i]

        i+=2

    return ret,total


def launch(lists,n,primeslist1,bmap):

    manager=multiprocessing.Manager()
    return_dict=manager.dict()
    jobs=[]
    procnum=0
    hmap,tmod=create_hashmap(lists[0],n)
    z=0
    print("[*]Launching attack")

    part=(sieve_interval+1)//workers
    rstart=round(n**0.5)
    rstop=part+round(n**0.5)
    if rstart == rstop:
        rstop+=1
    while z < workers:
        p=multiprocessing.Process(target=find_comb, args=(lists[0],n,procnum,return_dict,rstart,rstop,primeslist1,bmap,hmap))
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

def equation(y,x,n,mod):
    rem=(x**2)+y*-x+n
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
    bmap=[None]*p
    while y<(p//2)+1:
            if squareRootExists(n,p,y):
                if (p-y)%p == y:
                    rlist.append([y])
                    bmap[y]=len(rlist)-1
                else:
                    rlist.append([y,p-y])
                    bmap[y]=len(rlist)-1
                    bmap[p-y]=len(rlist)-1
                    #bmap[y]=1
                   # bmap[p-y]=1
            y+=1
  #  print("xlist: ",xlist)          
    return rlist,bmap

def find_solution_x(n,mod,y):
    ##to do: can use tonelli if this ends up taking too long
    rlist=[]
    x=0
    while x<mod:
        test,test2=equation(y,x,n,mod)
        if test == 0:
            rlist.append([y,x])     
        x+=1
    return rlist


def find_solution_x2(n,mod,y):
    ##to do: can use tonelli if this ends up taking too long
    rlist=[]
    x=0
    while x<mod:
        test,test2=equation(y,x,n,mod)
        if x**2%mod == y%mod:
        #if test == 0:
            rlist.append(x)     
        x+=1

   # if mod == 4 and y==0:
       # print(rlist)
       # time.sleep(100)    
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
        mult1=[2,[1]]
    else:   
        ylist,bmap=find_sol_for_p(n,mult1[0])
        mult1.append(ylist)

    lift=2
    liftlim=1
    if prime1==2:
        liftlim=8
    elif prime1==3:
        liftlim=4
    elif prime1 < 6:
        liftlim=2
    if prime1 < 3:
        print("fuk")
        while 1:
            oldmult1=copy.deepcopy(mult1)
            mult1=find_all_solp(n,prime1,lift)
            if(len(mult1[1])-len(oldmult1[1])>prime1-1):
                print("bleep")
                if lift > liftlim:
                    print("bloop")
                    mult1=oldmult1

                   # print("mult1: ",mult1)

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
    return test1,found1,mod1,bmap



def create_hashmap(lists,n):
    tmod=1
    hmap=[]
    i=0
    while i < len(lists):
        tmod*=lists[i]
        hmap.append([])
        j=0
        while j < lists[i]:
            hmap[-1].append([])
            k=0
            while k < len(lists[i+1]):
                new=[]
                xl=find_solution_x2(n,lists[i],(lists[i+1][k][0]**2-n*j))
                l=0
                while l < len(xl):

                    xl[l]=create_partial_results_small(lists,xl[l],lists[i])
                    l+=1
                new.extend(xl)
                hmap[-1][-1].append(new)      
                k+=1
            j+=1    
        i+=2
    return hmap ,tmod




def find_counterpart_square_check(clist,check):
    ###We need direct indexing here... 

    i=0
    total=1
    while i < len(clist):
        j=0
        while j < len(clist[i+1]):
            if check%clist[i] == clist[i+1][j]%clist[i]:
                total*=clist[i]
                break
            j+=1
        i+=2

    return total

def find_counterpart_square(clist,total2,xN,n,ubound):
    ###Change to not-brute-force later
    i=0
    while i < 1000 and i < ubound:
        total=find_counterpart_square_check(clist,i)
        test=total - xN*n
        if test < 0:
            i+=1
            continue
        if i**2 < test:
            return i,total,test
        i+=1
    return -1,0,0

def find_comb(lists,n,procnum,return_dict,rstart,rstop,primeslist1,bmap,hmap):
    ####Need to optimize this code
    results=[]
    a=rstart
    max_xN=5
    first_co_lbound=(n*max_xN)**2
    second_co_lbound=(n*max_xN)**2
    while a < rstop:
        same,total=collect(bmap,lists,a)
        if total < first_co_lbound: ###Only continue if the first coefficient has a very big modulus
            a+=1
            continue

        xN=1
        while xN < max_xN:
            clist,total2=collect_counterpart(same,hmap,lists,xN)
            if total2 < second_co_lbound:
                xN+=1
                continue
            ubound=round((total2-xN*n)**0.5)
            #print("found")
            res,newtotal,test=find_counterpart_square(clist,total2,xN,n,ubound)
            if res == -1 or a**2 > test:
                xN+=1
                continue


            print("["+str(procnum)+"] a: "+str(a)+" res: "+str(res)+" xN: "+str(xN)+" total: "+str(total)+" total2: "+str(total2)+" newtotal: "+str(newtotal))
            try1=a+res
            result=gcd(try1,n)
            if result != 1 and res != n:
                results.append(result)
                return_dict[procnum]=results
                return
            xN+=1
        a+=1
    print("["+str(procnum)+"] Worker exiting")
    return


def init(n,primeslist1):    
    global workers
    lists=[]
    mods=[]
    xlist=[]
    limit=n**g_limit_mult
    bmap_all=[]
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
                lists[i],found[i],mods[i],bmap=build_sols_list(prime1,n,lists[i],mods[i],limit)
                bmap_all.append(bmap)
                hit=1
            i+=1         
        if hit ==0:
            break 
    #print("bmap: ",bmap_all)
    print("lists: ",lists)     
    launch(lists,n,primeslist1,bmap_all)
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
        #p=307
        #q=397
      #  p=647
       # q=877

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
    primeslist.extend(get_primes(3,1000000))

    init(n,primeslist)
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


