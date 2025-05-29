###Author: Essbee Vanhoutte
###WORK IN PROGRESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###This demonstrates bruteforcing roots for quadratics in an attempt to reveal the factors
###It is high inefficient... but I know how to improve this. Just uploading my work in progress.


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
workers=8         #max amount of parallel processes to use
#g_limit_mult=1   #size of the modulus

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
        if i[0] not in fe:
            fe.append(i[0])
            if i[1]==-1:
                y=i[0]
                while 1:
                    y+=i[2]
                    if y<(mod//start):
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

def launch(lists,xlist,n):

    manager=multiprocessing.Manager()
    return_dict=manager.dict()
    jobs=[]
    procnum=0
  
    z=0
    print("[*]Launching attack")
    while z < workers:
        p=multiprocessing.Process(target=find_comb, args=(lists[z],xlist[z],n,procnum,return_dict))
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
    while y<p:
            if squareRootExists(n,p,y):
                rlist.append(y)
                xlist[-1].extend(find_solution_x(n,p,y))
            y+=1
  #  print("xlist: ",xlist)          
    return rlist,xlist

def find_solution_x(n,mod,y):
    ##to do: can use tonelli if this ends up taking too long
    rlist=[]
    x=0
    while x<mod:
        test,test2=equation(y,x,n,mod)
        if test == 0:
            rlist.append(x)     
        x+=1
    return [rlist]

def normalize_sols(n,sum1):  
    sum1,total=create_partial_results(sum1)
    return sum1,total    

def build_sols_list(prime1,n,test1,mod1,limit):
    found1=0
    mult1=[]
    mult1=[]
    mult1.append(prime1)
    xl=[]
    if prime1==2:
        mult1=[2,[1]]
    else:   
        ylist,xlist=find_sol_for_p(n,mult1[0])
        mult1.append(ylist)
        xl.extend(xlist)
    lift=2
    liftlim=1
    if prime1==2:
        liftlim=2
    elif prime1==3:
        liftlim=1
    elif prime1 < 1:
        liftlim=1
    if prime1 < 3:
        while 1:
            oldmult1=copy.deepcopy(mult1)
            mult1=find_all_solp(n,prime1,lift)
            if(len(mult1[1])-len(oldmult1[1])>prime1-1):
                if lift > liftlim:
                    mult1=oldmult1
                    break
            if lift > liftlim:
                mult1=oldmult1
                break       
            lift+=1 
    test1.append(mult1[0])
    test1.append(mult1[1])
    mod1*=mult1[0]
    if mod1>limit:
        found1=1
    return test1,found1,mod1,xl

def create_partial_results_x(sols):
    new=[]
    i=0
    while i < len(sols):
        j=0
        new.append(sols[i])
        new.append([])
        while j < len(sols[i+1]):
            z=0
            new[-1].append([])
            while z < len(sols[i+1][j]):
                k=0
                temp=sols[i+1][j][z]
                tot=sols[i]
                while k < len(sols):
                    if sols[k] != sols[i]:
                        inv=inverse(sols[k],sols[i])
                        temp=temp*inv*sols[k]
                        tot*=sols[k]
                    k+=2
                new[-1][-1].append(temp%tot)
                z+=1    
            j+=1
        i+=2    
    return new,tot 

def calc_x_combos(xlist,total):
    xcombos=[]
    enum=[]
    z=0
    while z < len(xlist):
        q=0
        enum.append([])
        while q < len(xlist[z+1]):
            w=0
                    
            while w < len(xlist[z+1][q]):
                enum[-1].append(xlist[z+1][q][w])
                w+=1
            q+=1
        z+=2

    for r in itertools.product(*enum):
        s=0
        for l in r:
            s+=l
        xcombos.append(s%total)


    return xcombos

def find_comb(lists,xlist,n,procnum,return_dict):
    #print("lists: ",lists)
    #print("xlist: ",xlist)
    lists,total=create_partial_results(lists)
    xlist,total=create_partial_results_x(xlist)
    #print("lists: ",lists)
    #print("xlist: ",xlist)
    results=[]
    xcombos=calc_x_combos(xlist,total)
   # print("xcombos: ",xcombos)
    i=0
    lenum=[]
    while i < len(lists):
        lenum.append(lists[i+1])
        i+=2

    for a in itertools.product(*[range(len(x)) for x in lenum]):
        i=0
        co=0
        rootcount=1
        xenum=[]
        while i < len(a):
            index=a[i]
            co+=lists[i*2+1][index]
            xenum.append(xlist[i*2+1][index])
            i+=1
        co=co%total
        xcombo_results=[]
        seen=[]
        for r in itertools.product(*xenum):
            root=0
            for l in r:
                root+=l   
                del l
            root=root%total
            if root not in seen:
                seen.append(root)
            else:
                continue
            target=(root**2-root*co)%n
            if len(xcombo_results)==0:
                ci=0
                while ci < len(xcombos):
                    xcombo_results.append((xcombos[ci]**2-xcombos[ci]*co)%n)
    

                    ci+=1

            xi=0
            while xi < len(xcombo_results):
                t=xcombo_results[xi]
                if t == target and xcombos[xi] != root and root+xcombos[xi] != co:
                    newroot=(xcombos[xi]+root)
                    try1=(newroot**2-newroot*co)%n
                    try2=(newroot**2+newroot*co)%n
                    res=gcd(try1,n)
                    if res != 1 and res != n:
                       # print("found factors of n "+str(n)+" : "+str(res)+" and "+str(n//res))
                        if res not in results:
                            results.append(res)
                            return_dict[procnum]=results
                    res=gcd(try2,n)
                    if res != 1 and res != n:
                      #  print("found factors of n "+str(n)+" : "+str(res)+" and "+str(n//res))
                        if res not in results:
                            results.append(res)
                            return_dict[procnum]=results
                
                xi+=1
            del r             
    return 

def init(n,primeslist1):    
    global workers
    lists=[]
    mods=[]
    xlist=[]
    limit=round(n**0.5)#n*g_limit_mult
    found=[]
    i=0
    while i < workers:
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
                lists[i],found[i],mods[i],xl=build_sols_list(prime1,n,lists[i],mods[i],limit)
                xlist[i].extend(xl)
                hit=1
            i+=1         
        if hit ==0:
            break      
    launch(lists,xlist,n)
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
  #  primeslist=[41,107]#[3,7,11,19]
  #  primeslist=[11,19]
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


