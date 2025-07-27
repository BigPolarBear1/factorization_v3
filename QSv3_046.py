###Author: Essbee Vanhoutte
###WORK IN PROGRESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
###Factorization_v3 minor_version: 046
###QS functions modified from: <to do: insert github repo>
###notes: use with python3 QSv3_046.py -keysize 130


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
sieve_interval_per_worker=4 #Amount of quadratic coefficients to check. Keep small because PoC wont stop until all are checked or we hit g_max_smooths.
base=1000

g_debug=1 #0 = No debug, 1 = Debug, 2 = A lot of debug
g_max_local_hmap=100000000 ###Need to cap this to prevent OOM.. higher is better especially if you have lots of ram since it will allow you to pull more smooths.
g_lift_lim=0.5
g_co_lim=0.5
g_check_threshold=70  ##Log value base 2 for when to check smooths with trial factorization. Eventually when we fix all the bugs we should be able to furhter lower this.
g_max_smooths=1500 ##Probably want to keep this atleast double the factor base

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

def make_vector(n_factors,factor_base): 
    '''turns factorization into an exponent vector mod 2'''
    exp_vector = [0] * (len(factor_base))
    for j in range(len(factor_base)):
        if factor_base[j] in n_factors:
            exp_vector[j] = (exp_vector[j] + n_factors.count(factor_base[j])) % 2
    return exp_vector

def QS(n,factor_list,sm,xlist,flist):
    factor_list.insert(0,2) ##To do: remove when we fix lifting for powers of 2
    if len(sm) < base:
        print("[i]Not enough smooth numbers found")
        if len(sm)==0:
            return

    if len(sm) > g_max_smooths: 
        print('[*]trimming smooth relations...')
        del sm[g_max_smooths:]
        del xlist[g_max_smooths:]
        del flist[g_max_smooths:]  
      
    is_square, t_matrix = build_matrix(factor_list, sm, flist)
    print("[*]Starting Gaussian elimination")
    start=default_timer()
    sol_rows,marks,M = gauss_elim(t_matrix) 
    if sol_rows == 0:
        return 0
    duration = default_timer() - start
                        
    print("[i]Gauss_elim took: "+str(duration)+" (seconds)") 
    print("[*]Checking solutions")
    start=default_timer()
    K=0
    while K < len(sol_rows):
        solution_vec = solve_row(sol_rows,M,marks,K)
        factor = solve(solution_vec,sm,n,xlist,flist) 
        if factor != 1 and factor != n:
            print("[i]Found factors of: "+str(n))
            print("P: ",factor)
            print("Q: ",n//factor)
            duration = default_timer() - start
                        
            print("[i]Checking solutions took: "+str(duration)+" (seconds)") 
            return factor, n/factor   

        K+=1   
    return 0

def solve_row(sol_rows,M,marks,K=0):
    solution_vec, indices = [],[]
    free_row = sol_rows[K][0] # may be multiple K
    for i in range(len(free_row)):
        if free_row[i] == 1: 
            indices.append(i)
    
    for r in range(len(M)): #rows with 1 in the same column will be dependent
        for i in indices:
            if M[r][i] == 1 and marks[r]:
                solution_vec.append(r)
                break
               
    solution_vec.append(sol_rows[K][1]) 
    return(solution_vec)

def solve(solution_vec,smooth_nums,N,xlist,factor_list):
    solution_nums = [smooth_nums[i] for i in solution_vec]
    x_nums = [xlist[i] for i in solution_vec]
    fac = [factor_list[i] for i in solution_vec]
    b=1
    for n in x_nums:
        b *= n
    allfac=[]
    for fa in fac:
        allfac.extend(fa)
    allfac.sort()
    c=len(allfac)-1
    while c > -1:
        allfac.pop(c)
        c-=2
    af=1
    for fac in allfac:
        if fac != -1:
            af*=fac    
    a=af    
    print(str(a)+"^2 = "+str(b)+"^2 mod "+str(N))
    if a**2%N != b**2%N:
        print("something went wrong")
        time.sleep(100000000)
    if b > a:
        temp=a
        a=b
        b=temp
    factor = gcd(a-b,N)
    print("factor: ",factor)
    return factor  

def gauss_elim(M):
    marks = [False]*len(M[0])
    
    for i in range(len(M)): #do for all rows
        row = M[i]
        for num in row: #search for pivot
            if num == 1:
                j = row.index(num) # column index
                marks[j] = True
                
                for k in chain(range(0,i),range(i+1,len(M))): #search for other 1s in the same column
                    if M[k][j] == 1:
                        for i in range(len(M[k])):
                            M[k][i] = (M[k][i] + row[i])%2
                break
    M = transpose(M)
    sol_rows = []
    for i in range(len(marks)): #find free columns (which have now become rows)
        if marks[i]== False:
            free_row = [M[i],i]
            sol_rows.append(free_row)
    
    if not sol_rows:
        return 0,0,0#("No solution found. Need more smooth numbers.")
    return sol_rows,marks,M

def transpose(matrix):
#transpose matrix so columns become rows, makes list comp easier to work with
    new_matrix = []
    for i in range(len(matrix[0])):
        new_row = []
        for row in matrix:
            new_row.append(row[i])
        new_matrix.append(new_row)
    return(new_matrix)    

def build_matrix(factor_base, smooth_nums, factors):
    M = []
    factor_base.insert(0, -1)
        
    for i in range(len(smooth_nums)):
        
        exp_vector = make_vector(factors[i],factor_base)
        M.append(exp_vector)

    M = transpose(M)
    return (False, M)

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
        time.sleep(3)
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
                    #print("check123[m]: ",copy1[m])
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
    gc.collect()
    return_dict2=manager.dict()
    duration = default_timer() - start
    print("[i]Creating iN datastructure in total took: "+str(duration))
    z=0
    print("[*]Launching attack with "+str(workers)+" workers\n")

    part=sieve_interval_per_worker
    rstart=1
    rstop=part
    if rstart == rstop:
        rstop+=1
    while z < workers:
        p=multiprocessing.Process(target=find_comb, args=(n,z,return_dict2,rstart,rstop,complete_hmap,primeslist1))
        rstart+=part  
        rstop+=part  
        jobs.append(p)
        p.start()
        z+=1            
    gc.collect()
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
            while a < len(item[0]):
                if item[0][a]**2%n not in seen:
                    seen.append(item[0][a]**2%n)
                    fsm.append(item[0][a])
                    fxlist.append(item[1][a])
                    flist.append(item[2][a])
                a+=1
        tlen=len(fsm)
        if tlen > lastlen:
            print("[i]Smooths found: "+str(tlen))
            lastlen=tlen
        if balive == 0 or tlen > g_max_smooths:
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
    hmaps=[]
    hmap_p={}
    iN=0      
    ret_hmaps=[]
    while iN < prime:
        new_square=(iN*4*n)%prime
        test=jacobi(new_square,prime)
        if test ==1:
            roots=tonelli(new_square,prime)
            root=roots[0]
            s=solve_lin_con(4*n,root**2,prime)
            ret_hmaps=lift_a(prime,n,root,prime-s,ret_hmaps)
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
       # if g_debug > 1:
           # print("[Debug 2]Prime**exp: "+str(prime**exp)+" root: "+str(root)+" rem2: "+str(rem2)+" rem: "+str(rem))
        if rem ==0:
            co2=(formal_deriv(0,root,z))%(prime**exp)
            rem,rem2=equation2(prime**exp-co2,root,n,prime**exp,z,z2) 
            if rem == 0:
                ret.append([prime**exp-co2,root])
        offset+=prime**(exp-1)
    return ret

def lift_a(prime,n,root,z,ret_hmaps):
    max_lim=round((4*n*sieve_interval_per_worker*workers)**g_lift_lim)
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
    if len(ret_hmaps)>0:
        temp_hmap=ret_hmaps[0]
    else:
        temp_hmap={}
        ret_hmaps.append(temp_hmap)
    try:
        test=temp_hmap[str(z)]
        test.extend([root,(prime-root)])
    except Exception as e:
        temp_hmap[str(z)]=[root,(prime-root)]

    while exp < 20:
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
            while iN < prime**exp and iN < sieve_interval_per_worker*workers:
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

def lift_collected(collected,n,z,sqlimit):
    all_ret=[]
    i=0
    while i< len(collected):
        ret=lift_collected_p(collected[i],n,collected[i+1],z,sqlimit)
        if len(ret)==0:
            ret=[[collected[i],1],collected[i+1]]
        all_ret.append(ret)
        i+=2
    return all_ret

def test_smooth_can(smooth_can,primeslist1):   
    faclist=[]
    if smooth_can < 0:
        faclist.append(-1)
    while smooth_can % 2 ==0: ###To do: we need to fix the lifting code for powers of 2 eventually and remove this....
        smooth_can//=2
        faclist.append(2)
    v=0
    while v < len(primeslist1):
        while smooth_can % primeslist1[v] ==0:
            smooth_can//=primeslist1[v]
            faclist.append(primeslist1[v])
        v+=1
    if smooth_can == -1 or smooth_can == 1:
        return faclist
    return 0

def check_4_smooths(hmap,n,x,primeslist1,return_dict,ret_array,procnum):
    ##To do: need to use hmap information to determine which ones are even worth checking
    smooths=[]
    xlist=[]
    faclist=[]
    lim=x*n
    for i,(k,v) in enumerate(hmap.items()):
        for i2,(k2,v2) in enumerate(v.items()):
            co=int(k2)
            fasd=v2[0]
            v2.sort()
            j=len(v2)-1
            mod=1
            prev=1
            temp_faclist=[]
            while j > -1:
                if v2[j][0]!=prev:
                    z=0
                    while z < v2[j][1]:
                        mod*=v2[j][0]
                        temp_faclist.append(v2[j][0])
                        z+=1
                prev=v2[j][0]
                j-=1
            sm=co**2+4*n*x
            sm=sm//mod
            if sm < 1:
                print("hit an error")
                continue
            if math.log2(sm)<g_check_threshold:
                f=test_smooth_can(sm,primeslist1)

                if f !=0:
                    f.extend(temp_faclist)
                    ret_array[0].append(co**2+4*n*x)
                    ret_array[1].append(co)
                    ret_array[2].append(f)
                    return_dict[procnum]=ret_array 
            del i2,k2,v2
        del i,k,v


    return  

def calc_distances(new,n,co,co_mod,curr,local_hmap,limit):
    ##Known bugs: If res = 0 but the modulus is 5. Then we need ot add 5 as a divider? I.e for iN =2, co = 133 and res =5
    ##Known bugs: exponents of co_mod need to be added aswell

    k=0
    while k < len(new):
        if k == curr:
            k+=1
            continue
        i=0
        while i < len(new[k]):
            j=0
            while j < len(new[k][i+1]):
                res=solve_lin_con(co_mod[0]**co_mod[1],new[k][i+1][j]-co,new[k][i][0]**new[k][i][1])
                if co+(co_mod[0]**co_mod[1])*res > limit:
                    j+=1
                    continue
                try:
                    test=local_hmap[str(res)]
                    try:
                        test2=test[str(co+(co_mod[0]**co_mod[1])*res)]
                        if co_mod not in test2: ##Note: maybe just maintain a sorted list
                            test2.append(copy.deepcopy(co_mod))
                        if new[k][i] not in test2: ##Note: maybe just maintain a sorted list
                            test2.append(copy.deepcopy(new[k][i]))

                    except Exception as e:
                        #print(e)
                        test[str(co+(co_mod[0]**co_mod[1])*res)]=[copy.deepcopy(co_mod),copy.deepcopy(new[k][i])]
                except Exception as e:
                    #print(e)
                    new_hmap={}
                    new_hmap[str(co+(co_mod[0]**co_mod[1])*res)]=[copy.deepcopy(co_mod),copy.deepcopy(new[k][i])]
                    local_hmap[str(res)]=new_hmap

                j+=1
            i+=2
        k+=1
    return 

def find_smooth(procnum,n,x,primeslist1,collected,return_dict,ret_array):
    limit=round((4*x*n)**g_co_lim)
    all_smooths=[]
    all_xlist=[]
    all_flist=[]
    if g_debug > 0:
        start=default_timer()
    if g_debug > 1:
        print("[Debug 2]Collected: "+str(collected)+" at iN: "+str(x))
    hmap={}
    mc=1
    hmap={}
    done=0
    new=collected
    k=len(new)-1
    while k >-1:
        local_hmap={}
        i=len(new[k])-2
        while i>-1:
            j=len(new[k][i+1])-1
            while j >-1:
                calc_distances(new,n,new[k][i+1][j],new[k][i],k,local_hmap,limit)
                j-=1
            i-=2
        check_4_smooths(local_hmap,n,x,primeslist1,return_dict,ret_array,procnum)
        del local_hmap
        gc.collect()
        k-=1
    if g_debug > 0:
        duration = default_timer() - start
        print("[Debug 1]["+str(procnum)+"]calc_distances took: "+str(duration))
    return


def find_comb(n,procnum,return_dict,rstart,rstop,hmap,primeslist1):
    gc.enable()
    test=[[],[],[]]
    sm=[]
    xl=[]
    fac=[]
    totali=rstart
    totali_max=rstop
    if g_debug > 1:
        print("[Debug 2]iN map: ",hmap)
    while totali < totali_max:
        collected=[]   
        i=0
        skip=0
        limit=(totali*4)*n
        sqlimit=math.ceil(limit**0.5)
        total_mod=1
        while i < len(hmap):
            exp=1
            temp=[]
            j=0
            while j < len(hmap[i]):
                mod=primeslist1[i]**exp
                temp.append([primeslist1[i],exp])
                temp.append([])
                try:
                    res=hmap[i][j][str(totali%mod)]
                    for re in res:
                        temp[-1].append(re%mod)
                    

                except Exception as e:
                    j+=1
                    continue
                exp+=1
                j+=1
            if len(temp[1])>0:
                collected.append(temp)
            
            i+=1
        if g_debug > 0:
            print("[Debug 1]["+str(procnum)+"]Checking at iN: "+str(totali)+"/"+str(totali_max))
        if len(collected)>2:
            find_smooth(procnum,n,totali,primeslist1,collected,return_dict,test)  
        totali+=1
        gc.collect()
        del collected
    print("["+str(procnum)+"] Exiting")
    return 0
     
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


