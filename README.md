Note: That last chapter in the paper is becoming a bit of a frankenstein chapter. I will first finish the PoC, then redo that entire chapter. Most of the math is already in there... its just not pretty looking and all over the place right now.

Just uploaded v3_004:

useage: python QSv3_004.py -keysize 40

Just uploading my work in progress. This one will easily factor 40 bit. Which is way too slow. But I have finally gained a good understanding.

OMG. I AM A FUCKING IDIOT. 
I get it now. Now that I understand how the correct iN value for a given coefficient is calculated.. I get it now. 
Proper PoC and fixed paper coming very soon................................................. fucking hell. That took way to long to understand something as simple as this.

The fact that these things take this long to figure out, makes me realize I really just suck at math. I don't think I'm cut out for this math stuff. Oh well... atleast factorization is about to break now hehehehe. Way too fucking slow though. 

I wish the FBI would surrender itself to me, then I don't have to drop what I'm about to drop. But can't have everything in life I guess. Some people about to have a really shit day.

Update: I'll fix it tomorrow (sunday), its too fucking warm and I can't think in this heat. I want to go be in the Arctic bc this sucks. 
What I'm currently doing wrong in this PoC... we use a static iN value for the initial small coefficient that we then incrememt... but if your small coefficient for example is: 1  and it is a quadratic residue mod N... then from every mod p<sub>i</sub> we can find an i-value that holds 1... we simply calculate the starting i (or iN) - value and indexes based on that. And then for each mod p<sub>i</sub> there will be an i-value that has a corresponding solution mapping .... it's like testing squaredness using legendre symbols... kind of. But because of how we are representing everything.. that gives us a fast way to figure out that correct i-value really quickly. Just too fucking warm, I'll go for a run and call it a day, whatever.
