Just uploaded v3_004:

useage: python QSv3_004.py -keysize 40

Just uploading my work in progress. This one will easily factor 40 bit. Which is way too slow. But I have finally gained a good understanding.

We need to find a small coefficient while maximizing the modulus and then try and find a small iN value where each solution maps to another solution.
The above code does this, but pretty poorly.
Next this will need to be replaced with linear algebra. 
My next version will do this. Pretty sure I know how to do it now.

OH!! I just went for a run, and I suddenly had the most brilliant idea! let me try to write it before I pass out, otherwise tomorrow. Waging cryptologic warfare against my enemies (the fbi and Belgian justice department) sure is exhausting. But I will triumph in the end, because no human can compete against a polar bear, hahahha.

Shit's getting too stressful. When I'm doing math, I do math 100%, I have been doing math at 100% for 2 years now. And I put massive amounts of pressure on myself to succeed at it. And honestly, right now, I can't deal with the legal harassment from Belgium and those fbi losers. I just need to get out of this place, there's plenty of countries where the US has no influence at all for me to go to.


TO DO: So the math in the paper and PoC, it's all correct, it works. We can find coefficient pairs that are congruent mod N in finite field m (when squared). And we also know the conditions it must meet to be also congruent mod N in the integers. Figuring out that part was the hardest part. That is now entirely finished. Now part 2 of the challenge begins. The question now is, how can we use these findings to construct an efficient algorithm? I'm fairly sure there is a similar method to using linear algebra as Quadratic sieve does, to combine coefficient pairs to generate a congruence mod N in the integers. I just got to find my focus again for a few days and its done. Just all the stress of the legal harassment, and unemployment for 2 years, not being able to see the friends I had known for 4 years anymore...  I feel like i am fighting an uphill battle against the entire world. Almost there now... I just now it, and people will see soon. If we can work with just coefficients, then it will be exponentially faster then quadratic sieve, and I know I'm right, I just 100% know it.
