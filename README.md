I just uploaded QSv3_030.py<br />
Use: python3 QSv3_030.py -keysize 50

This is a very quick first rought draft.
A lot needs to be done now. But it demonstrates smooth finding using lifting.
However right now its just lifting a coefficient mod p and checking if its smooth, which is lazy and not at all what we really want to do.
However, since this demonstrates how we can predict the exponent of the factor in the smooth candidate.... now we can really get started. 
Next we need to literally build smooths, no more bruteforce. Just query the hashmap, do some lifting mod p and be able to generate smooths without trial division of the smooth candidate.

Since we can figure out the exponent and factors now... this can be done. 

I will update the paper too once the PoC is ready.
I am very sleepy today, but I will start pushing updates regularly now. This will be much better then v2 when it's done. 

Microsoft shouldn't have fired my manager. They will pay the price. I'm never stopping. I'll destroy the world if I must.

Update: Doing some refactoring today. We should lift coefficients when we create the hashmap. Then we should look for 0 solution in the big coefficient (y<sub>0</sub>) instead of the small one like the uploaded PoC does. That then reduces the problem to finding a big modulus with small coefficient. And should throw out trial divison of smooth factors too... since if we calculate coefficients for every mod p<sub>i</sub><sup>a</sup> in the factor base.. then for any coefficients we can easily determine the factors plus the exponent
now. I'll get that done today. Then after that I need some algorithm to combine coefficients to build smooths.... if we combine two coefficients, it's either going to shrink or grow the factors that are outside the factor base.. I can see in my head how to do it now. I should have all the tools now. Anyway, I should go for a run soon. Without running I would have lost my sanity a long time ago. Ordered a bunch of new running shoes.. running over 1700km a year, it's kind of insane having to throw out running shoes every 3 months... it's so wasteful. Ofcourse, having to buy new ones, keeps revenue coming in for these companies.. so they don't care about durability.. they just need to convince people that having to buy new shoes every 3 months is a normal thing. Our world is so obviously sick, slowly destroying itself..... I will fix it with math, hahahahaha.
