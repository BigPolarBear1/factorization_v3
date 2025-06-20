I GOT IT!

Use: QSv3_009.py -keysize 50

Added v3_009 ... just fixes a small bug where the modulus for y<sub>0</sub> wasn't calculated correctly... next version will include transfering the result to a primefield..

Update: I am still missing something big time. Let me spent the weekend reading this paper on number field sieve and dissecting it properly. I know there is a connection, but it's a little bit more complicated then what I am imagining in my head. I'll get there very soon...

Update: Oh ok, I think I see now what went wrong. Give me a few days to correct it.... was 99.99% there. Just this last thing now. I get it now.

 Update: It really boils down to quickly finding that correct i-value without bruteforce. I'm working out some type of math using derivatives .... the problem with this vs number field sieve, is that in number field sieve we have a nice polynomial congruent to N. Here we have polynomials congruent N * i ... and unless we can figure out that i value, we cant take the correct root. But because the derivatives of both sides reveal the correct coefficient of the other side.. I suspect there is a way to do it.

 I.e in 534^2 = 1^2 + 4387 * 65, the i value is 65.

 For the lower factor the i value 13.

 I.e taking the derivatives:

 13\*41 - 534 = -1 <br/>
 13\*41 + 1 = 534 

 Ofcourse we don't know the lower factor. And we only know "1" as coefficient. However, if we choose 13 as i-value, then using the above, it is quite easy to arrive at 534 without knowledge of the lower factor.
 So that way we don't need to find a i value of 65, but 13 instead. Then the question becomes... how to quickly find that 13 without iterating 1,2,3,4,5,6,7... etc.

 I'll bash my head against it tomorrow. 

 I know I can get rid of that entire i-value problem by working with polynomial rings and all that complicated stuff number field sieve does.. but I really want to avoid it because having this i-value is more natural, other wise everything gets shifted and complicated. But I really need to overcome this issue of finding the i-value.

 I'll get some sleep.

ps: I'll have to refactor that last chapter in the paper again once this is done. It's closer now.... really getting to the root of the problem. Definitely closer then were I was a month ago. I've gained a much better understanding... it's just this last issue. It's very annoying. I do think I really need to work with derivatives going forward... none of that bullshit with squares, just linear congruences. 

pps: Oh shit, an idea just lit up in my head. I think I know how to calculate this i-value. Anyway, some sleep first.
