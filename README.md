# factorization_v3

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WORK IN PROGRESS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Useage: python3 QSv3_version_c.py -keysize 30

Just upload version_c (31 may 2025).
It's still very slow, nothing has been optimized. But this will be the foundation for future versions.
The approach is as following right now:

Given a root and coefficient x<sup>2</sup> - x*y = a  we can use a hashmap to find another root for the same coefficient such that  x<sup>2</sup> - x*y = (x<sup>2</sup> - x*y) - xN
I have to improve a lot of the parts of this code. I'll push updates regulary now. The performance ceiling for this should be fairly good once everything is completed. 
Going to take a break for the rest of the day. I'll push version_d tomorrow at the end of the day I think.
