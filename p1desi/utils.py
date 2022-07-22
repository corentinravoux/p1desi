import numpy as np

lambdaLy = 1215.673123130217

speed_light = 2.99792458 * 10**5

si4_1 = 1393.76
si4_2 = 1402.77
c4_1 = 1548.202
c4_2 = 1550.774

vsi4 = speed_light * np.log(si4_2 / si4_1)
lsi4=(si4_2 - si4_1)
ksi4=2*np.pi/(si4_2 - si4_1)
ksi4_speed = 2*np.pi/vsi4


vc4 = speed_light * np.log(c4_2 / c4_1)
lc4=(c4_2 - c4_1)
kc4=2*np.pi/(c4_2 - c4_1)
kc4_speed = 2 * np.pi/vc4




def return_key(dictionary,string,default_value):
    return(dictionary[string] if string in dictionary.keys() else default_value)
