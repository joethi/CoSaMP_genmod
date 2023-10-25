
import polynomial_chaos_utils as pcu


#Test functions in polynomial_chaos_utils.py
pcu.test_make_mi_mat(1,3)
pcu.test_make_mi_mat(11,3)
pcu.test_make_mi_mat(31,4)

d = 20
p = 3
mi_mat = make_mi_mat(d,p)

pcu.test_make_Psi(6,1,y_samples[0:50,:],mi_mat)
pcu.test_make_Psi(20,39,y_samples[0:50,:],mi_mat)
pcu.test_make_Psi(5,100,y_samples[0:50,:],mi_mat)
