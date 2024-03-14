import numpy as np
import scipy
import scipy.special as sp
import matplotlib.pyplot as plt
import time
import cmasher as cmr

def GreensFunc(r1,r2):
    """
    Electromagnetic dyadic Greens function in matrix form
    r1 - position vector of point 1
    r2 - position vector of point 2
    """
    k=2*np.pi
    r = np.linalg.norm(r1 - r2)
    if r==0:
        return 0j*np.eye(3)
    else:
        rel = (r1-r2)/r
        return 3/2*np.exp(1j*k*r)/(k*r)*((np.eye(3)-np.outer(rel,rel))+(1j/(k*r)-1/(k*r)**2)*(np.eye(3)-3*np.outer(rel,rel)))
    
def square_array(N, d):
    """
    Returns the positions of atoms in the XY plane in an NxN square
    d - interatomic distance in units of wavelengths
    """
    R=[]
    a=d/2
    for i in range(N):
        for j in range(N):
            R.append(np.array([-(N*a)+2*a*i+a,-(N*a)+2*a*j+a, 0]))
    return R

def Effective_Hamitonian(R, delta, delta_x, delta_y):
    n = len(R)
    G = np.zeros(shape=(3*n,3*n), dtype=complex)
    for i in range(3*n):
        for j in range(3*n):
            G[i][j]=GreensFunc(R[(i)//3], R[(j)//3])[i%3][j%3]
    for i in range(3*n):
        if i%3==0:
            G[i][i]=(delta+delta_x+1j)
        elif i%3==1:
            G[i][i]=(delta-delta_y+1j)
        elif i%3==2:
            G[i][i]=(delta+1j)
    return G

def polarisation(i, sigma, incident):
    """
    Returns unit polarisaion vector of the E-field
    sigma = +(-)1 - Right(left)-handed circular polarisation 
    sigma = 0 - Linear polarisation in x-direction
    """
    if sigma==0:
        if (i)%3==0:
            return np.cos(incident)
        elif (i)%3==2:
         return -np.sin(incident)
        else:
            return 0
    elif sigma == 1:
        if (i)%3==0:
            return 1/np.sqrt(2)*np.cos(incident)
        elif (i)%3==1:
            return +1j/np.sqrt(2)*(-np.sin(incident))
        else: 
            return 0
    elif sigma == -1:
        if (i)%3==0:
            return 1/np.sqrt(2)*np.cos(incident)
        elif (i)%3==1:
            return -1j/np.sqrt(2)*(-np.sin(incident))
        else: 
            return 0

def LG_field(x, y, z, incident, l, p, w0, sigma):
    """
    Electric field pattern of a Laguerre-Gaussian laser beam (without polarisation)
    r - radius from beam axis
    z - distance along beam axis
    l - azimuthal index
    p - radial index
    w0 - beam waist
    """
    x1 = x*np.cos(incident)-z*np.sin(incident)
    y1=y
    z1 = z*np.cos(incident)+x*np.sin(incident)
    z_r=np.pi*w0**2
    r = np.sqrt(x1**2+y1**2)
    theta = np.arctan2(y1,x1)
    def w(z):
        return w0*np.sqrt(1+(z/z_r)**2)
    def R(z):
        if z==0:
            return np.inf
        else:
            return z*(1+(z_r/z)**2)
    def G(z):
        return (np.abs(l)+2*p+1)*np.arctan(z/z_r)
    phase = np.exp(1j*(l*theta - G(z1) + 2*np.pi*r**2/(2*R(z1))+2*np.pi*z1))
    E=np.sqrt(2*np.math.factorial(p)/(np.pi*np.math.factorial(p+np.abs(l))))*(w0/w(z1))*(np.sqrt(2)*r/w(z1))**(np.abs(l))*sp.eval_genlaguerre(p,np.abs(l), (2*r**2/w(z1)**2))*np.exp(-r**2/(w(z1)**2))*phase
    E0 = E*np.array([[1],[1],[1]], dtype=complex)
    for i in range(3):
        E0[i][0]=E0[i][0]*polarisation(i, sigma, incident)
    return E0

def occupation(D, R, delta, delta_x, delta_y):
    n = len(R)
    def Effective_Hamitonian(delta, delta_x, delta_y):
        G = np.zeros(shape=(3*n,3*n), dtype=complex)
        for i in range(3*n):
            for j in range(3*n):
                G[i][j]=GreensFunc(R[(i)//3], R[(j)//3])[i%3][j%3]
        for i in range(3*n):
            if i%3==0:
                G[i][i]=(delta+delta_x+1j)
            elif i%3==1:
                G[i][i]=(delta-delta_y+1j)
            elif i%3==2:
                G[i][i]=(delta+1j)
        return G
    H = Effective_Hamitonian(delta, delta_x, delta_y)
    eigenvalues, eigenvectors = np.linalg.eig(H)
    I = []
    total_occ  = sum(np.abs(np.dot(eigenvectors[:,i], D.flatten()))**2 for i in range(3*n))
    for i in range(3*n):
        occ = np.abs(np.dot(eigenvectors[:,i], D.flatten()))**2
        I.append([occ/total_occ, np.log10(np.imag(eigenvalues[i]))])
    return I, total_occ


def ScatteringMatrix(R, l, p, w0, incident, sigma, delta, delta_x, delta_y):
    """
    Calculates the self-consistent solutions of the dipole moments for each atom from the scattering equation
    """
    n = len(R)
    E = np.zeros(shape=(3*n,1), dtype=complex)
    for i in range(3*n):
        x = R[i//3][0]
        y = R[i//3][1]
        z = R[i//3][2]
        E[i]=(LG_field(x,y,z,incident,l,p,w0, sigma))[i%3][0]
    def Effective_Hamitonian(delta, delta_x, delta_y):
        G = np.zeros(shape=(3*n,3*n), dtype=complex)
        for i in range(3*n):
            for j in range(3*n):
                G[i][j]=GreensFunc(R[(i)//3], R[(j)//3])[i%3][j%3]
        for i in range(3*n):
            if i%3==0:
                G[i][i]=(delta+delta_x+1j)
            elif i%3==1:
                G[i][i]=(delta-delta_y+1j)
            elif i%3==2:
                G[i][i]=(delta+1j)
        return G
    H = Effective_Hamitonian(delta, delta_x, delta_y)
    E1= np.linalg.solve(H, E).flatten()
    return E1


a=0.2
N=1
R = square_array(N, a) 

delta=1
delta_x=0
delta_y=0

incident = 0
l=0
p=0
sigma = 0
w0=0.3*0.2*N*np.cos(incident)

D=ScatteringMatrix(R, l, p, w0, incident, sigma, delta, delta_x, delta_y)
print(LG_field(0, 0, 0, incident, l, p, w0, sigma))
print(D)

   


        
    

