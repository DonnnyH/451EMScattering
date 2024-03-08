"""
Author: Harry Donegan
Solution of classical electromagnetic scattering problem for the interaction of laguerre-Gaussian beams interacting with a collection of point dipoles
Follows the formalism set out in:
'Cooperative resonances in light scattering from two-dimensional atomic arrays'
https://doi.org/10.1103/PhysRevLett.118.113601 (Supplemental information)
arxiv link: https://arxiv.org/abs/1610.00138
"""
import numpy as np
import scipy
import scipy.special as sp
import matplotlib.pyplot as plt
import time
import cmasher as cmr


def GreensFunc(r1,r2, delta):
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
        return -3/(2*(2*delta+1j))*np.exp(1j*k*r)/(k*r)*((np.eye(3)-np.outer(rel,rel))+(1j/(k*r)-1/(k*r)**2)*(np.eye(3)-3*np.outer(rel,rel)))
    
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

def ScatteringMatrix(R, l, p, w0, incident, sigma):
    """
    Calculates the self-consistent solutions of the dipole moments for each atom from the scattering equation
    """
    n = len(R)
    G = np.zeros(shape=(3*n,3*n), dtype=complex)
    E = np.zeros(shape=(3*n,1), dtype=complex)
    for i in range(3*n):
        x = R[i//3][0]
        y = R[i//3][1]
        z = R[i//3][2]
        E[i]=(LG_field(x,y,z,incident,l,p,w0, sigma))[i%3][0]
    for i in range(3*n):
        for j in range(3*n):
            G[i][j]=GreensFunc(R[(i)//3], R[(j)//3], 0)[i%3][j%3]
    M = np.eye(3*n, dtype=complex)-G
    if n==0:
        return np.array([[0],[0],[0]])
    else:
        In = np.linalg.inv(M)
        E1 = In@E
        return np.array_split(E1,n) 

def LG_field_Scat(x,y,z, incident, E, R, l, p, w0, sigma):
    """
    Computes the field after the scattering solution has been solved by including the induced fields of the dipoles
    """
    E0 = LG_field(x,y,z,incident,l,p,w0, sigma)
    q = np.array([x,y,z])
    if len(R)==0:
        E0=E0
    else:
        for i in range(len(R)):
            E0 = E0+GreensFunc(R[i],q,0)@ E[i]
    def reshaper(M):
        """
        Reshapes scattered field into correct array size
        """
        E=[0+0*0j,0+0*0j,0+0*0j]
        for i in range(len(M)):
            E[i]=M[i][0]
        return E
    return reshaper(E0)

def normalisation(l,p,w0):
    """
    Provided that the beam is a 'doughnut' LG beam (p=0) then this function normalises the beams intensity at its maximum (r=sqrt(l/2)*w0, z=0)
    """
    z_r=np.pi*w0**2
    def w(z):
        return w0*np.sqrt(1+(z/z_r)**2)
    def R(z):
        if z==0:
            return np.inf
        else:
            return z*(1+(z_r/z)**2)
    def G(z):
        return (np.abs(l)+2*p+1)*np.arctan(z/z_r)
    if p==0:
        E0=np.sqrt(2*np.math.factorial(p)/(np.pi*np.math.factorial(p+np.abs(l))))*(w0/w(0))*(np.sqrt(2)*(np.sqrt(np.abs(l))*w0/np.sqrt(2))/w(0))**(np.abs(l))*sp.eval_genlaguerre(p,np.abs(l), (2*(np.sqrt(np.abs(l))*w0/np.sqrt(2))**2/w(0)**2))*np.exp(-(np.sqrt(np.abs(l))*w0/np.sqrt(2))**2/(w(0)**2))
    else:
        E0=1
    return E0

N=20
a=0.2
R=square_array(N, a)


l=0
p=0
sigma = 0

m=150
x_bound = 5
z_bound = 10
incident = np.deg2rad(0)
w0=0.3*a*N*np.cos(incident)


pos = np.zeros(shape=(m,m), dtype=complex)

Escat = ScatteringMatrix(R, l, p, w0, incident, sigma)

cmap = cmr.amethyst
cmap1 = plt.get_cmap('cmr.amethyst')
cmap1_reversed = cmap1.reversed()


x=-x_bound
y=0
for i in range(m):
    z = -z_bound
    for j in range(m):
        I=np.linalg.norm(LG_field_Scat(x,y,z, incident, Escat, R, l, p, w0, sigma))**2/np.abs(normalisation(l, p, w0))**2
        if I>4:
            I=4
        pos[i][j]=I
        z = z+ (2*z_bound)/m
    x = x + (2*x_bound)/m


xlist = np.linspace(-x_bound,x_bound,m)
zlist = np.linspace(-z_bound,z_bound,m)
X, Z = np.meshgrid(zlist, xlist)


fig, ax = plt.subplots(1,1)
cp=ax.contourf(X, Z, pos, levels=150, cmap=cmap1_reversed)
plt.xlabel(r'$z/\lambda$', size=14)
plt.ylabel(r'$x/\lambda$', size=14)
cbar = fig.colorbar(cp)

plt.show()




    
    
    
    

    



