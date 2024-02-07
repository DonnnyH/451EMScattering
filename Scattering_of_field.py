"""
Author: Harry Donegan
Solution of classical electromagnetic scattering problem for the interaction of laguerre-Gaussian beams interacting with a collection of point dipoles
Follows the formalism set out in:
'Cooperative resonances in light scattering from two-dimensional atomic arrays'
https://doi.org/10.1103/PhysRevLett.118.113601 (Supplemental information)
arxiv link: https://arxiv.org/abs/1610.00138
"""
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt


def GreensFunc(r1, r2, k, i, j):
    """
    Electromagnetic dyadic greens function in index notation G_{ij},
    r1 - position vector of point 1
    r2 - position vector of point 2
    k - wavevector (we always set=1)
    i, j - tensor indices G_{ij}
    """
    r = np.linalg.norm(r1 - r2)
    if r==0:
        return 0
    r_i=r1[i]
    r_j=r2[j]
    G = (-1+(3-3j*k*r)/(k**2*r**2))*(r_i*r_j)/(r**2)
    if i==j:
        G = G + ((1+(1j*k*r-1)/(k**2*r**2)))
    else:
        G=G
    return 2*np.pi*np.exp(1j*k*r)*G/(4*np.pi*r)

def GreensMatrix(r1,r2):
    """
    Electromagnetic dyadic greens function in matrix form
    r1 - position vector of point 1
    r2 - position vector of point 2
    """
    r = np.linalg.norm(r1 - r2)
    if r==0:
        return np.zeros(shape=(3,3), dtype=complex)
    else:
        G = np.exp(1j*r)/(2*r)*np.array([[(-1+(3-3j*r)/(r**2))*(r1[0]*r2[0])/(r**2)+(1+(1j*r-1)/(r**2)),(-1+(3-3j*r)/(r**2))*(r1[0]*r2[1])/(r**2), (-1+(3-3j*r)/(r**2))*(r1[0]*r2[2])/(r**2)],[(-1+(3-3j*r)/(r**2))*(r1[1]*r2[0])/(r**2),(-1+(3-3j*r)/(r**2))*(r1[1]*r2[1])/(r**2)+(1+(1j*r-1)/(r**2)),(-1+(3-3j*r)/(r**2))*(r1[1]*r2[2])/(r**2)],[(-1+(3-3j*r)/(r**2))*(r1[2]*r2[0])/(r**2),(-1+(3-3j*r)/(r**2))*(r1[2]*r2[1])/(r**2),(-1+(3-3j*r)/(r**2))*(r1[2]*r2[2])/(r**2)+(1+(1j*r-1)/(r**2))]])
        return G
    
def GreensMatrix2(r1,r2):
    """
    Electromagnetic dyadic greens function in matrix form
    r1 - position vector of point 1
    r2 - position vector of point 2
    """
    k=2*np.pi
    r = np.linalg.norm(r1 - r2)
    rel = (r1-r2)/r
    if r==0:
        return np.zeros(shape=(3,3), dtype=complex)
    else:
        return np.exp(1j*k*r)*((1/(k*r)+1j/(k*r)**2-1/(k*r)**3)*np.eye(3)-(1/(k*r)+3j/(k*r)**2-3/(k*r)**3)*np.outer(rel,rel))
    
def GreensMatrix3(r1,r2,d):
    """
    Electromagnetic dyadic greens function in matrix form
    r1 - position vector of point 1
    r2 - position vector of point 2
    """
    k=2*np.pi
    r = np.linalg.norm(r1 - r2)
    rel = (r1-r2)/r
    if r==0:
        return np.zeros(shape=(3,3), dtype=complex)
    else:
        return np.exp(1j*k*r)/(k*r)*(np.cross(np.cross(rel, d), d)+(3*np.dot(rel, d)*rel-d)*(1/(k*r)**2-1j/(k*r)))

def GreenMatrixElements(r1,r2,i,j):
    unitv1 = np.zeros(shape=(1,3), dtype=complex)
    unitv2 = np.zeros(shape=(1,3), dtype=complex)
    unitv1[i]=1
    unitv2[j]=1
    return np.dot(unitv2 GreensMatrix3(r1,r2, unitv1))
    
    
    
    

def polarisation(i, sigma):
    """
    Returns unit polarisaion vector of the E-field
    sigma = +(-)1 - Right(left)-handed circular polarisation 
    sigma = 0 - Linear polarisation in x-direction
    """
    if sigma==0:
        if (i)%3==0:
            return 1
        else:
         return 0
    elif sigma == 1:
        if (i)%3==0:
            return 1/np.sqrt(2)
        elif (i)%3==1:
            return +1j/np.sqrt(2)
        else: 
            return 0
    elif sigma == -1:
        if (i)%3==0:
            return 1/np.sqrt(2)
        elif (i)%3==1:
            return -1j/np.sqrt(2)
        else: 
            return 0

def LG_intensity(r,z,theta, l, p, w0):
    """
    Electric field pattern of a laguerre-gaussian laser beam (without polarisation)
    r - radius from beam axis
    z - distance along beam axis
    l - azimuthal index
    p - radial index
    w0 - beam waist
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
    phase = np.exp(1j*(l*theta + G(z) - 2*np.pi*r**2/(2*R(z))-2*np.pi*z))
    E=np.sqrt(2*np.math.factorial(p)/(np.pi*np.math.factorial(p+np.abs(l))))*(w0/w(z))*(np.sqrt(2)*r/w(z))**(np.abs(l))*sp.eval_genlaguerre(p,np.abs(l), (2*r**2/w(z)))*np.exp(-r**2/(w(z)**2))*phase
    return E




l=0
p=0
w0=1


def ScatteringMatrix(R):
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
        r = np.sqrt(x**2+y**2)
        theta = np.arctan2(y,x)
        E[i]=(LG_intensity(r,z,theta,l,p,w0))*polarisation(i,0)
    for i in range(3*n):
        for j in range(3*n):
            G[i][j]=GreensMatrix2(R[(i)//3], R[(j)//3])[i%3][j%3]
    M = np.eye(3*n, dtype=complex)-3j/2*G
    In = np.linalg.inv(M)
    E1 = In@E
    return np.array_split(E1,n)
 

def LG_intensity_Scat(E, R,r,z,theta, l, p, w0, x, y):
    """
    Computes the field after the scattering solution has been solved by including the induced fields of the dipoles
    """
    E0 = np.array([LG_intensity(r,z,theta, l, p, w0),0,0], dtype=complex)
    q = np.array([x,y,z])
    for i in range(len(R)):
        E0 = E0+3j/2*GreensMatrix2(R[i],q)*E[i]
    I=np.linalg.norm(E0)**2
    if I>=1:
        return 1
    return I


x=-10
y=0
z=-10

"""
Precision of grid, mxm grid
"""
m=100

pos = np.zeros(shape=(m,m), dtype=complex)


a=0.2
def square_array(N, space):
    """
    Returns the positions of 
    """
    R=[]
    a=space/2
    for i in range(N):
        for j in range(N):
            R.append(np.array([-(N*a)+2*a*i+a,-(N*a)+2*a*j+a, 0]))
    return R

def atom_plotter(N, space):
    a=space/2
    R=[]
    for i in range(N):
        R.append(np.array([0,-(N*a)+2*a*i+a]))
    return R
    

R=square_array(25, a)
Escat = ScatteringMatrix(R)



for i in range(m):
    z = -10
    for j in range(m):
        r = np.sqrt(x**2+y**2)
        theta = 0
        I=(LG_intensity_Scat(Escat, R, r,z, theta, l, p, w0, x, 0))
        pos[i][j]=I
        z = z+ 20/m
    x = x + 20/m


xlist = np.linspace(-10,10,m)
zlist = np.linspace(-10,10,m)
X, Z = np.meshgrid(zlist, xlist)


fig, ax = plt.subplots(1,1)
cp=ax.contourf(X, Z, pos, levels=100, cmap="magma", vmin=0, vmax=1)
plt.xlabel(r'$z/\lambda$', size=14)
plt.ylabel(r'$x/\lambda$', size=14)
cbar = fig.colorbar(cp)

plt.show()