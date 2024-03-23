import numpy as np
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
    
def GreensFunc2(r1, r2, d):
    k=2*np.pi
    r = np.linalg.norm(r1 - r2)
    if r==0:
        return 0j*np.array([0,0,0])
    else:
        rel = (r1-r2)/r
        return 3/2*np.exp(1j*k*r)/(k*r)*(np.cross(np.cross(rel, d), rel)-(3*rel*np.dot(d, rel)*(1j/(k*r)-1/(k*r)**2)))
    
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

def UnitVectors(i):
    if i%3==0:
        return 1/np.sqrt(2)*np.array([1, 1j, 0])
    elif i%3==1:
        return np.array([0,0,1])
    elif i%3==2:
        return 1/np.sqrt(2)*np.array([1,-1j,0])

def Bare_Hamiltonian(R):
    n = len(R)    
    G = np.zeros(shape=(3*n,3*n), dtype=complex)
    for i in range(3*n):
        for j in range(3*n):
            G[i][j]=np.dot(np.conjugate(UnitVectors(j)), (GreensFunc(R[i//3], R[j//3])@UnitVectors(i)))
    G = G+1j*np.eye(3*n)
    return G


def Effective_Hamitonian(R, delta, zeeman):
    n = len(R)
    G = np.zeros(shape=(3*n,3*n), dtype=complex)
    for i in range(3*n):
        for j in range(3*n):
            G[i][j]=np.dot(np.conjugate(UnitVectors(j)), (GreensFunc(R[i//3], R[j//3])@UnitVectors(i)))
    for i in range(3*n):
        if i%3==0:
            G[i][i]=(delta-zeeman+1j)
        elif i%3==1:
            G[i][i]=(delta+1j)
        elif i%3==2:
            G[i][i]=(delta+zeeman+1j)
    return G

def polarisation(i, sigma, incident):
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


def occupation(D, R):
    n = len(R)
    H = Bare_Hamiltonian(R)
    eigenvalues, eigenvectors = np.linalg.eig(H)
    for i in range(len(eigenvectors)):
        eigenvectors[:,i]=np.round(eigenvectors[:,i], 14)
    tot_occ = sum(np.abs(np.dot(eigenvectors[:,i], D))**2 for i in range(3*n))
    I1=[]
    I2=[]
    for i in range(3*n):
        occ = np.abs(np.dot(eigenvectors[:,i], D))**2
        I1.append(occ/tot_occ)
        I2.append(np.log10(np.imag(eigenvalues[i])))
    return I1, I2


def ScatteringMatrix(R, delta, zeeman):
    """
    Calculates the self-consistent solutions of the dipole moments for each atom from the scattering equation
    """
    n = len(R)
    E = np.zeros(shape=(3*n,1), dtype=complex)
    E0 = np.array([1,0,0])
    for i in range(3*n):
        E[i]=np.dot(np.conjugate(UnitVectors(i)), E0)
    H = Effective_Hamitonian(R, delta, zeeman)
    E1= np.linalg.solve(H, E).flatten()
    return E1

def ScatteringMatrix_plane(R, incident, sigma, delta, zeeman):
    """
    Calculates the self-consistent solutions of the dipole moments for each atom from the scattering equation
    """
    n = len(R)
    E = np.zeros(shape=(3*n,1), dtype=complex)
    for i in range(3*n):
        E[i]=polarisation(i%3, sigma, incident)
    H = -Effective_Hamitonian(delta, zeeman)
    E1= np.linalg.solve(H, E).flatten()
    return E1


a=0.55
N=20
R = square_array(N, a) 

delta=0.65
zeeman=1.1


start=time.time()
D = ScatteringMatrix(R, delta, zeeman)
O = occupation(D, R)
end = time.time()
print(end-start)

x = np.linspace(0,0, 3*(N**2))

N=12
R = square_array(N, a)
start1=time.time()
D1 = ScatteringMatrix(R, delta, zeeman)
O2 = occupation(D1, R)
end1 = time.time()
print(end1-start1)

plt.scatter(O[1], O[0], color='b')
plt.show()



        
    

