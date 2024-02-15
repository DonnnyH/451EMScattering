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


def GreensFunc(r1,r2):
    """
    Electromagnetic dyadic Greens function in matrix form
    r1 - position vector of point 1
    r2 - position vector of point 2
    """
    k=2*np.pi
    r = np.linalg.norm(r1 - r2)
    if r==0:
        return np.zeros(shape=(3,3), dtype=complex)
    else:
        rel = (r1-r2)/r
        return np.exp(1j*k*r)/(k*r)*((np.eye(3)-np.outer(rel,rel))+(1j/(k*r)-1/(k*r)**2)*(np.eye(3)-3*np.outer(rel,rel)))
    
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
    Electric field pattern of a Laguerre-Gaussian laser beam (without polarisation)
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
    phase = np.exp(1j*(l*theta - G(z) + 2*np.pi*r**2/(2*R(z))+2*np.pi*z))
    E=np.sqrt(2*np.math.factorial(p)/(np.pi*np.math.factorial(p+np.abs(l))))*(w0/w(z))*(np.sqrt(2)*r/w(z))**(np.abs(l))*sp.eval_genlaguerre(p,np.abs(l), (2*r**2/w(z)**2))*np.exp(-r**2/(w(z)**2))*phase
    def normalisation(l,p,w0):
        """
        Provided that the beam is a 'doughnut' LG beam (p=0) then this function normalises the beams intensity at its maximum (r=sqrt(l/2)*w0, z=0)
        """
        if p==0:
            E0=np.sqrt(2*np.math.factorial(p)/(np.pi*np.math.factorial(p+np.abs(l))))*(w0/w(0))*(np.sqrt(2)*(np.sqrt(np.abs(l))*w0/np.sqrt(2))/w(0))**(np.abs(l))*sp.eval_genlaguerre(p,np.abs(l), (2*(np.sqrt(np.abs(l))*w0/np.sqrt(2))**2/w(0)**2))*np.exp(-(np.sqrt(np.abs(l))*w0/np.sqrt(2))**2/(w(0)**2))
        else:
            E0=1
        return E0
    return E/normalisation(l,p,w0)

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

def ScatteringMatrix(R, l, p, w0):
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
            G[i][j]=GreensFunc(R[(i)//3], R[(j)//3])[i%3][j%3]
    M = np.eye(3*n, dtype=complex)-3j/2*G
    if n==0:
        return np.array([[0],[0],[0]])
    else:
        In = np.linalg.inv(M)
        E1 = In@E
        return np.array_split(E1,n)

def LG_intensity_Scat(E, R,r,z,theta, l, p, w0, x, y):
    """
    Computes the field after the scattering solution has been solved by including the induced fields of the dipoles
    """
    E0 = np.array([[LG_intensity(r,z,theta, l, p, w0)],[0],[0]], dtype=complex)
    q = np.array([x,y,z])
    if len(R)==0:
        E0=E0
    else:
        for i in range(len(R)):
            E0 = E0+3j/2*GreensFunc(R[i],q)@ E[i]
    I=np.linalg.norm(E0)**2
    if I>=1:
        I=1
    return I

def reflectivity(R, acc, l, p, w0):
    """
    Computes the reflectivity of the array under excitation by taking the quotient of the integral of the scattered field for z>0 over the solid angle of the hemisphere to the integral of the incident field for z<0 over the solid angle of the hemisphere
    To compute the integral over the spheres, we calculate the field at discrete points over the hemisphere and apply the Lebedev quadrature
    int(f(\omega), d\omega)=2pi/N*sum(f(\omega_i)),
    where N is the number of grid points
    See https://en.wikipedia.org/wiki/Lebedev_quadrature
    """
    def normvect(thet, phi):
        """
        Unit Vector of normal to sphere
        thet - azimuthal angle [0,2*pi)
        phi - polar angle [0, pi)
        """
        return np.array([[np.sin(thet)*np.cos(phi)],[np.sin(thet)*np.sin(phi)],[np.cos(thet)]])
    Escat = ScatteringMatrix(R, l, p, w0)
    
    phi=0
    rad = 1
    refl = []
    for i in range(acc):
        thet = 0.01
        for i in range(acc):
            x = rad*np.sin(thet)*np.cos(phi)
            y = rad*np.sin(thet)*np.sin(phi)
            z= rad*np.cos(thet)
            r = np.sqrt(x**2+y**2)
            theta = np.arctan2(y,x)
            refl.append(np.dot(normvect(thet, phi),LG_intensity_Scat(Escat, R, r,z, theta, l, p, w0, x, y)))
            thet = thet + (np.pi-0.01)/(2*acc)
        phi = phi + 2*np.pi/acc
    ref = np.abs(np.sum(refl))**2   
        
    phi=0
    inc = []
    for i in range(acc):
        thet = (np.pi+0.01)/2
        for i in range(acc):
            x = rad*np.sin(thet)*np.cos(phi)
            y= rad*np.sin(thet)*np.sin(phi)
            z= rad*np.cos(thet)
            r = np.sqrt(x**2+y**2)
            theta = np.arctan2(y,x)
            inc.append(np.dot(normvect(thet, phi),np.array(LG_intensity(r, z, theta, l, p, w0))))
            thet = thet + (np.pi-0.01)/(2*acc)
        phi = phi + 2*np.pi/acc
    incident =np.abs(np.sum(inc))**2
    return ref, incident
    
"""       
N=20
a=0.2
R=square_array(N, a)

l=0
p=0
w0=0.3*a*N
Escat = ScatteringMatrix(R, l, p, w0)


m=150

pos = np.zeros(shape=(m,m), dtype=complex)

x=-10
y=0
for i in range(m):
    z = -10
    for j in range(m):
        r = np.sqrt(x**2+y**2)
        theta = np.arctan2(y,x)
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
"""
"""
Z1= np.linspace(-5,-0.1, 1000)
X=0
Y=0

reflect1=[]

for i in range(1000):
        r = np.sqrt(X**2+Y**2)
        theta = np.arctan2(Y,X)
        I=(LG_intensity_Scat(Escat, R, r, Z1[i], theta, l, p, w0, X, Y))/(np.abs(LG_intensity(r, Z1[i], theta, l, p, w0))**2)
        reflect1.append(I)
        
plt.plot(Z1, reflect1, color='b')

Z2= np.linspace(0.1,5, 1000)
X=0
Y=0

reflect2=[]

for i in range(1000):
        r = np.sqrt(X**2+Y**2)
        theta = np.arctan2(Y,X)
        I=(LG_intensity_Scat(Escat, R, r, Z2[i], theta, l, p, w0, X, Y))/(np.abs(LG_intensity(r, Z2[i], theta, l, p, w0))**2)
        reflect2.append(I)
        
plt.plot(Z2, reflect2, color='b')
plt.xlabel(r'z', size=14)
plt.ylabel(r'$|E/E_{0}|^2$', size=14)
plt.title(r'20x20 atomic array with $a=0.5\lambda$ under Gaussian Beam', size=10)
plt.show()
"""

a=np.linspace(0.15,1, 25)
reflc=[]
N=20
l=0
p=0

for i in range(len(a)):
    R=square_array(N, a[i])
    w0=0.3*a[i]*N
    Escat = ScatteringMatrix(R, l, p, w0)
    x=0.2*a[i]
    y=0*a[i]
    z=2.5
    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    I=(LG_intensity_Scat(Escat, R, r, z, theta, l, p, w0, x, y))/(np.abs(LG_intensity(r, z, theta, l, p, w0))**2)
    reflc.append(I)
    
plt.plot(a,reflc)
plt.show()

    
    
    
    

    


